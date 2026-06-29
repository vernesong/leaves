package leaves

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/vernesong/leaves/transformation"
	"github.com/vernesong/leaves/util"
)

type lgEnsembleJSON struct {
	Name                 string            `json:"name"`
	Version              string            `json:"version"`
	NumClasses           int               `json:"num_class"`
	NumTreesPerIteration int               `json:"num_tree_per_iteration"`
	MaxFeatureIdx        int               `json:"max_feature_idx"`
	AverageOutput        bool              `json:"average_output"`
	Objective            string            `json:"objective"`
	LabelIndex           int               `json:"label_index"`
	Trees                []json.RawMessage `json:"tree_info"`
}

type lgTreeJSON struct {
	NumLeaves int    `json:"num_leaves"`
	NumCat    uint32 `json:"num_cat"`
	IsLinear  bool   `json:"is_linear"`
	// Unused fields:
	// TreeIndex uint32  `json:"tree_index"`
	// Shrinkage float64 `json:"shrinkage"`
	RootRaw json.RawMessage `json:"tree_structure"`
	Root    interface{}
}

// lgLeafJSON represents a leaf node in a LightGBM JSON tree.
// For linear trees, it also includes the linear model fields.
type lgLeafJSON struct {
	LeafValue    float64   `json:"leaf_value"`
	LeafConst    float64   `json:"leaf_const"`
	LeafFeatures []int     `json:"leaf_features"`
	LeafCoeff    []float64 `json:"leaf_coeff"`
}

type lgNodeJSON struct {
	SplitIndex   uint32 `json:"split_index"`
	SplitFeature uint32 `json:"split_feature"`
	// Threshold could be float64 (for numerical decision) or string (for categorical, example "10||100||400")
	Threshold     interface{}     `json:"threshold"`
	DecisionType  string          `json:"decision_type"`
	DefaultLeft   bool            `json:"default_left"`
	MissingType   string          `json:"missing_type"`
	LeftChildRaw  json.RawMessage `json:"left_child"`
	RightChildRaw json.RawMessage `json:"right_child"`
	LeftChild     interface{}
	RightChild    interface{}
}

// lgObjective keeps parsed data from 'objective' field of lightgbm txt format
// 'multiclass num_class:13' parsed to
// lgObjective{name: 'multiclass', param: 'num_class', value:13}
type lgObjective struct {
	name  string
	param string
	value int
}

func lgObjectiveParse(objective string) (lgObjective, error) {
	tokens := strings.Split(objective, " ")
	objectiveStruct := lgObjective{}
	errorMsg := fmt.Errorf("unexpected objective field: '%s'", objective)
	if len(tokens) != 2 {
		return objectiveStruct, errorMsg
	}
	objectiveStruct.name = tokens[0]
	paramTokens := strings.Split(tokens[1], ":")
	if len(paramTokens) != 2 {
		return objectiveStruct, errorMsg
	}
	objectiveStruct.param = paramTokens[0]
	value, err := strconv.Atoi(paramTokens[1])
	if err != nil {
		return objectiveStruct, errorMsg
	}
	objectiveStruct.value = value
	return objectiveStruct, nil
}

func convertMissingType(decisionType uint32) (uint8, error) {
	missingTypeOrig := (decisionType >> 2) & 3
	missingType := uint8(0)
	if missingTypeOrig == 0 {
		// default value
	} else if missingTypeOrig == 1 {
		missingType = missingZero
	} else if missingTypeOrig == 2 {
		missingType = missingNan
	} else {
		return 0, fmt.Errorf("unknown missing type = %d", missingTypeOrig)
	}
	return missingType, nil
}

var stringToMissingType = map[string]uint8{
	"None": 0,
	"Zero": missingZero,
	"NaN":  missingNan,
}

func lgTreeFromReader(reader *bufio.Reader) (lgTree, error) {
	t := lgTree{}
	params, err := util.ReadParamsUntilBlank(reader)
	if err != nil {
		return t, err
	}
	numCategorical, err := params.ToInt("num_cat")
	if err != nil {
		return t, err
	}
	t.nCategorical = uint32(numCategorical)

	numLeaves, err := params.ToInt("num_leaves")
	if err != nil {
		return t, err
	}
	if numLeaves < 1 {
		return t, fmt.Errorf("num_leaves < 1")
	}

	leafValues, err := params.ToFloat64Slice("leaf_value")
	if err != nil {
		return t, err
	}
	t.leafValues = leafValues

	// check for linear tree (LightGBM v4+)
	if params.Contains("is_linear") {
		isLinear, err := params.ToInt("is_linear")
		if err != nil {
			return t, err
		}
		t.isLinear = isLinear == 1
	}

	// parse linear tree fields if present
	if t.isLinear {
		leafConsts, err := params.ToFloat64Slice("leaf_const")
		if err != nil {
			return t, err
		}
		t.leafConsts = leafConsts

		numFeatures, err := params.ToIntSlice("num_features")
		if err != nil {
			return t, err
		}
		if len(numFeatures) != numLeaves {
			return t, fmt.Errorf("num_features length (%d) != num_leaves (%d)", len(numFeatures), numLeaves)
		}

		t.leafFeatures = make([][]int, numLeaves)
		t.leafCoeffs = make([][]float64, numLeaves)

		// read flat leaf_features and leaf_coeff arrays
		var allFeatures []int
		var allCoeffs []float64
		totalFeatureCount := 0
		for _, nf := range numFeatures {
			totalFeatureCount += nf
		}
		if totalFeatureCount > 0 {
			allFeatures, err = params.ToIntSlice("leaf_features")
			if err != nil {
				return t, err
			}
			allCoeffs, err = params.ToFloat64Slice("leaf_coeff")
			if err != nil {
				return t, err
			}
		}

		// distribute flat arrays to per-leaf slices
		offset := 0
		for i, nf := range numFeatures {
			if nf > 0 {
				t.leafFeatures[i] = allFeatures[offset : offset+nf]
				t.leafCoeffs[i] = allCoeffs[offset : offset+nf]
				offset += nf
			}
		}
	}

	if numLeaves == 1 {
		// special case - constant value tree (linear trees with 1 leaf already handled above)
		return t, nil
	}

	leftChilds, err := params.ToInt32Slice("left_child")
	if err != nil {
		return t, err
	}
	rightChilds, err := params.ToInt32Slice("right_child")
	if err != nil {
		return t, err
	}
	decisionTypes, err := params.ToUint32Slice("decision_type")
	if err != nil {
		return t, err
	}
	splitFeatures, err := params.ToUint32Slice("split_feature")
	if err != nil {
		return t, err
	}
	thresholds, err := params.ToFloat64Slice("threshold")
	if err != nil {
		return t, err
	}

	catThresholds := make([]uint32, 0)
	catBoundaries := make([]uint32, 0)
	if numCategorical > 0 {
		// first element set to zero for consistency
		t.catBoundaries = make([]uint32, 1)
		catThresholds, err = params.ToUint32Slice("cat_threshold")
		if err != nil {
			return t, err
		}
		catBoundaries, err = params.ToUint32Slice("cat_boundaries")
		if err != nil {
			return t, err
		}
	}

	createNumericalNode := func(idx int32) (lgNode, error) {
		node := lgNode{}
		missingType, err := convertMissingType(decisionTypes[idx])
		if err != nil {
			return node, err
		}
		defaultType := uint8(0)
		if decisionTypes[idx]&(1<<1) > 0 {
			defaultType = defaultLeft
		}
		node = numericalNode(splitFeatures[idx], missingType, thresholds[idx], defaultType)
		if leftChilds[idx] < 0 {
			node.Flags |= leftLeaf
			node.Left = uint32(^leftChilds[idx])
		}
		if rightChilds[idx] < 0 {
			node.Flags |= rightLeaf
			node.Right = uint32(^rightChilds[idx])
		}
		return node, nil
	}

	createCategoricalNode := func(idx int32) (lgNode, error) {
		node := lgNode{}
		missingType, err := convertMissingType(decisionTypes[idx])
		if err != nil {
			return node, err
		}

		catIdx := uint32(thresholds[idx])
		catType := uint8(0)
		bitsetSize := catBoundaries[catIdx+1] - catBoundaries[catIdx]
		thresholdSlice := catThresholds[catBoundaries[catIdx]:catBoundaries[catIdx+1]]
		nBits := util.NumberOfSetBits(thresholdSlice)
		if nBits == 0 {
			return node, fmt.Errorf("no bits set")
		} else if nBits == 1 {
			i, err := util.FirstNonZeroBit(thresholdSlice)
			if err != nil {
				return node, fmt.Errorf("not reached error")
			}
			catIdx = i
			catType = catOneHot
		} else if bitsetSize == 1 {
			catIdx = catThresholds[catBoundaries[catIdx]]
			catType = catSmall
		} else {
			// regular case with large bitset
			catIdx = uint32(len(t.catBoundaries) - 1)
			t.catThresholds = append(t.catThresholds, thresholdSlice...)
			t.catBoundaries = append(t.catBoundaries, uint32(len(t.catThresholds)))
		}

		node = categoricalNode(splitFeatures[idx], missingType, catIdx, catType)
		if leftChilds[idx] < 0 {
			node.Flags |= leftLeaf
			node.Left = uint32(^leftChilds[idx])
		}
		if rightChilds[idx] < 0 {
			node.Flags |= rightLeaf
			node.Right = uint32(^rightChilds[idx])
		}
		return node, nil
	}
	createNode := func(idx int32) (lgNode, error) {
		if decisionTypes[idx]&1 > 0 {
			return createCategoricalNode(idx)
		}
		return createNumericalNode(idx)
	}
	numNodes := numLeaves - 1
	origNodeIdxStack := make([]uint32, 0, numNodes)
	convNodeIdxStack := make([]uint32, 0, numNodes)
	visited := make([]bool, numNodes)
	t.nodes = make([]lgNode, 0, numNodes)
	node, err := createNode(0)
	if err != nil {
		return t, err
	}
	t.nodes = append(t.nodes, node)
	origNodeIdxStack = append(origNodeIdxStack, 0)
	convNodeIdxStack = append(convNodeIdxStack, 0)
	for len(origNodeIdxStack) > 0 {
		convIdx := convNodeIdxStack[len(convNodeIdxStack)-1]
		if t.nodes[convIdx].Flags&rightLeaf == 0 {
			origIdx := rightChilds[origNodeIdxStack[len(origNodeIdxStack)-1]]
			if !visited[origIdx] {
				node, err := createNode(origIdx)
				if err != nil {
					return t, err
				}
				t.nodes = append(t.nodes, node)
				convNewIdx := len(t.nodes) - 1
				convNodeIdxStack = append(convNodeIdxStack, uint32(convNewIdx))
				origNodeIdxStack = append(origNodeIdxStack, uint32(origIdx))
				visited[origIdx] = true
				t.nodes[convIdx].Right = uint32(convNewIdx)
				continue
			}
		}
		if t.nodes[convIdx].Flags&leftLeaf == 0 {
			origIdx := leftChilds[origNodeIdxStack[len(origNodeIdxStack)-1]]
			if !visited[origIdx] {
				node, err := createNode(origIdx)
				if err != nil {
					return t, err
				}
				t.nodes = append(t.nodes, node)
				convNewIdx := len(t.nodes) - 1
				convNodeIdxStack = append(convNodeIdxStack, uint32(convNewIdx))
				origNodeIdxStack = append(origNodeIdxStack, uint32(origIdx))
				visited[origIdx] = true
				t.nodes[convIdx].Left = uint32(convNewIdx)
				continue
			}
		}
		origNodeIdxStack = origNodeIdxStack[:len(origNodeIdxStack)-1]
		convNodeIdxStack = convNodeIdxStack[:len(convNodeIdxStack)-1]
	}
	return t, nil
}

// LGEnsembleFromReader reads LightGBM model from `reader`
func LGEnsembleFromReader(reader *bufio.Reader, loadTransformation bool) (*Ensemble, error) {
	e := &lgEnsemble{name: "lightgbm.gbdt"}

	params, err := util.ReadParamsUntilBlank(reader)
	if err != nil {
		return nil, err
	}

	if err := params.CompareAll("version", []string{"v2", "v3", "v4"}); err != nil {
		return nil, err
	}
	nClasses, err := params.ToInt("num_class")
	if err != nil {
		return nil, err
	}
	nTreePerIteration, err := params.ToInt("num_tree_per_iteration")
	if err != nil {
		return nil, err
	}
	if nClasses != nTreePerIteration {
		return nil, fmt.Errorf("meet case when num_class (%d) != num_tree_per_iteration (%d)", nClasses, nTreePerIteration)
	} else if nClasses < 1 {
		return nil, fmt.Errorf("num_class (%d) should be > 0", nClasses)
	} else if nTreePerIteration < 1 {
		return nil, fmt.Errorf("num_tree_per_iteration (%d) should be > 0", nTreePerIteration)
	}
	e.nRawOutputGroups = nClasses

	maxFeatureIdx, err := params.ToInt("max_feature_idx")
	if err != nil {
		return nil, err
	}
	e.MaxFeatureIdx = maxFeatureIdx

	if params.Contains("average_output") {
		e.name = "lightgbm.rf"
		e.averageOutput = true
	}

	treeSizesStr, isFound := params["tree_sizes"]
	if !isFound {
		return nil, fmt.Errorf("no tree_sizes field")
	}
	treeSizes := strings.Split(treeSizesStr, " ")

	// NOTE: we rely on the fact that size of tree_sizes data is equal to number of trees
	nTrees := len(treeSizes)
	if nTrees == 0 {
		return nil, fmt.Errorf("no trees in file (based on tree_sizes value)")
	} else if nTrees%e.nRawOutputGroups != 0 {
		return nil, fmt.Errorf("wrong number of trees (%d) for number of class (%d)", nTrees, e.nRawOutputGroups)
	}

	var transform transformation.Transform
	transform = &transformation.TransformRaw{e.nRawOutputGroups}
	// NOTE: it seems that we don't nee to apply transformation to random forest models
	// TODO: check it
	if loadTransformation && !e.averageOutput {
		objectiveStr, err := params.ToString("objective")
		if err != nil {
			return nil, err
		}

		if objectiveStr == "poisson" || objectiveStr == "gamma" || objectiveStr == "tweedie" {
			transform = &transformation.TransformExponential{}
		} else if !strings.HasPrefix(objectiveStr, "regression") { // no transformation for regression
			objectiveStruct, err := lgObjectiveParse(objectiveStr)
			if err != nil {
				return nil, err
			}
			if objectiveStruct.name == "binary" && objectiveStruct.param == "sigmoid" {
				if objectiveStruct.value != 1 {
					return nil, fmt.Errorf("got sigmoid with value != 1 (got %d)", objectiveStruct.value)
				}
				transform = &transformation.TransformLogistic{}
			} else if objectiveStruct.name == "multiclass" && objectiveStruct.param == "num_class" {
				if objectiveStruct.value != e.nRawOutputGroups {
					return nil, fmt.Errorf("got multiclass num_class != %d (got %d)", e.nRawOutputGroups, objectiveStruct.value)
				}
				transform = &transformation.TransformSoftmax{objectiveStruct.value}
				// multiclass num_class:13
			} else {
				return nil, fmt.Errorf("unknown transformation function '%s'", objectiveStr)
			}
		}
	}

	e.Trees = make([]lgTree, 0, nTrees)
	for i := 0; i < nTrees; i++ {
		tree, err := lgTreeFromReader(reader)
		if err != nil {
			return nil, fmt.Errorf("error while reading %d tree: %s", i, err.Error())
		}
		e.Trees = append(e.Trees, tree)
	}
	return &Ensemble{e, transform}, nil
}

// LGEnsembleFromFile reads LightGBM model from binary file
func LGEnsembleFromFile(filename string, loadTransformation bool) (*Ensemble, error) {
	reader, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer reader.Close()
	bufReader := bufio.NewReader(reader)
	return LGEnsembleFromReader(bufReader, loadTransformation)
}

// unmarshalNode recursively unmarshal nodes data in the tree from JSON raw data. Tree's node can be:
// 1. leaf node (returns *lgLeafJSON)
// 2. inner node with decision rule (returns *lgNodeJSON)
func unmarshalNode(raw []byte) (interface{}, error) {
	node := &lgNodeJSON{}
	err := json.Unmarshal(raw, node)
	if err != nil {
		return nil, err
	}

	// check if this is a leaf node (no decision fields) or an inner node
	if node.MissingType == "" {
		// this is a leaf node - parse as lgLeafJSON
		leaf := &lgLeafJSON{}
		err = json.Unmarshal(raw, leaf)
		if err != nil {
			return nil, err
		}
		return leaf, nil
	}
	// inner node with decision rule
	node.LeftChild, err = unmarshalNode(node.LeftChildRaw)
	if err != nil {
		return nil, err
	}
	node.RightChild, err = unmarshalNode(node.RightChildRaw)
	if err != nil {
		return nil, err
	}
	return node, nil
}

// unmarshalTree unmarshal tree data from JSON raw data and convert it to `lgTree` structure
func unmarshalTree(raw []byte) (lgTree, error) {
	t := lgTree{}

	treeJSON := &lgTreeJSON{}
	err := json.Unmarshal(raw, treeJSON)
	if err != nil {
		return t, err
	}

	t.isLinear = treeJSON.IsLinear
	t.nCategorical = treeJSON.NumCat
	if t.nCategorical > 0 {
		// first element set to zero for consistency
		t.catBoundaries = make([]uint32, 1)
	}

	if treeJSON.NumLeaves < 1 {
		return t, fmt.Errorf("num_leaves < 1")
	}
	numNodes := treeJSON.NumLeaves - 1

	treeJSON.Root, err = unmarshalNode(treeJSON.RootRaw)
	if err != nil {
		return t, err
	}

	// Detect linear tree by checking for "leaf_const" in the raw JSON.
	// LightGBM JSON ToJSON() does not output an explicit "is_linear" field,
	// but linear trees always have "leaf_const" in leaf nodes.
	if !t.isLinear && strings.Contains(string(raw), `"leaf_const"`) {
		t.isLinear = true
	}

	// helper to add a leaf and return its index in leafValues
	addLeaf := func(leaf *lgLeafJSON) uint32 {
		idx := uint32(len(t.leafValues))
		t.leafValues = append(t.leafValues, leaf.LeafValue)
		if t.isLinear {
			t.leafConsts = append(t.leafConsts, leaf.LeafConst)
			t.leafFeatures = append(t.leafFeatures, leaf.LeafFeatures)
			t.leafCoeffs = append(t.leafCoeffs, leaf.LeafCoeff)
		}
		return idx
	}

	if leaf, ok := treeJSON.Root.(*lgLeafJSON); ok {
		// special case - constant value tree (1 leaf)
		addLeaf(leaf)
		return t, nil
	}

	createNumericalNode := func(nodeJSON *lgNodeJSON) (lgNode, error) {
		node := lgNode{}
		missingType, isFound := stringToMissingType[nodeJSON.MissingType]
		if !isFound {
			return node, fmt.Errorf("unknown missing_type '%s'", nodeJSON.MissingType)
		}
		defaultType := uint8(0)
		if nodeJSON.DefaultLeft {
			defaultType = defaultLeft
		}
		threshold, ok := nodeJSON.Threshold.(float64)
		if !ok {
			return node, fmt.Errorf("unexpected Threshold type %T", nodeJSON.Threshold)
		}
		node = numericalNode(nodeJSON.SplitFeature, missingType, threshold, defaultType)
		if leaf, ok := nodeJSON.LeftChild.(*lgLeafJSON); ok {
			node.Flags |= leftLeaf
			node.Left = addLeaf(leaf)
		}
		if leaf, ok := nodeJSON.RightChild.(*lgLeafJSON); ok {
			node.Flags |= rightLeaf
			node.Right = addLeaf(leaf)
		}
		return node, nil
	}

	createCategoricalNode := func(nodeJSON *lgNodeJSON) (lgNode, error) {
		node := lgNode{}
		missingType, isFound := stringToMissingType[nodeJSON.MissingType]
		if !isFound {
			return node, fmt.Errorf("unknown missing_type '%s'", nodeJSON.MissingType)
		}

		thresholdString, ok := nodeJSON.Threshold.(string)
		if !ok {
			return node, fmt.Errorf("unexpected Threshold type %T", nodeJSON.Threshold)
		}
		tokens := strings.Split(thresholdString, "||")

		nBits := len(tokens)
		catIdx := uint32(0)
		catType := uint8(0)
		if nBits == 0 {
			return node, fmt.Errorf("no bits set")
		} else if nBits == 1 {
			value, err := strconv.Atoi(tokens[0])
			if err != nil {
				return node, fmt.Errorf("can't convert %s: %s", tokens[0], err.Error())
			}
			catIdx = uint32(value)
			catType = catOneHot
		} else {
			thresholdValues := make([]int, len(tokens))
			for i, valueStr := range tokens {
				value, err := strconv.Atoi(valueStr)
				if err != nil {
					return node, fmt.Errorf("can't convert %s: %s", valueStr, err.Error())
				}
				thresholdValues[i] = value
			}

			bitset := util.ConstructBitset(thresholdValues)
			if len(bitset) == 1 {
				catIdx = bitset[0]
				catType = catSmall
			} else {
				// regular case with large bitset
				catIdx = uint32(len(t.catBoundaries) - 1)
				t.catThresholds = append(t.catThresholds, bitset...)
				t.catBoundaries = append(t.catBoundaries, uint32(len(t.catThresholds)))
			}
		}

		node = categoricalNode(nodeJSON.SplitFeature, missingType, catIdx, catType)
		if leaf, ok := nodeJSON.LeftChild.(*lgLeafJSON); ok {
			node.Flags |= leftLeaf
			node.Left = addLeaf(leaf)
		}
		if leaf, ok := nodeJSON.RightChild.(*lgLeafJSON); ok {
			node.Flags |= rightLeaf
			node.Right = addLeaf(leaf)
		}
		return node, nil
	}
	createNode := func(nodeJSON *lgNodeJSON) (lgNode, error) {
		if nodeJSON.DecisionType == "==" {
			return createCategoricalNode(nodeJSON)
		} else if nodeJSON.DecisionType == "<=" {
			return createNumericalNode(nodeJSON)
		} else {
			return lgNode{}, fmt.Errorf("unknown decision type '%s'", nodeJSON.DecisionType)
		}
	}

	type StackData struct {
		// pointer to parent's Left/RightChild field
		parentPtr *uint32
		nodeJSON  *lgNodeJSON
	}
	stack := make([]StackData, 0, numNodes)
	if root, ok := treeJSON.Root.(*lgNodeJSON); ok {
		stack = append(stack, StackData{nil, root})
	} else {
		return t, fmt.Errorf("unexpected type of Root: %T", treeJSON.Root)
	}
	// NOTE: we rely on fact that t.nodes won't be reallocated (`parentPtr` points to its data)
	t.nodes = make([]lgNode, 0, numNodes)

	for len(stack) > 0 {
		stackData := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		node, err := createNode(stackData.nodeJSON)
		if err != nil {
			return t, err
		}
		if stackData.parentPtr != nil {
			*stackData.parentPtr = uint32(len(t.nodes))
		}
		t.nodes = append(t.nodes, node)
		if node.Flags&leftLeaf == 0 {
			if left, ok := stackData.nodeJSON.LeftChild.(*lgNodeJSON); ok {
				stack = append(stack, StackData{&t.nodes[len(t.nodes)-1].Left, left})
			} else if _, ok := stackData.nodeJSON.LeftChild.(*lgLeafJSON); ok {
				// left child is a leaf, already handled in createNode
			} else {
				return t, fmt.Errorf("unexpected left child type %T", stackData.nodeJSON.LeftChild)
			}
		}
		if node.Flags&rightLeaf == 0 {
			if right, ok := stackData.nodeJSON.RightChild.(*lgNodeJSON); ok {
				stack = append(stack, StackData{&t.nodes[len(t.nodes)-1].Right, right})
			} else if _, ok := stackData.nodeJSON.RightChild.(*lgLeafJSON); ok {
				// right child is a leaf, already handled in createNode
			} else {
				return t, fmt.Errorf("unexpected right child type %T", stackData.nodeJSON.RightChild)
			}
		}
	}
	return t, nil
}

// LGEnsembleFromJSON reads LightGBM model from stream with JSON data
func LGEnsembleFromJSON(reader io.Reader, loadTransformation bool) (*Ensemble, error) {
	data := &lgEnsembleJSON{}

	dec := json.NewDecoder(reader)

	err := dec.Decode(data)
	if err != nil {
		return nil, err
	}

	e := &lgEnsemble{name: "lightgbm.gbdt"}

	if data.Name != "tree" {
		return nil, fmt.Errorf("expected 'name' field = 'tree' (got: '%s')", data.Name)
	}

	// support JSON model versions v2, v3, v4 (LightGBM 3.x+ uses v3/v4)
	if data.Version != "v2" && data.Version != "v3" && data.Version != "v4" {
		return nil, fmt.Errorf("expected 'version' field = 'v2', 'v3', or 'v4' (got: '%s')", data.Version)
	}

	if data.NumClasses != data.NumTreesPerIteration {
		return nil, fmt.Errorf(
			"meet case when num_class (%d) != num_tree_per_iteration (%d)",
			data.NumClasses,
			data.NumTreesPerIteration,
		)
	} else if data.NumClasses < 1 {
		return nil, fmt.Errorf("num_class (%d) should be > 0", data.NumClasses)
	} else if data.NumTreesPerIteration < 1 {
		return nil, fmt.Errorf("num_tree_per_iteration (%d) should be > 0", data.NumTreesPerIteration)
	}
	e.nRawOutputGroups = data.NumClasses
	e.MaxFeatureIdx = data.MaxFeatureIdx

	// handle random forest models
	if data.AverageOutput {
		e.name = "lightgbm.rf"
		e.averageOutput = true
	}

	// determine transformation function from objective
	var transform transformation.Transform
	transform = &transformation.TransformRaw{e.nRawOutputGroups}
	if loadTransformation && !e.averageOutput && data.Objective != "" {
		objectiveStr := data.Objective

		if objectiveStr == "poisson" || objectiveStr == "gamma" || objectiveStr == "tweedie" {
			transform = &transformation.TransformExponential{}
		} else if !strings.HasPrefix(objectiveStr, "regression") {
			objectiveStruct, err := lgObjectiveParse(objectiveStr)
			if err != nil {
				// for unknown objective, use raw output
				transform = &transformation.TransformRaw{e.nRawOutputGroups}
			} else if objectiveStruct.name == "binary" && objectiveStruct.param == "sigmoid" {
				if objectiveStruct.value != 1 {
					return nil, fmt.Errorf("got sigmoid with value != 1 (got %d)", objectiveStruct.value)
				}
				transform = &transformation.TransformLogistic{}
			} else if objectiveStruct.name == "multiclass" && objectiveStruct.param == "num_class" {
				if objectiveStruct.value != e.nRawOutputGroups {
					return nil, fmt.Errorf("got multiclass num_class != %d (got %d)", e.nRawOutputGroups, objectiveStruct.value)
				}
				transform = &transformation.TransformSoftmax{objectiveStruct.value}
			}
			// for other unknown objectives, raw output is used by default
		}
	}

	nTrees := len(data.Trees)
	if nTrees == 0 {
		return nil, fmt.Errorf("no trees in file (based on tree_sizes value)")
	} else if nTrees%e.nRawOutputGroups != 0 {
		return nil, fmt.Errorf("wrong number of trees (%d) for number of class (%d)", nTrees, e.nRawOutputGroups)
	}

	e.Trees = make([]lgTree, 0, nTrees)
	for i := 0; i < nTrees; i++ {
		tree, err := unmarshalTree(data.Trees[i])
		if err != nil {
			return nil, fmt.Errorf("error while reading %d tree: %s", i, err.Error())
		}
		e.Trees = append(e.Trees, tree)
	}
	return &Ensemble{e, transform}, nil
}
