package maths

import (
	"fmt"

	"github.com/langchain-ai/langgraph-go/pkg/stategraph"
)

// EdgeSpec describes a directed connection between two nodes for helper builders.
type EdgeSpec struct {
	From  string
	To    string
	Label string
}

// BuildGraph constructs a compiled state graph using the supplied node handlers and edges.
// Nodes must be provided in the map by identifier. The entry parameter selects the starting node,
// while exit optionally marks the node whose result should be returned.
func BuildGraph(entry, exit string, nodes map[string]stategraph.NodeFunc, edges []EdgeSpec) (*stategraph.CompiledStateGraph, error) {
	if len(nodes) == 0 {
		return nil, fmt.Errorf("maths graph builder: at least one node must be provided")
	}
	builder := stategraph.New()
	for id, handler := range nodes {
		if err := builder.AddNode(id, handler); err != nil {
			return nil, err
		}
	}
	for _, edge := range edges {
		if err := builder.AddEdge(edge.From, edge.To, stategraph.WithEdgeLabel(edge.Label)); err != nil {
			return nil, err
		}
	}
	if entry != "" {
		builder.SetEntryPoint(entry)
	}
	if exit != "" {
		builder.SetFinishPoint(exit)
	}
	return builder.Compile()
}

// NewDotGraph creates a single-node pipeline that runs the DotNode and returns its result.
func NewDotGraph(engine *Engine) (*stategraph.CompiledStateGraph, error) {
	nodes := map[string]stategraph.NodeFunc{"dot": DotNode(engine)}
	return BuildGraph("dot", "dot", nodes, nil)
}

// NewMatMulGraph creates a single-node pipeline executing the MatMulNode.
func NewMatMulGraph(engine *Engine) (*stategraph.CompiledStateGraph, error) {
	nodes := map[string]stategraph.NodeFunc{"matmul": MatMulNode(engine)}
	return BuildGraph("matmul", "matmul", nodes, nil)
}

// NewCosineTopKGraph creates a single-node pipeline executing the CosineTopK node.
func NewCosineTopKGraph(engine *Engine) (*stategraph.CompiledStateGraph, error) {
	nodes := map[string]stategraph.NodeFunc{"cosine_topk": CosineTopKNode(engine)}
	return BuildGraph("cosine_topk", "cosine_topk", nodes, nil)
}
