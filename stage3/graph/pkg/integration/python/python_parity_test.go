//go:build python

package python_test

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"testing"

	graphcli "github.com/langchain-ai/langgraph-go/pkg/cli"
)

// parityExample mirrors the simple double-add graph used in the Python examples.
var parityConfig = graphcli.GraphConfig{
	Entry: "double_add",
	Exit:  "double_add",
	Nodes: []graphcli.GraphNode{
		{ID: "add_one", Op: "add", Args: []any{1}},
		{ID: "double_add", Op: "add", Args: []any{1}},
	},
	Edges: []graphcli.GraphEdge{{From: "add_one", To: "double_add"}},
}

func TestGoMatchesPythonLangGraph(t *testing.T) {
	cfg := graphcli.ProjectConfig{Graph: parityConfig, InitialInput: 1}
	compiled, err := graphcli.BuildGraphFromConfig(cfg.Graph, nil)
	if err != nil {
		t.Fatalf("build graph: %v", err)
	}
	goOut, err := compiled.Invoke(testContext(t), cfg.InitialInput)
	if err != nil {
		t.Fatalf("go invoke: %v", err)
	}

	script := `import json\nfrom langgraph.graph import StateGraph\nclass State(dict): pass\ngraph = StateGraph(dict)\n
def add_one(value):\n    return value + 1\n\n
graph.add_node("add_one", lambda state: {"value": add_one(state["value"])})\n
def double_add(state):\n    return {"value": state["value"] + 1}\n\n
graph.add_node("double_add", double_add)\n\ngraph.add_edge("add_one", "double_add")\n\ngraph.set_entry_point("add_one")\ngraph.set_finish_point("double_add")\nres = graph.compile().invoke({"value": 1})\nprint(json.dumps(res))\n`
	cmd := exec.Command("python3", "-c", script)
	cmd.Env = append(os.Environ(), "PYTHONPATH="+thirdPartyPath())
	var out bytes.Buffer
	cmd.Stdout = &out
	stderr := new(bytes.Buffer)
	cmd.Stderr = stderr
	if err := cmd.Run(); err != nil {
		t.Skipf("python dependencies missing: %v\n%s", err, stderr)
	}
	var pyRes map[string]any
	if err := json.Unmarshal(out.Bytes(), &pyRes); err != nil {
		t.Fatalf("decode python output: %v", err)
	}
	if got, want := goOut.(float64), pyRes["value"]; want != got {
		t.Fatalf("mismatch: go=%v python=%v", got, want)
	}
}

func thirdPartyPath() string {
	if p := os.Getenv("LANGGRAPH_PYTHONPATH"); p != "" {
		return p
	}
	return "third_party/langgraph/libs/langgraph"
}

func testContext(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	return ctx
}
