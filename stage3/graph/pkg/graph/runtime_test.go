package graph_test

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	_ "github.com/mattn/go-sqlite3"

	sqlitestore "github.com/langchain-ai/langgraph-go/pkg/checkpoint/sqlite"
	"github.com/langchain-ai/langgraph-go/pkg/graph"
)

func TestGraphRunSequential(t *testing.T) {
	g := graph.New()

	g.RegisterNode("start", func(ctx context.Context, input any) (any, error) {
		val := input.(int)
		return val + 1, nil
	})
	g.RegisterNode("middle", func(ctx context.Context, input any) (any, error) {
		val := input.(int)
		return val * 2, nil
	})
	g.RegisterNode("end", func(ctx context.Context, input any) (any, error) {
		val := input.(int)
		return val - 3, nil
	})

	g.Connect("start", "middle")
	g.Connect("middle", "end")
	g.SetEntry("start")
	g.SetExit("end")

	out, err := g.Run(context.Background(), 5)
	if err != nil {
		t.Fatalf("Run returned error: %v", err)
	}

	if want := ((5 + 1) * 2) - 3; out != want {
		t.Fatalf("unexpected result: got %v want %v", out, want)
	}
}

func TestGraphRunContextCanceled(t *testing.T) {
	g := graph.New()
	g.RegisterNode("start", func(ctx context.Context, input any) (any, error) {
		return input, nil
	})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := g.Run(ctx, nil); err == nil {
		t.Fatal("expected context cancellation error")
	}
}

func TestGraphRunMissingEntry(t *testing.T) {
	g := graph.New()
	if _, err := g.Run(context.Background(), nil); err == nil {
		t.Fatal("expected error when entry node not set")
	}
}

func TestGraphPersistsOutputs(t *testing.T) {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("failed to open sqlite db: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })

	store, err := sqlitestore.NewStore(db)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	state := graph.NewStateManager(store)

	g := graph.New()
	g.UseStateManager(state)

	type payload struct {
		Count int `json:"count"`
	}

	g.RegisterNode("start", func(ctx context.Context, input any) (any, error) {
		val := input.(int)
		return payload{Count: val + 1}, nil
	})
	g.RegisterNode("end", func(ctx context.Context, input any) (any, error) {
		p := input.(payload)
		return payload{Count: p.Count * 2}, nil
	})
	g.Connect("start", "end")
	g.SetEntry("start")
	g.SetExit("end")

	out, err := g.Run(context.Background(), 5)
	if err != nil {
		t.Fatalf("Run returned error: %v", err)
	}
	p := out.(payload)
	if p.Count != 12 {
		t.Fatalf("unexpected result: %v", p)
	}

	var persisted payload
	ok, err := state.Load(context.Background(), graph.NodeID("end"), &persisted)
	if err != nil {
		t.Fatalf("state.Load failed: %v", err)
	}
	if !ok {
		t.Fatalf("expected persisted payload")
	}
	if persisted.Count != p.Count {
		t.Fatalf("checkpoint payload mismatch: got %d want %d", persisted.Count, p.Count)
	}
}

func TestGraphResumeSkipsExecutedNodes(t *testing.T) {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("failed to open sqlite db: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })

	store, err := sqlitestore.NewStore(db)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	state := graph.NewStateManager(store)

	callCounts := map[string]int{}

	g := graph.New()
	g.UseStateManager(state)

	g.RegisterNode("start", func(ctx context.Context, input any) (any, error) {
		callCounts["start"]++
		return input.(int) + 1, nil
	})
	g.RegisterNode("end", func(ctx context.Context, input any) (any, error) {
		callCounts["end"]++
		return input.(int) * 2, nil
	})
	g.Connect("start", "end")
	g.SetEntry("start")
	g.SetExit("end")

	out, err := g.Run(context.Background(), 3)
	if err != nil {
		t.Fatalf("initial run returned error: %v", err)
	}
	if out.(int) != (3+1)*2 {
		t.Fatalf("unexpected initial result: %v", out)
	}
	if callCounts["start"] != 1 || callCounts["end"] != 1 {
		t.Fatalf("unexpected call counts after first run: %+v", callCounts)
	}

	// Second run should reuse persisted results and skip handlers.
	out, err = g.Run(context.Background(), 3, graph.WithResume())
	if err != nil {
		t.Fatalf("resume run returned error: %v", err)
	}
	if out.(int) != (3+1)*2 {
		t.Fatalf("unexpected resume result: %v", out)
	}
	if callCounts["start"] != 1 || callCounts["end"] != 1 {
		t.Fatalf("handlers should not have re-executed on resume: %+v", callCounts)
	}
}

func TestGraphRunResultCollectsOutputs(t *testing.T) {
	g := graph.New()

	type payload struct {
		Count int
	}

	g.RegisterNode("start", func(ctx context.Context, input any) (any, error) {
		val := input.(int)
		return payload{Count: val + 1}, nil
	})
	g.RegisterNode("branch_add", func(ctx context.Context, input any) (any, error) {
		p := input.(payload)
		return payload{Count: p.Count + 10}, nil
	})
	g.RegisterNode("branch_mul", func(ctx context.Context, input any) (any, error) {
		p := input.(payload)
		return payload{Count: p.Count * 3}, nil
	})

	g.Connect("start", "branch_add")
	g.Connect("start", "branch_mul")
	g.SetEntry("start")
	g.SetExit("branch_mul")

	res, err := g.RunResult(context.Background(), 2)
	if err != nil {
		t.Fatalf("RunResult returned error: %v", err)
	}

	def := res.Default.(payload)
	if def.Count != (2+1)*3 {
		t.Fatalf("unexpected default output: %+v", def)
	}

	startOut, ok := res.Outputs["start"]
	if !ok {
		t.Fatalf("missing output for start node")
	}
	if startOut.(payload).Count != 3 {
		t.Fatalf("unexpected start output: %+v", startOut)
	}

	addOut := res.Outputs["branch_add"].(payload)
	if addOut.Count != 13 {
		t.Fatalf("unexpected branch_add output: %+v", addOut)
	}

	mulOut := res.Outputs["branch_mul"].(payload)
	if mulOut.Count != 9 {
		t.Fatalf("unexpected branch_mul output: %+v", mulOut)
	}
}

func TestGraphRoutedOutput(t *testing.T) {
	g := graph.New()

	g.RegisterNode("start", func(ctx context.Context, input any) (any, error) {
		return input, nil
	})
	g.RegisterNode("branch", func(ctx context.Context, input any) (any, error) {
		val := input.(float64)
		routed := graph.RoutedOutput{Routes: map[string]any{}, Default: val}
		if val >= 0 {
			routed.Routes["positive"] = val
		} else {
			routed.Routes["negative"] = val
		}
		return routed, nil
	})
	g.RegisterNode("positive", func(ctx context.Context, input any) (any, error) {
		return input.(float64) + 10, nil
	})
	g.RegisterNode("negative", func(ctx context.Context, input any) (any, error) {
		return input.(float64) - 10, nil
	})

	g.Connect("start", "branch")
	g.ConnectWithLabel("branch", "positive", "positive")
	g.ConnectWithLabel("branch", "negative", "negative")
	g.SetEntry("start")
	g.SetExit("positive")

	res, err := g.RunResult(context.Background(), 5.0)
	if err != nil {
		t.Fatalf("RunResult returned error: %v", err)
	}
	if res.Default.(float64) != 15 {
		t.Fatalf("unexpected positive branch result: %v", res.Default)
	}

	g.SetExit("negative")

	res, err = g.RunResult(context.Background(), -3.0)
	if err != nil {
		t.Fatalf("RunResult returned error: %v", err)
	}
	negOut, ok := res.Outputs[graph.NodeID("negative")]
	if !ok {
		t.Fatalf("missing negative branch output")
	}
	if negOut.(float64) != -13 {
		t.Fatalf("unexpected negative branch result: %v", negOut)
	}
}

func TestGraphLoopDirective(t *testing.T) {
	g := graph.New()

	g.RegisterNode("loop", func(ctx context.Context, input any) (any, error) {
		val := input.(float64)
		if val < 5 {
			return graph.LoopDirective{Continue: true, Next: val + 1}, nil
		}
		return graph.LoopDirective{Result: val}, nil
	})
	g.RegisterNode("after", func(ctx context.Context, input any) (any, error) {
		return input.(float64) + 2, nil
	})

	g.Connect("loop", "after")
	g.SetEntry("loop")
	g.SetExit("after")

	res, err := g.RunResult(context.Background(), 1.0)
	if err != nil {
		t.Fatalf("RunResult returned error: %v", err)
	}
	if got := res.Default.(float64); got != 7 {
		t.Fatalf("unexpected loop result: %v", got)
	}
}
func TestGraphJoinNode(t *testing.T) {
	g := graph.New()

	g.RegisterNode("start", func(ctx context.Context, input any) (any, error) {
		return input, nil
	})
	g.RegisterNode("branch_a", func(ctx context.Context, input any) (any, error) {
		return input.(float64) * 2, nil
	})
	g.RegisterNode("branch_b", func(ctx context.Context, input any) (any, error) {
		return input.(float64) + 3, nil
	})
	g.RegisterJoinNode("join", func(ctx context.Context, inputs []any) (any, error) {
		sum := 0.0
		for _, v := range inputs {
			switch vv := v.(type) {
			case float64:
				sum += vv
			case float32:
				sum += float64(vv)
			case int:
				sum += float64(vv)
			case int64:
				sum += float64(vv)
			default:
				return nil, fmt.Errorf("unsupported type %T", v)
			}
		}
		return sum, nil
	})
	g.RegisterNode("after", func(ctx context.Context, input any) (any, error) {
		return input.(float64) - 1, nil
	})

	g.Connect("start", "branch_a")
	g.Connect("start", "branch_b")
	g.Connect("branch_a", "join")
	g.Connect("branch_b", "join")
	g.Connect("join", "after")
	g.SetEntry("start")
	g.SetExit("after")

	res, err := g.RunResult(context.Background(), 4.0)
	if err != nil {
		t.Fatalf("RunResult returned error: %v", err)
	}
	if got := res.Default.(float64); got != 14 {
		t.Fatalf("unexpected after value: %v", got)
	}
}

func TestGraphJoinTimeoutTriggers(t *testing.T) {
	g := graph.New()

	g.RegisterNode("start", func(ctx context.Context, input any) (any, error) {
		return input, nil
	})
	g.RegisterNode("branch", func(ctx context.Context, input any) (any, error) {
		val := input.(float64)
		return graph.RoutedOutput{
			Routes:  map[string]any{"positive": val},
			Default: nil,
		}, nil
	})
	g.RegisterJoinNodeWithConfig("join", func(ctx context.Context, inputs []any) (any, error) {
		sum := 0.0
		for _, raw := range inputs {
			if raw == nil {
				continue
			}
			switch v := raw.(type) {
			case float64:
				sum += v
			case float32:
				sum += float64(v)
			case int:
				sum += float64(v)
			}
		}
		return sum, nil
	}, graph.JoinConfig{Timeout: 10 * time.Millisecond})

	g.Connect("start", "branch")
	g.ConnectWithLabel("branch", "join", "positive")
	g.ConnectWithLabel("branch", "join", "negative")

	g.SetEntry("start")
	g.SetExit("join")

	_, err := g.RunResult(context.Background(), 2.0)
	if err == nil || !strings.Contains(err.Error(), "timed out waiting for inputs") {
		t.Fatalf("expected join timeout error, got %v", err)
	}
}

func TestGraphMissingHandlerFails(t *testing.T) {
	g := graph.New()

	g.RegisterNode("start", func(ctx context.Context, input any) (any, error) {
		return input, nil
	})

	g.Connect("start", "missing")
	g.SetEntry("start")
	g.SetExit("missing")

	_, err := g.RunResult(context.Background(), 1)
	if err == nil || !strings.Contains(err.Error(), "handler not registered") {
		t.Fatalf("expected missing handler error, got %v", err)
	}
}

func TestGraphParallelExecution(t *testing.T) {
	g := graph.New()

	for i := 0; i < 5; i++ {
		id := graph.NodeID(fmt.Sprintf("node_%d", i))
		g.RegisterNode(id, func(ctx context.Context, input any) (any, error) {
			if v, ok := input.(int); ok {
				return v + 1, nil
			}
			return input, nil
		})
		if i > 0 {
			g.Connect(graph.NodeID(fmt.Sprintf("node_%d", i-1)), id)
		}
	}

	g.SetEntry("node_0")
	g.SetExit("node_4")

	res, err := g.RunResult(context.Background(), 0, graph.WithParallelism(4))
	if err != nil {
		t.Fatalf("RunResult returned error: %v", err)
	}
	if got := res.Default.(int); got != 5 {
		t.Fatalf("unexpected result: %d", got)
	}
}

func TestGraphNodeRetries(t *testing.T) {
	g := graph.New()

	attempts := 0
	handler := func(ctx context.Context, input any) (any, error) {
		attempts++
		if attempts < 3 {
			return nil, errors.New("temporary failure")
		}
		return input.(int) + 1, nil
	}

	g.RegisterNodeWithConfig("start", handler, graph.NodeExecConfig{Retries: 2, RetryDelay: 5 * time.Millisecond})
	g.SetEntry("start")
	g.SetExit("start")

	res, err := g.RunResult(context.Background(), 0)
	if err != nil {
		t.Fatalf("RunResult error: %v", err)
	}
	if val := res.Default.(int); val != 1 {
		t.Fatalf("unexpected value: %d", val)
	}
	if attempts != 3 {
		t.Fatalf("expected 3 attempts, got %d", attempts)
	}
}

func TestGraphNodeTimeout(t *testing.T) {
	g := graph.New()

	handler := func(ctx context.Context, input any) (any, error) {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(20 * time.Millisecond):
			return input, nil
		}
	}

	g.RegisterNodeWithConfig("slow", handler, graph.NodeExecConfig{Timeout: 5 * time.Millisecond, Retries: 0})
	g.SetEntry("slow")
	g.SetExit("slow")

	_, err := g.RunResult(context.Background(), 0)
	if err == nil {
		t.Fatalf("expected timeout error")
	}
}

func TestGraphSynchronousMode(t *testing.T) {
	buildGraph := func(counter *int) *graph.Graph {
		g := graph.New()
		g.RegisterNode("start", func(ctx context.Context, input any) (any, error) {
			return input.(int) + 1, nil
		})
		g.RegisterNode("direct", func(ctx context.Context, input any) (any, error) {
			return input.(int) * 2, nil
		})
		g.RegisterNode("loop", func(ctx context.Context, input any) (any, error) {
			val := input.(int)
			*counter++
			if val < 4 {
				return graph.LoopDirective{Continue: true, Next: val + 1}, nil
			}
			return val, nil
		})
		g.RegisterJoinNodeWithConfig("join", func(ctx context.Context, inputs []any) (any, error) {
			total := 0.0
			for _, raw := range inputs {
				switch n := raw.(type) {
				case int:
					total += float64(n)
				case float64:
					total += n
				default:
					return nil, errors.New("unexpected type")
				}
			}
			return total, nil
		}, graph.JoinConfig{})
		g.RegisterNode("finish", func(ctx context.Context, input any) (any, error) {
			return int(input.(float64)) + 1, nil
		})

		g.Connect("start", "direct")
		g.Connect("start", "loop")
		g.Connect("direct", "join")
		g.Connect("loop", "join")
		g.Connect("join", "finish")
		g.SetEntry("start")
		g.SetExit("finish")
		return g
	}

	var asyncLoops int
	asyncGraph := buildGraph(&asyncLoops)
	asyncRes, err := asyncGraph.RunResult(context.Background(), 1, graph.WithParallelism(2))
	if err != nil {
		t.Fatalf("async run error: %v", err)
	}
	if asyncLoops != 3 {
		t.Fatalf("expected 3 loop evaluations in async mode, got %d", asyncLoops)
	}
	if got := asyncRes.Default.(int); got != 9 {
		t.Fatalf("unexpected async output: %d", got)
	}

	var syncLoops int
	syncGraph := buildGraph(&syncLoops)
	syncRes, err := syncGraph.RunResult(context.Background(), 1, graph.WithParallelism(2), graph.WithExecutionMode(graph.ExecutionModeSynchronous))
	if err != nil {
		t.Fatalf("sync run error: %v", err)
	}
	if syncLoops != 3 {
		t.Fatalf("expected 3 loop evaluations in sync mode, got %d", syncLoops)
	}
	if got := syncRes.Default.(int); got != 9 {
		t.Fatalf("unexpected sync output: %d", got)
	}
	if asyncRes.Default != syncRes.Default {
		t.Fatalf("sync/asynchronous outputs differ: async=%v sync=%v", asyncRes.Default, syncRes.Default)
	}
}
