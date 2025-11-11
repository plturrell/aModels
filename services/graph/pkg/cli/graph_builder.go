package cli

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/langchain-ai/langgraph-go/pkg/graph"
	"github.com/langchain-ai/langgraph-go/pkg/stategraph"
)

// BuildGraphFromConfig instantiates a LangGraph state graph from the declarative
// graph configuration and returns the compiled runtime.
func BuildGraphFromConfig(cfg GraphConfig, stateManager *graph.StateManager) (*stategraph.CompiledStateGraph, error) {
	if len(cfg.Nodes) == 0 {
		cfg = DefaultGraphConfig()
	}

	builder := stategraph.New()
	if stateManager != nil {
		builder.UseStateManager(stateManager)
	}

	nodeSet := make(map[string]struct{}, len(cfg.Nodes))
	conditionalFuncs := make(map[string]stategraph.ConditionalFunc)
	for _, node := range cfg.Nodes {
		if strings.TrimSpace(node.ID) == "" {
			return nil, errors.New("graph node is missing an id")
		}
		if _, exists := nodeSet[node.ID]; exists {
			return nil, fmt.Errorf("duplicate node id %q", node.ID)
		}
		execCfg, execKeys, err := parseNodeExecOptions(node.Options)
		if err != nil {
			return nil, fmt.Errorf("node %s: %w", node.ID, err)
		}
		op := strings.ToLower(strings.TrimSpace(node.Op))
		switch op {
		case "join", "aggregate":
			joinHandler, joinCfg, joinKeys, err := makeJoinHandler(node, execCfg)
			if err != nil {
				return nil, fmt.Errorf("node %s: %w", node.ID, err)
			}
			if err := validateOptionKeys(node.Options, mergeOptionKeys(execKeys, joinKeys)); err != nil {
				return nil, fmt.Errorf("node %s: %w", node.ID, err)
			}
			if err := builder.AddJoinNode(
				node.ID,
				joinHandler,
				stategraph.WithJoinTimeout(joinCfg.Timeout),
				stategraph.WithJoinExecConfig(joinCfg.Exec),
			); err != nil {
				return nil, fmt.Errorf("node %s: %w", node.ID, err)
			}
		default:
			handler, conditional, err := makeNodeHandler(node)
			if err != nil {
				return nil, fmt.Errorf("node %s: %w", node.ID, err)
			}
			if err := validateOptionKeys(node.Options, execKeys); err != nil {
				return nil, fmt.Errorf("node %s: %w", node.ID, err)
			}
			if err := builder.AddNode(
				node.ID,
				handler,
				stategraph.WithNodeExecConfig(execCfg),
			); err != nil {
				return nil, fmt.Errorf("node %s: %w", node.ID, err)
			}
			if conditional != nil {
				conditionalFuncs[node.ID] = conditional
			}
		}
		nodeSet[node.ID] = struct{}{}
	}

	conditionalRoutes := make(map[string]map[string]string)
	for _, edge := range cfg.Edges {
		if edge.From == "" || edge.To == "" {
			return nil, fmt.Errorf("edge has empty endpoint: %+v", edge)
		}
		if _, ok := nodeSet[edge.From]; !ok {
			return nil, fmt.Errorf("edge references unknown node %q", edge.From)
		}
		if _, ok := nodeSet[edge.To]; !ok {
			return nil, fmt.Errorf("edge references unknown node %q", edge.To)
		}
		if cond := conditionalFuncs[edge.From]; cond != nil && strings.TrimSpace(edge.Label) != "" {
			mapping := conditionalRoutes[edge.From]
			if mapping == nil {
				mapping = make(map[string]string)
				conditionalRoutes[edge.From] = mapping
			}
			mapping[edge.Label] = edge.To
			continue
		}
		if err := builder.AddEdge(edge.From, edge.To, stategraph.WithEdgeLabel(edge.Label)); err != nil {
			return nil, err
		}
	}

	for _, cond := range cfg.Conditionals {
		source := strings.TrimSpace(cond.Source)
		if source == "" {
			return nil, fmt.Errorf("conditional edge missing source")
		}
		if _, ok := nodeSet[source]; !ok {
			return nil, fmt.Errorf("conditional edge references unknown node %q", source)
		}
		if len(cond.PathMap) == 0 {
			return nil, fmt.Errorf("conditional edge for %s requires non-empty path_map", source)
		}
		mapping := make(map[string]string, len(cond.PathMap))
		for route, dest := range cond.PathMap {
			routeKey := strings.TrimSpace(route)
			destKey := strings.TrimSpace(dest)
			if routeKey == "" {
				return nil, fmt.Errorf("conditional edge for %s has empty route label", source)
			}
			if destKey == "" {
				return nil, fmt.Errorf("conditional edge for %s route %s has empty destination", source, route)
			}
			if destKey != stategraph.END {
				if _, ok := nodeSet[destKey]; !ok {
					return nil, fmt.Errorf("conditional edge for %s points to unknown node %q", source, destKey)
				}
			}
			mapping[routeKey] = destKey
		}
		conditionalRoutes[source] = mapping
	}

	for nodeID, mapping := range conditionalRoutes {
		cond := conditionalFuncs[nodeID]
		if cond == nil {
			return nil, fmt.Errorf("conditional edges defined for node %s which does not provide conditional outputs", nodeID)
		}
		if err := builder.AddConditionalEdges(nodeID, cond, mapping); err != nil {
			return nil, err
		}
	}

	if cfg.Entry != "" {
		builder.SetEntryPoint(cfg.Entry)
	}
	if cfg.Exit != "" {
		builder.SetFinishPoint(cfg.Exit)
	}

	return builder.Compile()
}

func makeNodeHandler(node GraphNode) (graph.NodeHandler, stategraph.ConditionalFunc, error) {
	op := strings.ToLower(strings.TrimSpace(node.Op))

	switch op {
	case "add", "bias", "sum":
		vals, err := toFloatArgs(node.Args)
		if err != nil {
			return nil, nil, err
		}
		return func(ctx context.Context, input any) (any, error) {
			val, err := expectFloat(input)
			if err != nil {
				return nil, err
			}
			for _, a := range vals {
				val += a
			}
			return val, nil
		}, nil, nil
	case "multiply", "scale":
		vals, err := toFloatArgs(node.Args)
		if err != nil {
			return nil, nil, err
		}
		return func(ctx context.Context, input any) (any, error) {
			val, err := expectFloat(input)
			if err != nil {
				return nil, err
			}
			for _, a := range vals {
				val *= a
			}
			return val, nil
		}, nil, nil
	case "pow", "power":
		vals, err := toFloatArgs(node.Args)
		if err != nil {
			return nil, nil, err
		}
		if len(vals) == 0 {
			return nil, nil, errors.New("power op requires an exponent argument")
		}
		exponent := vals[0]
		return func(ctx context.Context, input any) (any, error) {
			val, err := expectFloat(input)
			if err != nil {
				return nil, err
			}
			return math.Pow(val, exponent), nil
		}, nil, nil
	case "set", "constant":
		vals, err := toFloatArgs(node.Args)
		if err != nil {
			return nil, nil, err
		}
		if len(vals) == 0 {
			return nil, nil, errors.New("set op requires a value argument")
		}
		value := vals[0]
		return func(context.Context, any) (any, error) {
			return value, nil
		}, nil, nil
	case "noop", "pass", "identity", "":
		return func(ctx context.Context, input any) (any, error) {
			return input, nil
		}, nil, nil
	case "branch", "route":
		comparator := "ge"
		threshold := 0.0
		if len(node.Args) >= 2 {
			comparator = strings.ToLower(fmt.Sprintf("%v", node.Args[0]))
			val, err := toFloat(node.Args[1])
			if err != nil {
				return nil, nil, fmt.Errorf("threshold: %w", err)
			}
			threshold = val
		}
		trueLabel := "true"
		falseLabel := "false"
		if len(node.Args) >= 3 {
			if s, ok := node.Args[2].(string); ok && s != "" {
				trueLabel = s
			}
		}
		if len(node.Args) >= 4 {
			if s, ok := node.Args[3].(string); ok && s != "" {
				falseLabel = s
			}
		}
		trueLabel = strings.TrimSpace(trueLabel)
		falseLabel = strings.TrimSpace(falseLabel)
		handler := func(ctx context.Context, input any) (any, error) {
			return expectFloat(input)
		}
		conditional := func(ctx context.Context, output any) ([]string, error) {
			val, err := expectFloat(output)
			if err != nil {
				return nil, err
			}
			matched, err := evaluateComparator(comparator, val, threshold)
			if err != nil {
				return nil, err
			}
			if matched {
				if trueLabel == "" {
					return nil, nil
				}
				return []string{trueLabel}, nil
			}
			if falseLabel == "" {
				return nil, nil
			}
			return []string{falseLabel}, nil
		}
		return handler, conditional, nil
	case "loop_until", "loop":
		if len(node.Args) < 3 {
			return nil, nil, errors.New("loop_until requires comparator, threshold, and step arguments")
		}
		comparator := strings.ToLower(fmt.Sprintf("%v", node.Args[0]))
		threshold, err := toFloat(node.Args[1])
		if err != nil {
			return nil, nil, fmt.Errorf("threshold: %w", err)
		}
		step, err := toFloat(node.Args[2])
		if err != nil {
			return nil, nil, fmt.Errorf("step: %w", err)
		}
		return func(ctx context.Context, input any) (any, error) {
			val, err := expectFloat(input)
			if err != nil {
				return nil, err
			}
			matched, err := evaluateComparator(comparator, val, threshold)
			if err != nil {
				return nil, err
			}
			if matched {
				return graph.LoopDirective{Continue: true, Next: val + step}, nil
			}
			return graph.LoopDirective{Result: val}, nil
		}, nil, nil
	default:
		return nil, nil, fmt.Errorf("unknown op %q", node.Op)
	}
}

func makeJoinHandler(node GraphNode, execCfg graph.NodeExecConfig) (graph.JoinHandler, graph.JoinConfig, map[string]struct{}, error) {
	strategy := "collect"
	recognized := make(map[string]struct{})
	if node.Options != nil {
		if v, ok := node.Options["aggregate"]; ok {
			recognized["aggregate"] = struct{}{}
			strategy = strings.ToLower(fmt.Sprintf("%v", v))
		}
	} else if len(node.Args) > 0 {
		if s, ok := node.Args[0].(string); ok && s != "" {
			strategy = strings.ToLower(s)
		}
	}

	joinTimeout := time.Duration(0)
	if node.Options != nil {
		if v, ok := node.Options["join_timeout_ms"]; ok {
			recognized["join_timeout_ms"] = struct{}{}
			scalar, err := toFloat(v)
			if err != nil {
				return nil, graph.JoinConfig{}, recognized, fmt.Errorf("join_timeout_ms: %w", err)
			}
			if scalar < 0 {
				return nil, graph.JoinConfig{}, recognized, fmt.Errorf("join_timeout_ms cannot be negative")
			}
			if scalar > 0 {
				joinTimeout = time.Duration(scalar * float64(time.Millisecond))
			}
		}
	}

	var aggregate func([]any) (any, error)
	switch strategy {
	case "collect", "list":
		aggregate = func(inputs []any) (any, error) {
			out := make([]any, len(inputs))
			copy(out, inputs)
			return out, nil
		}
	case "sum":
		aggregate = func(inputs []any) (any, error) {
			vals, err := numericInputs(inputs)
			if err != nil {
				return nil, err
			}
			total := 0.0
			for _, v := range vals {
				total += v
			}
			return total, nil
		}
	case "avg", "average", "mean":
		aggregate = func(inputs []any) (any, error) {
			vals, err := numericInputs(inputs)
			if err != nil {
				return nil, err
			}
			if len(vals) == 0 {
				return 0.0, nil
			}
			total := 0.0
			for _, v := range vals {
				total += v
			}
			return total / float64(len(vals)), nil
		}
	case "max":
		aggregate = func(inputs []any) (any, error) {
			vals, err := numericInputs(inputs)
			if err != nil {
				return nil, err
			}
			if len(vals) == 0 {
				return 0.0, nil
			}
			m := vals[0]
			for _, v := range vals[1:] {
				if v > m {
					m = v
				}
			}
			return m, nil
		}
	case "min":
		aggregate = func(inputs []any) (any, error) {
			vals, err := numericInputs(inputs)
			if err != nil {
				return nil, err
			}
			if len(vals) == 0 {
				return 0.0, nil
			}
			m := vals[0]
			for _, v := range vals[1:] {
				if v < m {
					m = v
				}
			}
			return m, nil
		}
	case "first":
		aggregate = func(inputs []any) (any, error) {
			if len(inputs) == 0 {
				return nil, nil
			}
			return inputs[0], nil
		}
	case "last":
		aggregate = func(inputs []any) (any, error) {
			if len(inputs) == 0 {
				return nil, nil
			}
			return inputs[len(inputs)-1], nil
		}
	case "concat":
		aggregate = func(inputs []any) (any, error) {
			var b strings.Builder
			for i, val := range inputs {
				if i > 0 {
					b.WriteString(" ")
				}
				b.WriteString(fmt.Sprintf("%v", val))
			}
			return b.String(), nil
		}
	default:
		return nil, graph.JoinConfig{}, recognized, fmt.Errorf("unknown join aggregate %q", strategy)
	}

	handler := func(ctx context.Context, inputs []any) (any, error) {
		return aggregate(inputs)
	}

	return handler, graph.JoinConfig{Timeout: joinTimeout, Exec: execCfg}, recognized, nil
}

func parseNodeExecOptions(opts map[string]any) (graph.NodeExecConfig, map[string]struct{}, error) {
	var cfg graph.NodeExecConfig
	recognized := make(map[string]struct{})
	if opts == nil {
		return cfg, recognized, nil
	}

	if raw, ok := opts["timeout_ms"]; ok {
		recognized["timeout_ms"] = struct{}{}
		timeout, err := toFloat(raw)
		if err != nil {
			return cfg, recognized, fmt.Errorf("timeout_ms: %w", err)
		}
		if timeout < 0 {
			return cfg, recognized, fmt.Errorf("timeout_ms cannot be negative")
		}
		if timeout > 0 {
			cfg.Timeout = time.Duration(timeout * float64(time.Millisecond))
		}
	}

	var retryRaw any
	if raw, ok := opts["retry"]; ok {
		recognized["retry"] = struct{}{}
		retryRaw = raw
	} else if raw, ok := opts["retries"]; ok {
		recognized["retries"] = struct{}{}
		retryRaw = raw
	}
	if retryRaw != nil {
		retry, err := toFloat(retryRaw)
		if err != nil {
			return cfg, recognized, fmt.Errorf("retry: %w", err)
		}
		if retry < 0 {
			return cfg, recognized, fmt.Errorf("retry cannot be negative")
		}
		if float64(int(retry)) != retry {
			return cfg, recognized, fmt.Errorf("retry must be an integer")
		}
		cfg.Retries = int(retry)
	}

	if raw, ok := opts["retry_delay_ms"]; ok {
		recognized["retry_delay_ms"] = struct{}{}
		delay, err := toFloat(raw)
		if err != nil {
			return cfg, recognized, fmt.Errorf("retry_delay_ms: %w", err)
		}
		if delay < 0 {
			return cfg, recognized, fmt.Errorf("retry_delay_ms cannot be negative")
		}
		if delay > 0 {
			cfg.RetryDelay = time.Duration(delay * float64(time.Millisecond))
		}
	}

	return cfg, recognized, nil
}

func numericInputs(inputs []any) ([]float64, error) {
	vals := make([]float64, len(inputs))
	for i, raw := range inputs {
		v, err := toFloat(raw)
		if err != nil {
			return nil, fmt.Errorf("aggregate input %d: %w", i, err)
		}
		vals[i] = v
	}
	return vals, nil
}

func toFloatArgs(args []any) ([]float64, error) {
	vals := make([]float64, len(args))
	for i, raw := range args {
		val, err := toFloat(raw)
		if err != nil {
			return nil, fmt.Errorf("arg %d: %w", i, err)
		}
		vals[i] = val
	}
	return vals, nil
}

func toFloat(v any) (float64, error) {
	switch n := v.(type) {
	case float64:
		return n, nil
	case float32:
		return float64(n), nil
	case int:
		return float64(n), nil
	case int64:
		return float64(n), nil
	case int32:
		return float64(n), nil
	case uint:
		return float64(n), nil
	case uint32:
		return float64(n), nil
	case uint64:
		return float64(n), nil
	case json.Number:
		return n.Float64()
	default:
		return 0, fmt.Errorf("cannot convert %T to float", v)
	}
}

func expectFloat(input any) (float64, error) {
	switch v := input.(type) {
	case float64:
		return v, nil
	case float32:
		return float64(v), nil
	case int:
		return float64(v), nil
	case int64:
		return float64(v), nil
	case json.Number:
		return v.Float64()
	default:
		return 0, fmt.Errorf("expected numeric input, got %T", input)
	}
}

func evaluateComparator(op string, value, threshold float64) (bool, error) {
	switch op {
	case "gt", ">", "greater", "greater_than":
		return value > threshold, nil
	case "gte", ">=", "ge", "greater_equal":
		return value >= threshold, nil
	case "lt", "<", "less", "less_than":
		return value < threshold, nil
	case "lte", "<=", "le", "less_equal":
		return value <= threshold, nil
	case "eq", "==", "equals":
		return value == threshold, nil
	case "neq", "!=", "not_equal":
		return value != threshold, nil
	default:
		return false, fmt.Errorf("unknown comparator %q", op)
	}
}

func mergeOptionKeys(sets ...map[string]struct{}) map[string]struct{} {
	out := make(map[string]struct{})
	for _, s := range sets {
		for k := range s {
			out[k] = struct{}{}
		}
	}
	return out
}

func validateOptionKeys(opts map[string]any, allowed map[string]struct{}) error {
	if opts == nil || len(opts) == 0 {
		return nil
	}
	for k := range opts {
		if _, ok := allowed[k]; !ok {
			return fmt.Errorf("unknown option %q", k)
		}
	}
	return nil
}
