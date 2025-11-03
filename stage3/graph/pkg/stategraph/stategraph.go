package stategraph

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/langchain-ai/langgraph-go/pkg/graph"
)

// START and END mirror the LangGraph Python constants and are provided so call
// sites can express entry/finish edges using the same identifiers.
const (
	START = "__start__"
	END   = "__end__"
)

// NodeFunc is the callable signature used by StateGraph nodes. It mirrors the
// Python StateGraph contract where a node receives an input payload and returns
// a value (or an error) that flows to downstream nodes.
type NodeFunc = graph.NodeHandler

// JoinFunc represents the handler signature for join nodes.
type JoinFunc = graph.JoinHandler

// ConditionalFunc evaluates the output of a node and returns one or more route
// labels that determine which conditional edges should fire.
type ConditionalFunc func(ctx context.Context, value any) ([]string, error)

// StateGraph is a higher-level facade that mirrors the Python StateGraph API
// while reusing the existing Go runtime implementation.
type StateGraph struct {
	mu           sync.RWMutex
	nodes        map[string]*nodeSpec
	edges        map[edgeKey]struct{}
	entry        string
	exit         string
	stateManager *graph.StateManager
}

type nodeSpec struct {
	name        string
	handler     NodeFunc
	execConfig  graph.NodeExecConfig
	isJoin      bool
	joinHandler JoinFunc
	joinConfig  graph.JoinConfig
	conditional *conditionalSpec
}

type conditionalSpec struct {
	path    ConditionalFunc
	mapping map[string]string
}

type edgeKey struct {
	From  string
	To    string
	Label string
}

// NodeOption customises node execution.
type NodeOption func(*nodeSpec)

// JoinOption customises join node execution.
type JoinOption func(*nodeSpec)

// EdgeOption customises edge registration.
type EdgeOption func(*edgeKey)

// WithNodeExecConfig applies the full execution configuration for a node.
func WithNodeExecConfig(cfg graph.NodeExecConfig) NodeOption {
	return func(spec *nodeSpec) {
		spec.execConfig = cfg
	}
}

// WithNodeTimeout sets a per-node execution timeout.
func WithNodeTimeout(d time.Duration) NodeOption {
	return func(spec *nodeSpec) {
		spec.execConfig.Timeout = d
	}
}

// WithNodeRetries controls how many times a node should retry after failure.
func WithNodeRetries(n int) NodeOption {
	return func(spec *nodeSpec) {
		if n < 0 {
			n = 0
		}
		spec.execConfig.Retries = n
	}
}

// WithNodeRetryDelay sets the backoff delay between retries.
func WithNodeRetryDelay(d time.Duration) NodeOption {
	return func(spec *nodeSpec) {
		spec.execConfig.RetryDelay = d
	}
}

// WithJoinTimeout configures how long a join waits for upstream inputs.
func WithJoinTimeout(d time.Duration) JoinOption {
	return func(spec *nodeSpec) {
		spec.joinConfig.Timeout = d
	}
}

// WithJoinExecConfig applies execution settings for the synthetic join node
// handler.
func WithJoinExecConfig(cfg graph.NodeExecConfig) JoinOption {
	return func(spec *nodeSpec) {
		spec.joinConfig.Exec = cfg
	}
}

// WithEdgeLabel assigns an optional label to an edge so conditional routing can
// target specific downstream nodes.
func WithEdgeLabel(label string) EdgeOption {
	return func(edge *edgeKey) {
		edge.Label = label
	}
}

// New returns an empty StateGraph builder.
func New() *StateGraph {
	return &StateGraph{
		nodes: make(map[string]*nodeSpec),
		edges: make(map[edgeKey]struct{}),
	}
}

// UseStateManager enables checkpoint persistence using the provided manager.
func (s *StateGraph) UseStateManager(sm *graph.StateManager) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.stateManager = sm
}

// AddNode registers a standard (non-join) node with the supplied handler.
func (s *StateGraph) AddNode(name string, handler NodeFunc, opts ...NodeOption) error {
	name = strings.TrimSpace(name)
	if name == "" {
		return fmt.Errorf("stategraph: node name must be provided")
	}
	if handler == nil {
		return fmt.Errorf("stategraph: node %q requires a handler", name)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.nodes[name]; exists {
		return fmt.Errorf("stategraph: node %q already exists", name)
	}
	spec := &nodeSpec{name: name, handler: handler}
	for _, opt := range opts {
		if opt != nil {
			opt(spec)
		}
	}
	s.nodes[name] = spec
	return nil
}

// AddJoinNode registers a join node that aggregates multiple inbound values.
func (s *StateGraph) AddJoinNode(name string, handler JoinFunc, opts ...JoinOption) error {
	name = strings.TrimSpace(name)
	if name == "" {
		return fmt.Errorf("stategraph: join node name must be provided")
	}
	if handler == nil {
		return fmt.Errorf("stategraph: join node %q requires a handler", name)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.nodes[name]; exists {
		return fmt.Errorf("stategraph: node %q already exists", name)
	}
	spec := &nodeSpec{
		name:        name,
		isJoin:      true,
		joinHandler: handler,
	}
	for _, opt := range opts {
		if opt != nil {
			opt(spec)
		}
	}
	s.nodes[name] = spec
	return nil
}

// AddEdge wires two nodes together. Special identifiers START and END mirror
// LangGraph's Python API and simply set the entry or finish point.
func (s *StateGraph) AddEdge(from, to string, opts ...EdgeOption) error {
	from = strings.TrimSpace(from)
	to = strings.TrimSpace(to)
	if from == "" || to == "" {
		return fmt.Errorf("stategraph: edges require non-empty endpoints")
	}

	edge := edgeKey{From: from, To: to}
	for _, opt := range opts {
		if opt != nil {
			opt(&edge)
		}
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	switch {
	case from == START:
		s.entry = to
		return nil
	case to == END:
		s.exit = from
		return nil
	}

	s.edges[edge] = struct{}{}
	return nil
}

// AddConditionalEdges associates a routing function with a node. The function
// returns route labels which are matched against the provided pathMap. Each
// route automatically registers a labeled edge to the corresponding destination.
func (s *StateGraph) AddConditionalEdges(source string, path ConditionalFunc, pathMap map[string]string) error {
	source = strings.TrimSpace(source)
	if source == "" {
		return fmt.Errorf("stategraph: conditional edges require a source node")
	}
	if path == nil {
		return fmt.Errorf("stategraph: conditional edges for %q require a path function", source)
	}
	if len(pathMap) == 0 {
		return fmt.Errorf("stategraph: conditional edges for %q require a non-empty path map", source)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	spec, ok := s.nodes[source]
	if !ok {
		return fmt.Errorf("stategraph: conditional edges reference unknown node %q", source)
	}
	if spec.isJoin {
		return fmt.Errorf("stategraph: conditional edges are not supported on join node %q", source)
	}
	if spec.conditional != nil {
		return fmt.Errorf("stategraph: conditional edges already configured for node %q", source)
	}

	mapping := make(map[string]string, len(pathMap))
	for route, dest := range pathMap {
		routeKey := strings.TrimSpace(route)
		destKey := strings.TrimSpace(dest)
		if routeKey == "" {
			return fmt.Errorf("stategraph: conditional path for node %q contains an empty route label", source)
		}
		if destKey == "" {
			return fmt.Errorf("stategraph: conditional path for node %q route %q has empty destination", source, routeKey)
		}
		mapping[routeKey] = destKey
		if destKey != END {
			s.edges[edgeKey{From: source, To: destKey, Label: routeKey}] = struct{}{}
		}
	}
	spec.conditional = &conditionalSpec{
		path:    path,
		mapping: mapping,
	}
	return nil
}

// SetEntryPoint explicitly selects the graph entry node.
func (s *StateGraph) SetEntryPoint(node string) {
	node = strings.TrimSpace(node)
	if node == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.entry = node
}

// SetFinishPoint marks which node's output is returned as the graph result.
func (s *StateGraph) SetFinishPoint(node string) {
	node = strings.TrimSpace(node)
	if node == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.exit = node
}

// Compile materialises the underlying runtime graph and returns a compiled
// facade that exposes Invoke/Run helpers mirroring the Python implementation.
func (s *StateGraph) Compile() (*CompiledStateGraph, error) {
	s.mu.RLock()
	nodes := make(map[string]nodeSpec, len(s.nodes))
	for name, spec := range s.nodes {
		cloned := *spec
		if spec.conditional != nil {
			cond := *spec.conditional
			cond.mapping = copyStringMap(spec.conditional.mapping)
			cloned.conditional = &cond
		}
		nodes[name] = cloned
	}
	edges := make([]edgeKey, 0, len(s.edges))
	for edge := range s.edges {
		edges = append(edges, edge)
	}
	entry := s.entry
	exit := s.exit
	stateMgr := s.stateManager
	s.mu.RUnlock()

	if len(nodes) == 0 {
		return nil, fmt.Errorf("stategraph: no nodes have been registered")
	}

	g := graph.New()
	if stateMgr != nil {
		g.UseStateManager(stateMgr)
	}

	for name, spec := range nodes {
		if spec.isJoin {
			if spec.joinHandler == nil {
				return nil, fmt.Errorf("stategraph: join node %q is missing a handler", name)
			}
			g.RegisterJoinNodeWithConfig(graph.NodeID(name), spec.joinHandler, spec.joinConfig)
			continue
		}
		if spec.handler == nil {
			return nil, fmt.Errorf("stategraph: node %q has no handler", name)
		}
		handler := spec.handler
		if spec.conditional != nil {
			cond := spec.conditional
			handler = wrapConditionalHandler(name, handler, *cond)
		}
		g.RegisterNodeWithConfig(graph.NodeID(name), handler, spec.execConfig)
	}

	for _, edge := range edges {
		if _, ok := nodes[edge.From]; !ok {
			return nil, fmt.Errorf("stategraph: edge references unknown source node %q", edge.From)
		}
		if edge.To != END {
			if _, ok := nodes[edge.To]; !ok {
				return nil, fmt.Errorf("stategraph: edge references unknown destination node %q", edge.To)
			}
		}
		if edge.To == END {
			continue
		}
		g.ConnectWithLabel(graph.NodeID(edge.From), graph.NodeID(edge.To), edge.Label)
	}

	if entry != "" {
		if _, ok := nodes[entry]; !ok {
			return nil, fmt.Errorf("stategraph: entry node %q not registered", entry)
		}
		g.SetEntry(graph.NodeID(entry))
	}
	if exit != "" {
		if _, ok := nodes[exit]; !ok {
			return nil, fmt.Errorf("stategraph: finish node %q not registered", exit)
		}
		g.SetExit(graph.NodeID(exit))
	}

	return &CompiledStateGraph{graph: g}, nil
}

func wrapConditionalHandler(name string, base NodeFunc, cond conditionalSpec) NodeFunc {
	return func(ctx context.Context, input any) (any, error) {
		out, err := base(ctx, input)
		if err != nil {
			return nil, err
		}
		routes, err := cond.path(ctx, out)
		if err != nil {
			return nil, fmt.Errorf("stategraph: node %q conditional errored: %w", name, err)
		}
		if len(routes) == 0 {
			return out, nil
		}
		routed := graph.RoutedOutput{
			Routes: make(map[string]any, len(routes)),
		}
		for _, raw := range routes {
			route := strings.TrimSpace(raw)
			if route == "" {
				continue
			}
			dest, ok := cond.mapping[route]
			if !ok {
				return nil, fmt.Errorf("stategraph: node %q returned unknown route %q", name, route)
			}
			if dest == END {
				continue
			}
			routed.Routes[route] = out
		}
		if len(routed.Routes) == 0 {
			return out, nil
		}
		return routed, nil
	}
}

func copyStringMap(in map[string]string) map[string]string {
	if in == nil {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

// CompiledStateGraph wraps the runtime graph and exposes helpers that mirror
// the Python compiled graph surface.
type CompiledStateGraph struct {
	graph *graph.Graph
}

// Invoke executes the compiled graph with the provided input and returns the
// output from the configured finish node.
func (c *CompiledStateGraph) Invoke(ctx context.Context, input any, opts ...graph.RunOption) (any, error) {
	if c == nil || c.graph == nil {
		return nil, fmt.Errorf("stategraph: compiled graph is nil")
	}
	return c.graph.Run(ctx, input, opts...)
}

// RunResult executes the graph while also returning the per-node outputs.
func (c *CompiledStateGraph) RunResult(ctx context.Context, input any, opts ...graph.RunOption) (graph.Result, error) {
	if c == nil || c.graph == nil {
		return graph.Result{}, fmt.Errorf("stategraph: compiled graph is nil")
	}
	return c.graph.RunResult(ctx, input, opts...)
}

// Runtime exposes the underlying graph for advanced scenarios or migration
// work that still depends on the lower-level API.
func (c *CompiledStateGraph) Runtime() *graph.Graph {
	if c == nil {
		return nil
	}
	return c.graph
}
