package graph

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

// NodeID uniquely identifies a node in a LangGraph pipeline.
type NodeID string

// NodeHandler is the Go equivalent of a LangGraph node callable. The handler
// receives the upstream payload and returns the downstream value (or an error).
type NodeHandler func(ctx context.Context, input any) (any, error)

// Edge describes a directed connection between two nodes in the graph.
type Edge struct {
	From  NodeID
	To    NodeID
	Label string
}

// Result captures the outputs produced by a graph execution run.
type Result struct {
	// Default corresponds to the configured exit node (or entry node when no
	// explicit exit is set).
	Default any
	// Outputs holds the materialized value for every node that executed.
	Outputs map[NodeID]any
}

// Graph is a lightweight, concurrency-safe execution plan. The initial version
// keeps the API intentionally simple so downstream packages can begin wiring up
// nodes while we port the more advanced runtime behaviors (streaming, retries,
// conditional edges, etc.).
type Graph struct {
	mu           sync.RWMutex
	nodes        map[NodeID]NodeHandler
	adjacency    map[NodeID][]Edge
	inbound      map[NodeID]int
	joiners      map[NodeID]bool
	joinTimeouts map[NodeID]time.Duration
	nodeExec     map[NodeID]NodeExecConfig
	entry        NodeID
	explicitEx   NodeID
	state        *StateManager
}

type executionFrame struct {
	id    NodeID
	input any
}

// JoinInput represents the aggregated values delivered to a join handler.
type JoinInput struct {
	Inputs []any
}

// JoinHandler aggregates multiple inputs before producing a single output.
type JoinHandler func(ctx context.Context, inputs []any) (any, error)

// NodeExecConfig controls per-node execution behaviour.
type NodeExecConfig struct {
	Timeout    time.Duration
	Retries    int
	RetryDelay time.Duration
}

// JoinConfig provides additional configuration for join nodes.
type JoinConfig struct {
	Timeout time.Duration
	Exec    NodeExecConfig
}

// RoutedOutput allows nodes to direct specific downstream edges to different
// values. Routes may be keyed by edge label, child node id, or "*"/Default.
type RoutedOutput struct {
	Routes  map[string]any
	Default any
}

// LoopDirective instructs the runtime to re-execute the current node with a new
// input until Continue is false, at which point Result (or the final Next) is
// treated as the node output.
type LoopDirective struct {
	Continue bool
	Next     any
	Result   any
}

// New constructs an empty graph with sensible defaults.
func New() *Graph {
	return &Graph{
		nodes:        make(map[NodeID]NodeHandler),
		adjacency:    make(map[NodeID][]Edge),
		inbound:      make(map[NodeID]int),
		joiners:      make(map[NodeID]bool),
		joinTimeouts: make(map[NodeID]time.Duration),
		nodeExec:     make(map[NodeID]NodeExecConfig),
	}
}

// UseStateManager configures the graph to persist node outputs using the
// provided state manager. Passing nil disables persistence.
func (g *Graph) UseStateManager(sm *StateManager) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.state = sm
}

// SetEntry explicitly sets the entry node. Useful when registering nodes in
// arbitrary order.
func (g *Graph) SetEntry(id NodeID) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.entry = id
}

// SetExit explicitly sets the node whose output should be returned from Run.
func (g *Graph) SetExit(id NodeID) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.explicitEx = id
}

// RegisterNode installs the handler for the given node identifier.
func (g *Graph) RegisterNode(id NodeID, handler NodeHandler) {
	g.RegisterNodeWithConfig(id, handler, NodeExecConfig{})
}

// RegisterNodeWithConfig installs the handler with execution configuration.
func (g *Graph) RegisterNodeWithConfig(id NodeID, handler NodeHandler, cfg NodeExecConfig) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.entry == "" {
		g.entry = id
	}
	g.nodes[id] = handler
	if _, ok := g.adjacency[id]; !ok {
		g.adjacency[id] = nil
	}
	if g.nodeExec == nil {
		g.nodeExec = make(map[NodeID]NodeExecConfig)
	}
	g.nodeExec[id] = cfg
}

// Connect wires two nodes together via a directed edge.
func (g *Graph) Connect(from, to NodeID) {
	g.ConnectWithLabel(from, to, "")
}

// RegisterJoinNode installs a join handler that executes once all inbound
// messages have been collected.
func (g *Graph) RegisterJoinNode(id NodeID, handler JoinHandler) {
	g.RegisterJoinNodeWithConfig(id, handler, JoinConfig{})
}

// RegisterJoinNodeWithConfig installs a join handler with optional configuration.
func (g *Graph) RegisterJoinNodeWithConfig(id NodeID, handler JoinHandler, cfg JoinConfig) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.entry == "" {
		g.entry = id
	}
	if g.joiners == nil {
		g.joiners = make(map[NodeID]bool)
	}
	if g.joinTimeouts == nil {
		g.joinTimeouts = make(map[NodeID]time.Duration)
	}
	if g.nodeExec == nil {
		g.nodeExec = make(map[NodeID]NodeExecConfig)
	}
	wrapped := func(ctx context.Context, input any) (any, error) {
		joinInput, ok := input.(JoinInput)
		if !ok {
			joinInput = JoinInput{Inputs: []any{input}}
		}
		return handler(ctx, joinInput.Inputs)
	}
	g.nodes[id] = wrapped
	if _, ok := g.adjacency[id]; !ok {
		g.adjacency[id] = nil
	}
	g.joiners[id] = true
	g.joinTimeouts[id] = cfg.Timeout
	g.nodeExec[id] = cfg.Exec
}

// ConnectWithLabel wires two nodes together with an optional label used by
// routed outputs.
func (g *Graph) ConnectWithLabel(from, to NodeID, label string) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.adjacency[from] = append(g.adjacency[from], Edge{From: from, To: to, Label: label})
	g.inbound[to]++
	if _, ok := g.adjacency[to]; !ok {
		g.adjacency[to] = nil
	}
}

// Run executes the graph starting from the entry node and returns the value
// produced by the configured exit node.
func (g *Graph) Run(ctx context.Context, input any, opts ...RunOption) (any, error) {
	res, err := g.RunResult(ctx, input, opts...)
	if err != nil {
		return nil, err
	}
	return res.Default, nil
}

// RunResult executes the graph and returns outputs for every node alongside the
// default exit value. It currently performs a breadth-first traversal, passing
// each node's output as the downstream input. Future revisions will mirror the
// full Pregel runtime.

func (g *Graph) RunResult(ctx context.Context, input any, opts ...RunOption) (Result, error) {
	cfg := runConfig{}
	for _, opt := range opts {
		opt(&cfg)
	}

	if err := ctx.Err(); err != nil {
		return Result{}, err
	}

	g.mu.RLock()
	entry := g.entry
	explicitEx := g.explicitEx
	state := g.state

	// Snapshot maps to avoid holding the lock during execution.
	nodes := make(map[NodeID]NodeHandler, len(g.nodes))
	for id, handler := range g.nodes {
		nodes[id] = handler
	}
	adj := make(map[NodeID][]Edge, len(g.adjacency))
	for id, children := range g.adjacency {
		cloned := append([]Edge(nil), children...)
		adj[id] = cloned
	}
	g.mu.RUnlock()

	if entry == "" {
		return Result{}, errors.New("graph: entry node not set")
	}

	workers := cfg.parallelism
	if workers <= 1 {
		workers = 1
	}

	procCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	workCh := make(chan executionFrame, workers*2+4)
	var wg sync.WaitGroup

	enqueue := func(frame executionFrame) {
		wg.Add(1)
		select {
		case workCh <- frame:
		case <-procCtx.Done():
			wg.Done()
		}
	}

	var outputsMu sync.Mutex
	outputs := make(map[NodeID]any)

	var joinMu sync.Mutex
	joinBuffers := make(map[NodeID][]any)
	joinDeadlines := make(map[NodeID]time.Time)
	joinWaiters := make(map[NodeID]struct{})

	var errMu sync.Mutex
	var resultErr error

	setError := func(err error) {
		if err == nil {
			return
		}
		errMu.Lock()
		if resultErr == nil {
			resultErr = err
			cancel()
		}
		errMu.Unlock()
	}

	hasError := func() bool {
		errMu.Lock()
		defer errMu.Unlock()
		return resultErr != nil
	}

	processFrame := func(frame executionFrame) ([]executionFrame, error) {
		if err := procCtx.Err(); err != nil && hasError() {
			return nil, err
		}

		handler := nodes[frame.id]
		var out any
		resumed := false

		if cfg.resume && state != nil {
			var cached any
			loaded, err := state.Load(procCtx, frame.id, &cached)
			if err != nil {
				return nil, err
			}
			if loaded {
				out = cached
				resumed = true
				joinMu.Lock()
				delete(joinBuffers, frame.id)
				delete(joinDeadlines, frame.id)
				delete(joinWaiters, frame.id)
				joinMu.Unlock()
			}
		}

		if !resumed {
			if handler == nil {
				return nil, fmt.Errorf("graph: handler not registered for node %s", frame.id)
			}
			var callInput any
			if g.joiners != nil && g.joiners[frame.id] {
				joinMu.Lock()
				buf := append(joinBuffers[frame.id], frame.input)
				joinBuffers[frame.id] = buf
				required := g.inbound[frame.id]
				if required == 0 {
					required = len(buf)
				}
				timeout := g.joinTimeouts[frame.id]
				if len(buf) == 1 && timeout > 0 {
					joinDeadlines[frame.id] = time.Now().Add(timeout)
				}
				deadline := joinDeadlines[frame.id]
				if len(buf) < required {
					if !deadline.IsZero() && time.Now().After(deadline) {
						joinMu.Unlock()
						return nil, fmt.Errorf("graph: join %s timed out waiting for inputs", frame.id)
					}
					if timeout > 0 {
						if _, waiting := joinWaiters[frame.id]; !waiting {
							joinWaiters[frame.id] = struct{}{}
							wait := time.Until(joinDeadlines[frame.id])
							if wait < 0 {
								wait = 0
							}
							joinMu.Unlock()
							wg.Add(1)
							go func(id NodeID, wait time.Duration) {
								defer wg.Done()
								timer := time.NewTimer(wait)
								defer timer.Stop()
								select {
								case <-procCtx.Done():
									return
								case <-timer.C:
								}
								joinMu.Lock()
								defer joinMu.Unlock()
								if _, active := joinWaiters[id]; !active {
									return
								}
								buf := joinBuffers[id]
								required := g.inbound[id]
								if required == 0 {
									required = len(buf)
								}
								if len(buf) < required {
									delete(joinWaiters, id)
									setError(fmt.Errorf("graph: join %s timed out waiting for inputs", id))
									return
								}
								delete(joinWaiters, id)
							}(frame.id, wait)
							return nil, nil
						}
						joinMu.Unlock()
						return nil, nil
					}
					joinMu.Unlock()
					return nil, nil
				}
				delete(joinBuffers, frame.id)
				if !deadline.IsZero() {
					delete(joinDeadlines, frame.id)
				}
				if _, waiting := joinWaiters[frame.id]; waiting {
					delete(joinWaiters, frame.id)
				}
				joinMu.Unlock()
				callInput = JoinInput{Inputs: buf}
			} else {
				callInput = frame.input
			}
			var err error
			out, err = g.runWithConfig(procCtx, frame.id, handler, callInput)
			if err != nil {
				return nil, err
			}
		}

		if directive, ok := out.(LoopDirective); ok {
			if directive.Continue {
				nextInput := directive.Next
				if nextInput == nil {
					nextInput = frame.input
				}
				return []executionFrame{{id: frame.id, input: nextInput}}, nil
			}
			if directive.Result != nil {
				out = directive.Result
			} else if directive.Next != nil {
				out = directive.Next
			} else {
				out = frame.input
			}
		}

		outputsMu.Lock()
		outputs[frame.id] = out
		outputsMu.Unlock()

		if state != nil && !resumed {
			if err := state.Save(procCtx, frame.id, out); err != nil {
				return nil, err
			}
		}

		next := g.dispatch(out, adj[frame.id])
		return next, nil
	}

	runSynchronous := func(ctx context.Context, initial executionFrame, workers int, step func(executionFrame) ([]executionFrame, error)) error {
		current := []executionFrame{initial}
		nextBatch := make([]executionFrame, 0)

		for len(current) > 0 {
			if err := ctx.Err(); err != nil {
				return err
			}

			nextBatch = nextBatch[:0]
			if workers <= 1 || len(current) == 1 {
				for _, frame := range current {
					nextFrames, err := step(frame)
					if err != nil {
						return err
					}
					if len(nextFrames) > 0 {
						nextBatch = append(nextBatch, nextFrames...)
					}
				}
			} else {
				var (
					mu      sync.Mutex
					wgStep  sync.WaitGroup
					stepErr error
					once    sync.Once
				)
				sem := make(chan struct{}, workers)
				for _, frame := range current {
					frame := frame
					wgStep.Add(1)
					sem <- struct{}{}
					go func() {
						defer wgStep.Done()
						defer func() { <-sem }()

						nextFrames, err := step(frame)
						if err != nil {
							once.Do(func() { stepErr = err })
							return
						}
						if len(nextFrames) == 0 {
							return
						}
						mu.Lock()
						nextBatch = append(nextBatch, nextFrames...)
						mu.Unlock()
					}()
				}
				wgStep.Wait()
				if stepErr != nil {
					return stepErr
				}
			}

			current = append([]executionFrame(nil), nextBatch...)
			nextBatch = nextBatch[:0]
		}
		return nil
	}

	if cfg.mode == ExecutionModeSynchronous {
		initial := executionFrame{id: entry, input: input}
		err := runSynchronous(procCtx, initial, workers, processFrame)
		if err != nil {
			return Result{}, err
		}
		if resultErr != nil {
			return Result{}, resultErr
		}
		if err := procCtx.Err(); err != nil && !errors.Is(err, context.Canceled) {
			return Result{}, err
		}
		target := explicitEx
		if target == "" {
			target = entry
		}
		return Result{
			Default: outputs[target],
			Outputs: outputs,
		}, nil
	}

	var workerWG sync.WaitGroup
	for i := 0; i < workers; i++ {
		workerWG.Add(1)
		go func() {
			defer workerWG.Done()
			for frame := range workCh {
				if hasError() {
					wg.Done()
					continue
				}
				nextFrames, err := processFrame(frame)
				if err != nil {
					setError(err)
				}
				if err == nil {
					for _, nf := range nextFrames {
						enqueue(nf)
					}
				}
				wg.Done()
			}
		}()
	}

	enqueue(executionFrame{id: entry, input: input})

	go func() {
		wg.Wait()
		close(workCh)
	}()

	workerWG.Wait()

	if resultErr != nil {
		return Result{}, resultErr
	}
	if err := procCtx.Err(); err != nil && !errors.Is(err, context.Canceled) {
		return Result{}, err
	}

	target := explicitEx
	if target == "" {
		target = entry
	}
	return Result{
		Default: outputs[target],
		Outputs: outputs,
	}, nil
}

func (g *Graph) dispatch(out any, edges []Edge) []executionFrame {
	frames := make([]executionFrame, 0, len(edges))
	switch routed := out.(type) {
	case RoutedOutput:
		frames = append(frames, dispatchRouted(routed, edges)...)
	default:
		for _, edge := range edges {
			frames = append(frames, executionFrame{id: edge.To, input: out})
		}
	}
	return frames
}

func dispatchRouted(r RoutedOutput, edges []Edge) []executionFrame {
	frames := make([]executionFrame, 0, len(edges))
	routes := r.Routes
	if routes == nil {
		routes = make(map[string]any)
	}
	defaultVal, hasDefault := routes["*"]
	if r.Default != nil {
		defaultVal = r.Default
		hasDefault = true
	}
	for _, edge := range edges {
		if val, ok := routes[string(edge.To)]; ok {
			frames = append(frames, executionFrame{id: edge.To, input: val})
			continue
		}
		if edge.Label != "" {
			if val, ok := routes[edge.Label]; ok {
				frames = append(frames, executionFrame{id: edge.To, input: val})
				continue
			}
		}
		if hasDefault {
			frames = append(frames, executionFrame{id: edge.To, input: defaultVal})
		}
	}
	return frames
}

func (g *Graph) runWithConfig(ctx context.Context, id NodeID, handler NodeHandler, input any) (any, error) {
	cfg := g.nodeExec[id]
	attempts := cfg.Retries + 1
	var lastErr error
	for attempt := 0; attempt < attempts; attempt++ {
		attemptCtx := ctx
		var cancel context.CancelFunc
		if cfg.Timeout > 0 {
			attemptCtx, cancel = context.WithTimeout(ctx, cfg.Timeout)
		} else {
			cancel = func() {}
		}
		out, err := handler(attemptCtx, input)
		cancel()
		if err == nil {
			return out, nil
		}
		lastErr = err
		if attempt < cfg.Retries {
			if cfg.RetryDelay > 0 {
				select {
				case <-ctx.Done():
					return nil, ctx.Err()
				case <-time.After(cfg.RetryDelay):
				}
			}
			continue
		}
	}
	if lastErr == nil {
		lastErr = errors.New("execution failed")
	}
	return nil, fmt.Errorf("graph: node %s failed after %d attempts: %w", id, attempts, lastErr)
}

type runConfig struct {
	resume      bool
	parallelism int
	mode        ExecutionMode
}

// ExecutionMode controls how the runtime schedules node execution.
type ExecutionMode int

const (
	// ExecutionModeAsync executes nodes as inputs become available. This mirrors
	// the prior breadth-first traversal and is the default.
	ExecutionModeAsync ExecutionMode = iota
	// ExecutionModeSynchronous advances the graph in Pregel-style supersteps,
	// inserting a barrier between each wave of node executions.
	ExecutionModeSynchronous
)

// String returns a human-readable representation of the execution mode.
func (m ExecutionMode) String() string {
	switch m {
	case ExecutionModeSynchronous:
		return "synchronous"
	default:
		return "async"
	}
}

// RunOption controls optional behaviors for graph execution.
type RunOption func(*runConfig)

// WithResume instructs the runtime to attempt loading checkpoints for each node
// before executing its handler. Nodes with persisted outputs are skipped and
// their values are replayed downstream.
func WithResume() RunOption {
	return func(cfg *runConfig) {
		cfg.resume = true
	}
}

// WithParallelism configures the number of concurrent workers used to execute
// the graph. Values <= 1 fall back to sequential execution.
func WithParallelism(n int) RunOption {
	return func(cfg *runConfig) {
		if n > 1 {
			cfg.parallelism = n
		}
	}
}

// WithExecutionMode overrides the runtime scheduling strategy.
func WithExecutionMode(mode ExecutionMode) RunOption {
	return func(cfg *runConfig) {
		cfg.mode = mode
	}
}
