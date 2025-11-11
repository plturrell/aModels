package cli

import (
	"fmt"
	"strings"

	"github.com/langchain-ai/langgraph-go/pkg/stategraph"
)

// ValidateProjectConfig performs integrity checks over the declarative graph configuration.
func ValidateProjectConfig(cfg *ProjectConfig) error {
	if cfg == nil {
		return fmt.Errorf("project config is nil")
	}
	if err := validateCheckpoint(cfg.Checkpoint); err != nil {
		return err
	}
	if len(cfg.Graph.Nodes) == 0 {
		return fmt.Errorf("graph must contain at least one node")
	}

	if _, err := parseExecutionMode(cfg.Graph.Options.ExecutionMode); err != nil {
		return err
	}

	for _, node := range cfg.Graph.Nodes {
		execCfg, execKeys, err := parseNodeExecOptions(node.Options)
		if err != nil {
			return fmt.Errorf("node %s: %w", node.ID, err)
		}
		var allowed map[string]struct{}
		switch strings.ToLower(strings.TrimSpace(node.Op)) {
		case "join", "aggregate":
			_, joinCfg, joinKeys, err := makeJoinHandler(node, execCfg)
			if err != nil {
				return fmt.Errorf("node %s: %w", node.ID, err)
			}
			allowed = mergeOptionKeys(execKeys, joinKeys)
			_ = joinCfg // execution config already applied; Validate only needs errors.
		default:
			allowed = execKeys
		}
		if err := validateOptionKeys(node.Options, allowed); err != nil {
			return fmt.Errorf("node %s: %w", node.ID, err)
		}
	}

	conditionalMap := make(map[string]map[string]string, len(cfg.Graph.Conditionals))
	nodeOps := make(map[string]string, len(cfg.Graph.Nodes))
	for _, node := range cfg.Graph.Nodes {
		id := strings.TrimSpace(node.ID)
		if id == "" {
			return fmt.Errorf("graph node id cannot be empty")
		}
		if _, exists := nodeOps[id]; exists {
			return fmt.Errorf("duplicate node id %q", id)
		}
		nodeOps[id] = strings.ToLower(strings.TrimSpace(node.Op))
	}

	for _, cond := range cfg.Graph.Conditionals {
		source := strings.TrimSpace(cond.Source)
		if source == "" {
			return fmt.Errorf("conditional edge missing source")
		}
		if _, ok := nodeOps[source]; !ok {
			return fmt.Errorf("conditional edge references unknown node %q", source)
		}
		if len(cond.PathMap) == 0 {
			return fmt.Errorf("conditional edge for %s requires non-empty path_map", source)
		}
		if _, exists := conditionalMap[source]; exists {
			return fmt.Errorf("conditional edges already defined for node %s", source)
		}
		mapping := make(map[string]string, len(cond.PathMap))
		for route, dest := range cond.PathMap {
			routeKey := strings.TrimSpace(route)
			destKey := strings.TrimSpace(dest)
			if routeKey == "" {
				return fmt.Errorf("conditional edge for %s has empty route label", source)
			}
			if destKey == "" {
				return fmt.Errorf("conditional edge for %s route %s has empty destination", source, route)
			}
			if destKey != stategraph.END {
				if _, ok := nodeOps[destKey]; !ok {
					return fmt.Errorf("conditional edge for %s points to unknown node %q", source, destKey)
				}
			}
			mapping[routeKey] = destKey
		}
		conditionalMap[source] = mapping
	}

	if cfg.Graph.Entry != "" {
		if _, ok := nodeOps[cfg.Graph.Entry]; !ok {
			return fmt.Errorf("entry node %q is not defined", cfg.Graph.Entry)
		}
	}
	if cfg.Graph.Exit != "" {
		if _, ok := nodeOps[cfg.Graph.Exit]; !ok {
			return fmt.Errorf("exit node %q is not defined", cfg.Graph.Exit)
		}
	}

	inDegree := make(map[string]int)
	outDegree := make(map[string]int)
	adjacency := make(map[string][]string)
	for id := range nodeOps {
		adjacency[id] = nil
	}

	for _, edge := range cfg.Graph.Edges {
		from := strings.TrimSpace(edge.From)
		to := strings.TrimSpace(edge.To)
		if from == "" || to == "" {
			return fmt.Errorf("edges must specify both from and to nodes")
		}
		if _, ok := nodeOps[from]; !ok {
			return fmt.Errorf("edge references unknown node %q", from)
		}
		if _, ok := nodeOps[to]; !ok {
			return fmt.Errorf("edge references unknown node %q", to)
		}
		adjacency[from] = append(adjacency[from], to)
		inDegree[to]++
		outDegree[from]++
	}

	for source, routes := range conditionalMap {
		for _, dest := range routes {
			if dest == "" || dest == stategraph.END {
				continue
			}
			adjacency[source] = append(adjacency[source], dest)
			inDegree[dest]++
			outDegree[source]++
		}
	}

	for id, op := range nodeOps {
		switch op {
		case "join", "aggregate":
			if inDegree[id] < 2 {
				return fmt.Errorf("join node %q requires at least two inbound edges", id)
			}
		case "branch", "route":
			outgoing := 0
			labelsValid := true
			labelCount := 0
			for _, edge := range cfg.Graph.Edges {
				if strings.TrimSpace(edge.From) != id {
					continue
				}
				outgoing++
				label := strings.TrimSpace(edge.Label)
				if label == "" {
					labelsValid = false
				} else {
					labelCount++
				}
			}
			if mapping, ok := conditionalMap[id]; ok {
				if len(mapping) < 1 {
					return fmt.Errorf("branch node %q conditional mapping must include at least one route", id)
				}
				labelCount += len(mapping)
			} else {
				if outgoing == 0 {
					return fmt.Errorf("branch node %q has no outbound edges", id)
				}
				if !labelsValid {
					return fmt.Errorf("branch node %q requires labels on outbound edges", id)
				}
			}
			if labelCount < 2 {
				return fmt.Errorf("branch node %q requires at least two labeled routes", id)
			}
		}
	}

	// Cycle detection via Kahn's algorithm.
	indegreeCopy := make(map[string]int, len(nodeOps))
	for id := range nodeOps {
		indegreeCopy[id] = inDegree[id]
	}
	queue := make([]string, 0, len(nodeOps))
	for id := range nodeOps {
		if indegreeCopy[id] == 0 {
			queue = append(queue, id)
		}
	}
	processed := 0
	for len(queue) > 0 {
		node := queue[0]
		queue = queue[1:]
		processed++
		for _, next := range adjacency[node] {
			indegreeCopy[next]--
			if indegreeCopy[next] == 0 {
				queue = append(queue, next)
			}
		}
	}
	if processed != len(nodeOps) {
		return fmt.Errorf("graph contains a cycle; processed %d of %d nodes", processed, len(nodeOps))
	}

	// Reachability check.
	visited := make(map[string]bool, len(nodeOps))
	startNodes := []string{}
	if cfg.Graph.Entry != "" {
		startNodes = append(startNodes, cfg.Graph.Entry)
	} else {
		for id := range nodeOps {
			if inDegree[id] == 0 {
				startNodes = append(startNodes, id)
			}
		}
		if len(startNodes) == 0 {
			for _, node := range cfg.Graph.Nodes {
				startNodes = append(startNodes, node.ID)
				break
			}
		}
	}

	queue = append([]string{}, startNodes...)
	for len(queue) > 0 {
		node := queue[0]
		queue = queue[1:]
		if visited[node] {
			continue
		}
		visited[node] = true
		for _, next := range adjacency[node] {
			queue = append(queue, next)
		}
	}
	if len(visited) != len(nodeOps) {
		for id := range nodeOps {
			if !visited[id] {
				return fmt.Errorf("node %q is unreachable from entry", id)
			}
		}
	}

	return nil
}

func validateCheckpoint(spec string) error {
	spec = strings.TrimSpace(spec)
	switch {
	case spec == "":
		return fmt.Errorf("checkpoint backend must be specified (hana, redis, sqlite)")
	case spec == "hana":
		return nil
	case spec == "redis":
		return nil
	case strings.HasPrefix(spec, "redis://"):
		return nil
	case strings.HasPrefix(spec, "sqlite:"):
		if strings.TrimPrefix(spec, "sqlite:") == "" {
			return fmt.Errorf("sqlite checkpoint requires a file path")
		}
		return nil
	default:
		return fmt.Errorf("unsupported checkpoint backend %q", spec)
	}
}
