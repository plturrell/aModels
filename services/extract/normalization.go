package main

import (
	"fmt"
	"strings"
)

type normalizationInput struct {
	Nodes               []Node
	Edges               []Edge
	ProjectID           string
	SystemID            string
	InformationSystemID string
	Catalog             *Catalog
}

type normalizationResult struct {
	Nodes      []Node
	Edges      []Edge
	RootNodeID string
	Stats      map[string]any
	Warnings   []string
}

func normalizeGraph(input normalizationInput) normalizationResult {
	result := normalizationResult{
		Stats:    map[string]any{},
		Warnings: []string{},
	}

	originalNodeCount := len(input.Nodes)
	originalEdgeCount := len(input.Edges)

	nodeMap := make(map[string]Node)
	nodeOrder := make([]string, 0, originalNodeCount)
	droppedNodes := 0

	for _, node := range input.Nodes {
		id := strings.TrimSpace(node.ID)
		if id == "" {
			droppedNodes++
			result.Warnings = append(result.Warnings, "dropped node with empty id")
			continue
		}

		normalized := Node{
			ID:    id,
			Type:  canonicalType(node.Type),
			Label: canonicalLabel(node.Label, id),
			Props: copyProperties(node.Props),
		}

		if existing, ok := nodeMap[id]; ok {
			normalized.Type = preferType(existing.Type, normalized.Type)
			normalized.Label = preferLabel(existing.Label, normalized.Label, id)
			normalized.Props = mergeProperties(existing.Props, normalized.Props)
			nodeMap[id] = normalized
			continue
		}

		nodeMap[id] = normalized
		nodeOrder = append(nodeOrder, id)
	}

	edgeMap := make(map[string]Edge)
	edgeOrder := make([]string, 0, originalEdgeCount)
	droppedEdges := 0

	for _, edge := range input.Edges {
		src := strings.TrimSpace(edge.SourceID)
		dst := strings.TrimSpace(edge.TargetID)
		label := canonicalLabel(edge.Label, "RELATIONSHIP")

		if src == "" || dst == "" {
			droppedEdges++
			result.Warnings = append(result.Warnings, "dropped edge with missing source or target")
			continue
		}
		if _, ok := nodeMap[src]; !ok {
			droppedEdges++
			result.Warnings = append(result.Warnings, fmt.Sprintf("dropped edge %s->%s (missing source node)", src, dst))
			continue
		}
		if _, ok := nodeMap[dst]; !ok {
			droppedEdges++
			result.Warnings = append(result.Warnings, fmt.Sprintf("dropped edge %s->%s (missing target node)", src, dst))
			continue
		}

		key := edgeKey(src, dst, label)
		normalized := Edge{
			SourceID: src,
			TargetID: dst,
			Label:    label,
			Props:    copyProperties(edge.Props),
		}

		if existing, ok := edgeMap[key]; ok {
			normalized.Props = mergeProperties(existing.Props, normalized.Props)
			edgeMap[key] = normalized
			continue
		}

		edgeMap[key] = normalized
		edgeOrder = append(edgeOrder, key)
	}

	rootID := selectRootNode(nodeMap, nodeOrder)

	catalogNodesAdded := 0
	catalogEntriesAdded := 0
	var catalogChanged bool
	if input.Catalog != nil {
		if nodeAdded, updated := ensureCatalogNode(input.Catalog, nodeMap, &nodeOrder, input.ProjectID, "project"); nodeAdded || updated {
			if nodeAdded {
				catalogNodesAdded++
			}
			if updated {
				catalogChanged = true
				catalogEntriesAdded++
			}
		}
		if rootID == "" && strings.TrimSpace(input.ProjectID) != "" {
			rootID = strings.TrimSpace(input.ProjectID)
		}

		if nodeAdded, updated := ensureCatalogNode(input.Catalog, nodeMap, &nodeOrder, input.SystemID, "system"); nodeAdded || updated {
			if nodeAdded {
				catalogNodesAdded++
			}
			if updated {
				catalogChanged = true
				catalogEntriesAdded++
			}
		}

		if nodeAdded, updated := ensureCatalogNode(input.Catalog, nodeMap, &nodeOrder, input.InformationSystemID, "information-system"); nodeAdded || updated {
			if nodeAdded {
				catalogNodesAdded++
			}
			if updated {
				catalogChanged = true
				catalogEntriesAdded++
			}
		}

		if catalogChanged {
			if err := input.Catalog.Save(); err != nil {
				result.Warnings = append(result.Warnings, fmt.Sprintf("failed to persist catalog update: %v", err))
			}
		}
	}

	if rootID == "" {
		result.Warnings = append(result.Warnings, "graph produced no root node; catalog containment edges skipped")
	} else {
		addCatalogContainmentEdge(edgeMap, &edgeOrder, rootID, input.ProjectID)
		addCatalogContainmentEdge(edgeMap, &edgeOrder, rootID, input.SystemID)
		addCatalogContainmentEdge(edgeMap, &edgeOrder, rootID, input.InformationSystemID)
	}

	finalNodes := make([]Node, 0, len(nodeOrder))
	for _, id := range nodeOrder {
		if node, ok := nodeMap[id]; ok {
			finalNodes = append(finalNodes, node)
		}
	}

	finalEdges := make([]Edge, 0, len(edgeOrder))
	for _, key := range edgeOrder {
		if edge, ok := edgeMap[key]; ok {
			finalEdges = append(finalEdges, edge)
		}
	}

	result.Nodes = finalNodes
	result.Edges = finalEdges
	result.RootNodeID = rootID

	result.Stats["original_node_count"] = originalNodeCount
	result.Stats["unique_node_count"] = len(nodeMap)
	result.Stats["dropped_nodes"] = droppedNodes
	result.Stats["catalog_nodes_added"] = catalogNodesAdded
	result.Stats["catalog_entries_added"] = catalogEntriesAdded
	result.Stats["original_edge_count"] = originalEdgeCount
	result.Stats["unique_edge_count"] = len(edgeMap)
	result.Stats["dropped_edges"] = droppedEdges
	result.Stats["root_node_id"] = rootID

	return result
}

func canonicalType(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	if value == "" {
		return "unknown"
	}
	return value
}

func canonicalLabel(value, fallback string) string {
	value = strings.TrimSpace(value)
	if value != "" {
		return value
	}
	return strings.TrimSpace(fallback)
}

func preferType(existing, candidate string) string {
	if existing != "" && existing != "unknown" {
		return existing
	}
	if candidate == "" {
		return "unknown"
	}
	return candidate
}

func preferLabel(existing, candidate, fallback string) string {
	if existing != "" {
		return existing
	}
	if candidate != "" {
		return candidate
	}
	return fallback
}

func copyProperties(props map[string]any) map[string]any {
	if props == nil {
		return nil
	}
	cp := make(map[string]any, len(props))
	for k, v := range props {
		cp[k] = v
	}
	return cp
}

func mergeProperties(base, extra map[string]any) map[string]any {
	if base == nil && extra == nil {
		return nil
	}
	if base == nil {
		return copyProperties(extra)
	}
	if extra == nil {
		return copyProperties(base)
	}
	out := copyProperties(base)
	for k, v := range extra {
		out[k] = v
	}
	return out
}

func edgeKey(src, dst, label string) string {
	return src + "->" + dst + "#" + label
}

func ensureCatalogNode(c *Catalog, nodeMap map[string]Node, order *[]string, id, kind string) (nodeAdded bool, catalogUpdated bool) {
	id = strings.TrimSpace(id)
	if id == "" {
		return false, false
	}

	if _, ok := nodeMap[id]; !ok {
		nodeMap[id] = Node{
			ID:    id,
			Type:  kind,
			Label: id,
		}
		*order = append(*order, id)
		nodeAdded = true
	} else {
		node := nodeMap[id]
		if node.Type == "unknown" || node.Type == "" {
			node.Type = kind
		}
		if strings.TrimSpace(node.Label) == "" {
			node.Label = id
		}
		nodeMap[id] = node
	}

	switch kind {
	case "project":
		catalogUpdated = c.EnsureProject(id, id)
	case "system":
		catalogUpdated = c.EnsureSystem(id, id)
	case "information-system":
		catalogUpdated = c.EnsureInformationSystem(id, id)
	}

	return nodeAdded, catalogUpdated
}

func addCatalogContainmentEdge(edgeMap map[string]Edge, order *[]string, rootID, nodeID string) bool {
	nodeID = strings.TrimSpace(nodeID)
	if nodeID == "" || nodeID == rootID {
		return false
	}
	label := "CONTAINS"
	key := edgeKey(nodeID, rootID, label)
	if _, exists := edgeMap[key]; exists {
		return false
	}
	edgeMap[key] = Edge{
		SourceID: nodeID,
		TargetID: rootID,
		Label:    label,
	}
	*order = append(*order, key)
	return true
}

func selectRootNode(nodeMap map[string]Node, order []string) string {
	if len(order) == 0 {
		return ""
	}

	preferredTypes := []string{"table", "document", "control-m-job", "project", "system"}

	for _, t := range preferredTypes {
		for _, id := range order {
			if node, ok := nodeMap[id]; ok && node.Type == t {
				return id
			}
		}
	}

	return order[0]
}
