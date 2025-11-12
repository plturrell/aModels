package integrations

import (
	"bytes"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
)

// SignavioElementSummary captures a single process element.
type SignavioElementSummary struct {
	ID   string `json:"id"`
	Type string `json:"type"`
	Name string `json:"name"`
}

// SignavioProcessSummary captures high-level metadata about a Signavio process.
type SignavioProcessSummary struct {
	ID           string                       `json:"id"`
	Name         string                       `json:"name"`
	SourceFile   string                       `json:"source_file"`
	ElementCount int                          `json:"element_count"`
	ElementTypes map[string]int               `json:"element_types"`
	Elements     []SignavioElementSummary     `json:"elements,omitempty"`
	Labels       map[string]string            `json:"labels,omitempty"`
	Properties   map[string]map[string]string `json:"properties,omitempty"`
}

// SignavioMetadata aggregates details about processed Signavio artifacts.
type SignavioMetadata struct {
	ProcessCount int                      `json:"process_count"`
	SourceFiles  int                      `json:"source_files"`
	Processes    []SignavioProcessSummary `json:"processes,omitempty"`
	Errors       []string                 `json:"errors,omitempty"`
}

func loadSignavioArtifacts(paths []string, logger *log.Logger) ([]Node, []Edge, SignavioMetadata) {
	metadata := SignavioMetadata{}
	if len(paths) == 0 {
		return nil, nil, metadata
	}

	var nodes []Node
	var edges []Edge
	nodeIDs := make(map[string]struct{})
	edgeIDs := make(map[string]struct{})

	for _, rawPath := range paths {
		path := strings.TrimSpace(rawPath)
		if path == "" {
			continue
		}

		metadata.SourceFiles++

		data, err := os.ReadFile(path)
		if err != nil {
			msg := fmt.Sprintf("failed to read Signavio file %q: %v", path, err)
			if logger != nil {
				logger.Println(msg)
			}
			metadata.Errors = append(metadata.Errors, msg)
			continue
		}

		ext := strings.ToLower(filepath.Ext(path))
		var (
			processNodes     []Node
			processEdges     []Edge
			processSummaries []SignavioProcessSummary
		)

		switch ext {
		case ".json":
			processNodes, processEdges, processSummaries, err = parseSignavioJSON(data, path)
		case ".bpmn", ".xml":
			processNodes, processEdges, processSummaries, err = parseSignavioBPMN(data, path)
		default:
			msg := fmt.Sprintf("unsupported Signavio file format %q for %s", ext, path)
			if logger != nil {
				logger.Println(msg)
			}
			metadata.Errors = append(metadata.Errors, msg)
			continue
		}

		if err != nil {
			msg := fmt.Sprintf("failed to parse Signavio file %q: %v", path, err)
			if logger != nil {
				logger.Println(msg)
			}
			metadata.Errors = append(metadata.Errors, msg)
			continue
		}

		for i := range processNodes {
			if _, exists := nodeIDs[processNodes[i].ID]; exists {
				continue
			}
			nodeIDs[processNodes[i].ID] = struct{}{}
			nodes = append(nodes, processNodes[i])
		}

		for i := range processEdges {
			edgeKey := fmt.Sprintf("%s->%s:%s", processEdges[i].SourceID, processEdges[i].TargetID, processEdges[i].Label)
			if _, exists := edgeIDs[edgeKey]; exists {
				continue
			}
			edgeIDs[edgeKey] = struct{}{}
			edges = append(edges, processEdges[i])
		}

		if len(processSummaries) > 0 {
			metadata.ProcessCount += len(processSummaries)
			metadata.Processes = append(metadata.Processes, processSummaries...)
		}
	}

	return nodes, edges, metadata
}

func parseSignavioJSON(data []byte, path string) ([]Node, []Edge, []SignavioProcessSummary, error) {
	type jsonElement struct {
		ID            string            `json:"id"`
		Type          string            `json:"type"`
		Name          string            `json:"name"`
		Documentation string            `json:"documentation"`
		Properties    map[string]string `json:"properties"`
		Labels        map[string]string `json:"labels"`
		Extra         map[string]any    `json:"-"`
	}
	type jsonFlow struct {
		ID        string `json:"id"`
		SourceRef string `json:"sourceRef"`
		TargetRef string `json:"targetRef"`
		Name      string `json:"name"`
	}
	type jsonProcess struct {
		ID       string        `json:"id"`
		Name     string        `json:"name"`
		Elements []jsonElement `json:"elements"`
		Flows    []jsonFlow    `json:"flows"`
	}
	type jsonContainer struct {
		Processes []jsonProcess `json:"processes"`
	}

	var payload jsonContainer
	if err := json.Unmarshal(data, &payload); err != nil {
		return nil, nil, nil, err
	}

	sourceFile := filepath.Base(path)
	var nodes []Node
	var edges []Edge
	summaries := make([]SignavioProcessSummary, 0, len(payload.Processes))

	for _, proc := range payload.Processes {
		processID := proc.ID
		if strings.TrimSpace(processID) == "" {
			processID = fmt.Sprintf("signavio-process-%d", len(summaries)+1)
		}
		processName := proc.Name
		if strings.TrimSpace(processName) == "" {
			processName = processID
		}

		summary := SignavioProcessSummary{
			ID:           processID,
			Name:         processName,
			SourceFile:   sourceFile,
			ElementTypes: make(map[string]int),
			Properties:   make(map[string]map[string]string),
			Labels:       make(map[string]string),
		}

		for _, element := range proc.Elements {
			elementID := element.ID
			if strings.TrimSpace(elementID) == "" {
				continue
			}

			elementType := strings.ToLower(strings.TrimSpace(element.Type))
			if elementType == "" {
				elementType = "activity"
			}
			elementLabel := strings.TrimSpace(element.Name)
			if elementLabel == "" {
				elementLabel = strings.Title(elementType)
			}

			nodeID := fmt.Sprintf("%s:%s", processID, elementID)
			props := map[string]any{
				"source":       "signavio",
				"process_id":   processID,
				"process_name": processName,
				"element_id":   elementID,
				"element_type": elementType,
				"source_file":  sourceFile,
			}
			if element.Documentation != "" {
				props["documentation"] = element.Documentation
			}
			if len(element.Properties) > 0 {
				props["properties"] = element.Properties
				summary.Properties[elementID] = element.Properties
			}
			if len(element.Labels) > 0 {
				props["labels"] = element.Labels
				summary.Labels[elementID] = elementLabel
			}

			nodes = append(nodes, Node{
				ID:    nodeID,
				Type:  fmt.Sprintf("signavio_%s", elementType),
				Label: elementLabel,
				Props: props,
			})

			summary.ElementCount++
			summary.ElementTypes[elementType]++
			summary.Elements = append(summary.Elements, SignavioElementSummary{
				ID:   elementID,
				Type: elementType,
				Name: elementLabel,
			})
		}

		for _, flow := range proc.Flows {
			if strings.TrimSpace(flow.SourceRef) == "" || strings.TrimSpace(flow.TargetRef) == "" {
				continue
			}

			edgeProps := map[string]any{
				"source":       "signavio",
				"process_id":   processID,
				"process_name": processName,
				"source_file":  sourceFile,
			}
			if flow.Name != "" {
				edgeProps["label"] = flow.Name
			}

			edges = append(edges, Edge{
				SourceID: fmt.Sprintf("%s:%s", processID, flow.SourceRef),
				TargetID: fmt.Sprintf("%s:%s", processID, flow.TargetRef),
				Label:    "signavio_sequence_flow",
				Props:    edgeProps,
			})
		}

		summaries = append(summaries, summary)
	}

	return nodes, edges, summaries, nil
}

func parseSignavioBPMN(data []byte, path string) ([]Node, []Edge, []SignavioProcessSummary, error) {
	decoder := xml.NewDecoder(bytes.NewReader(data))
	sourceFile := filepath.Base(path)

	var nodes []Node
	var edges []Edge
	var summaries []SignavioProcessSummary
	nodeIndex := make(map[string]*Node)

	var currentProcess *SignavioProcessSummary
	processStack := make([]*SignavioProcessSummary, 0)

	for {
		token, err := decoder.Token()
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, nil, nil, err
		}

		switch elem := token.(type) {
		case xml.StartElement:
			local := stripXMLNamespace(elem.Name.Local)
			if local == "process" {
				processID := attr(elem, "id")
				if processID == "" {
					processID = fmt.Sprintf("signavio-process-%d", len(summaries)+1)
				}
				processName := attr(elem, "name")
				if processName == "" {
					processName = processID
				}
				process := SignavioProcessSummary{
					ID:           processID,
					Name:         processName,
					SourceFile:   sourceFile,
					ElementTypes: make(map[string]int),
					Properties:   make(map[string]map[string]string),
					Labels:       make(map[string]string),
				}
				summaries = append(summaries, process)
				currentProcess = &summaries[len(summaries)-1]
				processStack = append(processStack, currentProcess)
				continue
			}

			if currentProcess == nil {
				continue
			}

			if local == "documentation" {
				// Documentation handled during CharData
				continue
			}

			elementID := attr(elem, "id")
			if elementID == "" {
				continue
			}

			if local == "sequenceFlow" {
				sourceRef := attr(elem, "sourceRef")
				targetRef := attr(elem, "targetRef")
				if sourceRef == "" || targetRef == "" {
					continue
				}
				edgeKey := fmt.Sprintf("%s:%s->%s", currentProcess.ID, sourceRef, targetRef)
				edges = append(edges, Edge{
					SourceID: fmt.Sprintf("%s:%s", currentProcess.ID, sourceRef),
					TargetID: fmt.Sprintf("%s:%s", currentProcess.ID, targetRef),
					Label:    "signavio_sequence_flow",
					Props: map[string]any{
						"source":       "signavio",
						"process_id":   currentProcess.ID,
						"process_name": currentProcess.Name,
						"source_file":  sourceFile,
						"edge_id":      edgeKey,
					},
				})
				continue
			}

			nodeKey := fmt.Sprintf("%s:%s", currentProcess.ID, elementID)
			if _, exists := nodeIndex[nodeKey]; exists {
				continue
			}

			elementType := strings.ToLower(local)
			elementName := attr(elem, "name")
			if elementName == "" {
				elementName = strings.Title(elementType)
			}

			props := map[string]any{
				"source":       "signavio",
				"process_id":   currentProcess.ID,
				"process_name": currentProcess.Name,
				"element_id":   elementID,
				"element_type": elementType,
				"source_file":  sourceFile,
			}

			node := Node{
				ID:    nodeKey,
				Type:  fmt.Sprintf("signavio_%s", elementType),
				Label: elementName,
				Props: props,
			}
			nodes = append(nodes, node)
			nodeIndex[nodeKey] = &nodes[len(nodes)-1]

			currentProcess.ElementCount++
			currentProcess.ElementTypes[elementType]++
			currentProcess.Labels[elementID] = elementName
			currentProcess.Elements = append(currentProcess.Elements, SignavioElementSummary{
				ID:   elementID,
				Type: elementType,
				Name: elementName,
			})
		case xml.EndElement:
			local := stripXMLNamespace(elem.Name.Local)
			if local == "process" && len(processStack) > 0 {
				processStack = processStack[:len(processStack)-1]
				if len(processStack) == 0 {
					currentProcess = nil
				} else {
					currentProcess = processStack[len(processStack)-1]
				}
			}
		case xml.CharData:
			// Optionally capture documentation text if needed in the future.
			_ = strings.TrimSpace(string(elem))
		}
	}

	return nodes, edges, summaries, nil
}

func stripXMLNamespace(name string) string {
	if idx := strings.Index(name, ":"); idx != -1 {
		return name[idx+1:]
	}
	return name
}

func attr(el xml.StartElement, name string) string {
	for _, attr := range el.Attr {
		if stripXMLNamespace(attr.Name.Local) == name {
			return attr.Value
		}
	}
	return ""
}
