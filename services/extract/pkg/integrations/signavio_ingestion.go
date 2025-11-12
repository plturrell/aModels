package integrations

import (
	"github.com/plturrell/aModels/services/extract/pkg/graph"
)

import (
	"bytes"
	"crypto/sha1"
	"encoding/xml"
	"errors"
	"fmt"
	"strings"
	"time"
)

// SignavioModel represents a normalized Signavio BPMN process.
type SignavioModel struct {
	ID            string
	Name          string
	Lanes         []SignavioLane
	Tasks         []SignavioTask
	Events        []SignavioEvent
	Gateways      []SignavioGateway
	SequenceFlows []SignavioSequenceFlow
}

// SignavioLane represents a BPMN lane.
type SignavioLane struct {
	ID           string
	Name         string
	FlowNodeRefs []string
}

// SignavioTask represents a BPMN task.
type SignavioTask struct {
	ID   string
	Name string
	Type string
}

// SignavioEvent represents a BPMN event (start/end/intermediate).
type SignavioEvent struct {
	ID   string
	Name string
	Type string
}

// SignavioGateway represents a BPMN gateway.
type SignavioGateway struct {
	ID   string
	Name string
	Type string
}

// SignavioSequenceFlow represents a BPMN sequence flow.
type SignavioSequenceFlow struct {
	ID        string
	Name      string
	SourceRef string
	TargetRef string
	Condition string
}

func parseSignavioProcesses(data []byte) ([]SignavioModel, error) {
	if len(bytes.TrimSpace(data)) == 0 {
		return nil, errors.New("empty Signavio BPMN payload")
	}

	decoder := xml.NewDecoder(bytes.NewReader(data))
	decoder.Strict = false

	var defs signavioDefinitions
	if err := decoder.Decode(&defs); err != nil {
		return nil, fmt.Errorf("decode signavio bpmn: %w", err)
	}

	if len(defs.Processes) == 0 {
		return nil, errors.New("signavio bpmn contains no processes")
	}

	models := make([]SignavioModel, 0, len(defs.Processes))
	for _, proc := range defs.Processes {
		model := SignavioModel{
			ID:   strings.TrimSpace(proc.ID),
			Name: strings.TrimSpace(proc.Name),
		}

		for _, laneSet := range proc.LaneSets {
			for _, lane := range laneSet.Lanes {
				model.Lanes = append(model.Lanes, SignavioLane{
					ID:           strings.TrimSpace(lane.ID),
					Name:         strings.TrimSpace(lane.Name),
					FlowNodeRefs: normalizeSignavioRefs(lane.FlowNodeRefs),
				})
			}
		}

		addTasks := func(tasks []signavioTask, taskType string) {
			for _, t := range tasks {
				model.Tasks = append(model.Tasks, SignavioTask{
					ID:   strings.TrimSpace(t.ID),
					Name: strings.TrimSpace(t.Name),
					Type: taskType,
				})
			}
		}

		addTasks(proc.Tasks, "task")
		addTasks(proc.UserTasks, "userTask")
		addTasks(proc.ServiceTasks, "serviceTask")
		addTasks(proc.ManualTasks, "manualTask")
		addTasks(proc.ScriptTasks, "scriptTask")
		addTasks(proc.CallActivities, "callActivity")
		addTasks(proc.BusinessRuleTasks, "businessRuleTask")
		addTasks(proc.SubProcesses, "subProcess")

		addEvents := func(events []signavioEvent, eventType string) {
			for _, e := range events {
				model.Events = append(model.Events, SignavioEvent{
					ID:   strings.TrimSpace(e.ID),
					Name: strings.TrimSpace(e.Name),
					Type: eventType,
				})
			}
		}

		addEvents(proc.StartEvents, "startEvent")
		addEvents(proc.EndEvents, "endEvent")
		addEvents(proc.IntermediateCatchEvents, "intermediateCatchEvent")
		addEvents(proc.IntermediateThrowEvents, "intermediateThrowEvent")

		addGateways := func(gateways []signavioGateway, gatewayType string) {
			for _, g := range gateways {
				model.Gateways = append(model.Gateways, SignavioGateway{
					ID:   strings.TrimSpace(g.ID),
					Name: strings.TrimSpace(g.Name),
					Type: gatewayType,
				})
			}
		}

		addGateways(proc.ExclusiveGateways, "exclusiveGateway")
		addGateways(proc.ParallelGateways, "parallelGateway")
		addGateways(proc.InclusiveGateways, "inclusiveGateway")
		addGateways(proc.EventBasedGateways, "eventBasedGateway")

		for _, flow := range proc.SequenceFlows {
			model.SequenceFlows = append(model.SequenceFlows, SignavioSequenceFlow{
				ID:        strings.TrimSpace(flow.ID),
				Name:      strings.TrimSpace(flow.Name),
				SourceRef: strings.TrimSpace(flow.SourceRef),
				TargetRef: strings.TrimSpace(flow.TargetRef),
				Condition: strings.TrimSpace(flow.ConditionExpression.Text),
			})
		}

		models = append(models, model)
	}

	return models, nil
}

func signavioModelsToGraph(models []SignavioModel, sourceLabel string) ([]graph.Node, []graph.Edge) {
	nodes := make([]graph.Node, 0)
	edges := make([]graph.Edge, 0)

	for _, model := range models {
		processNodeID := fmt.Sprintf("signavio:process:%s", stableSignavioID(model.ID, model.Name))
		processProps := map[string]any{
			"process_id":    model.ID,
			"task_count":    len(model.Tasks),
			"lane_count":    len(model.Lanes),
			"event_count":   len(model.Events),
			"gateway_count": len(model.Gateways),
		}
		if sourceLabel != "" {
			processProps["source"] = sourceLabel
		}

		nodes = append(nodes, graph.Node{
			ID:    processNodeID,
			Type:  "signavio-process",
			Label: fallbackSignavioLabel(model.Name, model.ID),
			Props: processProps,
		})

		nodeIDs := make(map[string]string)
		for _, task := range model.Tasks {
			addSignavioComponent(&nodes, &edges, nodeIDs, processNodeID, task.ID, task.Name, "signavio-task", map[string]any{
				"task_type": task.Type,
			})
		}

		for _, event := range model.Events {
			addSignavioComponent(&nodes, &edges, nodeIDs, processNodeID, event.ID, event.Name, "signavio-event", map[string]any{
				"event_type": event.Type,
			})
		}

		for _, gateway := range model.Gateways {
			addSignavioComponent(&nodes, &edges, nodeIDs, processNodeID, gateway.ID, gateway.Name, "signavio-gateway", map[string]any{
				"gateway_type": gateway.Type,
			})
		}

		laneNodeIDs := make(map[string]string)
		for _, lane := range model.Lanes {
			laneID := fmt.Sprintf("signavio:lane:%s", stableSignavioID(lane.ID, lane.Name))
			nodes = append(nodes, graph.Node{
				ID:    laneID,
				Type:  "signavio-lane",
				Label: fallbackSignavioLabel(lane.Name, lane.ID),
				Props: map[string]any{
					"lane_id": lane.ID,
				},
			})

			edges = append(edges, graph.Edge{
				SourceID: processNodeID,
				TargetID: laneID,
				Label:    "HAS_LANE",
			})

			laneNodeIDs[lane.ID] = laneID
		}

		for _, lane := range model.Lanes {
			laneNodeID := laneNodeIDs[lane.ID]
			if laneNodeID == "" {
				continue
			}
			for _, ref := range lane.FlowNodeRefs {
				if targetNodeID, ok := nodeIDs[ref]; ok {
					edges = append(edges, graph.Edge{
						SourceID: laneNodeID,
						TargetID: targetNodeID,
						Label:    "CONTAINS",
					})
				}
			}
		}

		for _, flow := range model.SequenceFlows {
			sourceNodeID, okSrc := nodeIDs[flow.SourceRef]
			targetNodeID, okTgt := nodeIDs[flow.TargetRef]
			if !okSrc || !okTgt {
				continue
			}

			props := map[string]any{
				"flow_id": flow.ID,
			}
			if flow.Name != "" {
				props["flow_name"] = flow.Name
			}
			if flow.Condition != "" {
				props["condition"] = flow.Condition
			}

			edges = append(edges, graph.Edge{
				SourceID: sourceNodeID,
				TargetID: targetNodeID,
				Label:    "SIGNAVIO_FLOW",
				Props:    props,
			})
		}
	}

	return nodes, edges
}

func addSignavioComponent(nodes *[]graph.Node, edges *[]graph.Edge, nodeIDs map[string]string, processNodeID, rawID, name, nodeType string, props map[string]any) {
	normalizedID := fmt.Sprintf("signavio:node:%s", stableSignavioID(rawID, name))
	if props == nil {
		props = map[string]any{}
	}
	if rawID != "" {
		props["bpmn_id"] = rawID
	}
	*nodes = append(*nodes, graph.Node{
		ID:    normalizedID,
		Type:  nodeType,
		Label: fallbackSignavioLabel(name, rawID),
		Props: props,
	})
	nodeIDs[rawID] = normalizedID
	*edges = append(*edges, graph.Edge{
		SourceID: processNodeID,
		TargetID: normalizedID,
		Label:    "HAS_COMPONENT",
		Props: map[string]any{
			"component_type": nodeType,
		},
	})
}

func buildSignavioSummary(model SignavioModel) string {
	var b strings.Builder
	b.WriteString(fmt.Sprintf("Process %s with %d tasks. ", fallbackSignavioLabel(model.Name, model.ID), len(model.Tasks)))
	if len(model.Tasks) > 0 {
		taskNames := make([]string, 0, len(model.Tasks))
		for _, task := range model.Tasks {
			if task.Name != "" {
				taskNames = append(taskNames, task.Name)
			}
		}
		if len(taskNames) > 0 {
			b.WriteString("Tasks: ")
			b.WriteString(strings.Join(taskNames, ", "))
			b.WriteString(". ")
		}
	}
	if len(model.Events) > 0 {
		eventTypes := make(map[string]int)
		for _, event := range model.Events {
			eventTypes[event.Type]++
		}
		b.WriteString("Events: ")
		fragments := make([]string, 0, len(eventTypes))
		for typ, count := range eventTypes {
			fragments = append(fragments, fmt.Sprintf("%d %s", count, typ))
		}
		b.WriteString(strings.Join(fragments, ", "))
		b.WriteString(". ")
	}
	return b.String()
}

func normalizeSignavioRefs(refs []string) []string {
	result := make([]string, 0, len(refs))
	for _, ref := range refs {
		ref = strings.TrimSpace(ref)
		if ref != "" {
			result = append(result, ref)
		}
	}
	return result
}

func stableSignavioID(id, fallback string) string {
	id = strings.TrimSpace(id)
	if id != "" {
		return sanitizeSignavioID(id)
	}
	fallback = strings.TrimSpace(fallback)
	if fallback == "" {
		fallback = fmt.Sprintf("signavio_%d", time.Now().UnixNano())
	}
	h := sha1.Sum([]byte(fallback))
	return fmt.Sprintf("%x", h[:8])
}

func sanitizeSignavioID(id string) string {
	id = strings.TrimSpace(id)
	if id == "" {
		return ""
	}
	id = strings.ReplaceAll(id, " ", "_")
	id = strings.ReplaceAll(id, "/", "_")
	id = strings.ReplaceAll(id, ":", "_")
	return strings.ToLower(id)
}

func fallbackSignavioLabel(name, id string) string {
	name = strings.TrimSpace(name)
	if name != "" {
		return name
	}
	id = strings.TrimSpace(id)
	if id != "" {
		return id
	}
	return "Signavio Artifact"
}

type signavioDefinitions struct {
	XMLName   xml.Name          `xml:"definitions"`
	Processes []signavioProcess `xml:"process"`
}

type signavioProcess struct {
	ID                      string                    `xml:"id,attr"`
	Name                    string                    `xml:"name,attr"`
	LaneSets                []signavioLaneSet         `xml:"laneSet"`
	Tasks                   []signavioTask            `xml:"task"`
	UserTasks               []signavioTask            `xml:"userTask"`
	ServiceTasks            []signavioTask            `xml:"serviceTask"`
	ManualTasks             []signavioTask            `xml:"manualTask"`
	ScriptTasks             []signavioTask            `xml:"scriptTask"`
	CallActivities          []signavioTask            `xml:"callActivity"`
	BusinessRuleTasks       []signavioTask            `xml:"businessRuleTask"`
	SubProcesses            []signavioTask            `xml:"subProcess"`
	StartEvents             []signavioEvent           `xml:"startEvent"`
	EndEvents               []signavioEvent           `xml:"endEvent"`
	IntermediateCatchEvents []signavioEvent           `xml:"intermediateCatchEvent"`
	IntermediateThrowEvents []signavioEvent           `xml:"intermediateThrowEvent"`
	ExclusiveGateways       []signavioGateway         `xml:"exclusiveGateway"`
	ParallelGateways        []signavioGateway         `xml:"parallelGateway"`
	InclusiveGateways       []signavioGateway         `xml:"inclusiveGateway"`
	EventBasedGateways      []signavioGateway         `xml:"eventBasedGateway"`
	SequenceFlows           []signavioSequenceFlowXML `xml:"sequenceFlow"`
}

type signavioLaneSet struct {
	Lanes []signavioLaneXML `xml:"lane"`
}

type signavioLaneXML struct {
	ID           string   `xml:"id,attr"`
	Name         string   `xml:"name,attr"`
	FlowNodeRefs []string `xml:"flowNodeRef"`
}

type signavioTask struct {
	ID   string `xml:"id,attr"`
	Name string `xml:"name,attr"`
}

type signavioEvent struct {
	ID   string `xml:"id,attr"`
	Name string `xml:"name,attr"`
}

type signavioGateway struct {
	ID   string `xml:"id,attr"`
	Name string `xml:"name,attr"`
}

type signavioSequenceFlowXML struct {
	ID                  string                         `xml:"id,attr"`
	Name                string                         `xml:"name,attr"`
	SourceRef           string                         `xml:"sourceRef,attr"`
	TargetRef           string                         `xml:"targetRef,attr"`
	ConditionExpression signavioConditionExpressionXML `xml:"conditionExpression"`
}

type signavioConditionExpressionXML struct {
	Text string `xml:",chardata"`
}
