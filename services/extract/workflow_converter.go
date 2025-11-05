package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
)

// LangGraphNode represents a node in a LangGraph workflow.
type LangGraphNode struct {
	ID          string         `json:"id"`
	Type        string         `json:"type"`        // agent, conditional, sql, etc.
	Label       string         `json:"label"`
	Config      map[string]any `json:"config,omitempty"`
	Condition   string         `json:"condition,omitempty"`   // For conditional nodes
	AgentType   string         `json:"agent_type,omitempty"` // For agent nodes
	SQLQuery    string         `json:"sql_query,omitempty"`   // For SQL nodes
	Properties  map[string]any `json:"properties,omitempty"`
}

// LangGraphEdge represents an edge in a LangGraph workflow.
type LangGraphEdge struct {
	Source      string         `json:"source"`
	Target      string         `json:"target"`
	Condition   string         `json:"condition,omitempty"`   // Conditional routing
	Properties  map[string]any `json:"properties,omitempty"`
}

// LangGraphWorkflow represents a complete LangGraph workflow.
type LangGraphWorkflow struct {
	ID          string          `json:"id"`
	Name        string          `json:"name"`
	Nodes       []LangGraphNode `json:"nodes"`
	Edges       []LangGraphEdge `json:"edges"`
	EntryPoint  string          `json:"entry_point"`
	Metadata    map[string]any  `json:"metadata,omitempty"`
}

// AgentFlowNode represents a node in a LangFlow/AgentFlow workflow.
type AgentFlowNode struct {
	ID          string         `json:"id"`
	Type        string         `json:"type"`        // AgentNode, SQLAgent, ConditionalNode, etc.
	Data        map[string]any `json:"data"`
	Position    map[string]float64 `json:"position,omitempty"`
}

// AgentFlowEdge represents an edge in a LangFlow/AgentFlow workflow.
type AgentFlowEdge struct {
	ID          string         `json:"id"`
	Source      string         `json:"source"`
	Target      string         `json:"target"`
	SourceHandle string       `json:"sourceHandle,omitempty"`
	TargetHandle string       `json:"targetHandle,omitempty"`
	Type        string         `json:"type"`        // default, conditional
	Condition   string         `json:"condition,omitempty"`
}

// AgentFlowWorkflow represents a complete LangFlow/AgentFlow workflow.
type AgentFlowWorkflow struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Nodes       []AgentFlowNode `json:"nodes"`
	Edges       []AgentFlowEdge `json:"edges"`
	Metadata    map[string]any   `json:"metadata,omitempty"`
}

// WorkflowConverter converts Petri nets to LangGraph and AgentFlow workflows.
type WorkflowConverter struct {
	logger           *log.Logger
	extractServiceURL string
	useSemanticSearch bool
}

// NewWorkflowConverter creates a new workflow converter.
func NewWorkflowConverter(logger *log.Logger) *WorkflowConverter {
	return &WorkflowConverter{
		logger:            logger,
		useSemanticSearch: os.Getenv("USE_SAP_RPT_EMBEDDINGS") == "true",
	}
}

// SetExtractServiceURL sets the Extract service URL for semantic search.
func (wc *WorkflowConverter) SetExtractServiceURL(url string) {
	wc.extractServiceURL = url
}

// ConvertPetriNetToLangGraph converts a Petri net to a LangGraph workflow.
func (wc *WorkflowConverter) ConvertPetriNetToLangGraph(net *PetriNet) *LangGraphWorkflow {
	workflow := &LangGraphWorkflow{
		ID:        fmt.Sprintf("langgraph_%s", net.ID),
		Name:      fmt.Sprintf("LangGraph: %s", net.Name),
		Nodes:     []LangGraphNode{},
		Edges:     []LangGraphEdge{},
		Metadata: map[string]any{
			"source":         "petri_net",
			"petri_net_id":   net.ID,
			"place_count":    len(net.Places),
			"transition_count": len(net.Transitions),
		},
	}

	nodeMap := make(map[string]string) // Petri net ID -> LangGraph node ID
	nodeCounter := 0

	// Convert places to conditional nodes
	for _, place := range net.Places {
		nodeID := fmt.Sprintf("node_%d", nodeCounter)
		nodeCounter++

		var nodeType string
		var condition string

		if place.Type == "initial" {
			nodeType = "entry"
			condition = "true" // Always proceed from initial
		} else {
			nodeType = "conditional"
			condition = fmt.Sprintf("check_condition('%s')", place.Label)
		}

		node := LangGraphNode{
			ID:        nodeID,
			Type:      nodeType,
			Label:     place.Label,
			Condition: condition,
			Properties: map[string]any{
				"place_id":        place.ID,
				"place_type":      place.Type,
				"initial_tokens":  place.InitialTokens,
				"original_place":  place.Properties,
			},
		}

		workflow.Nodes = append(workflow.Nodes, node)
		nodeMap[place.ID] = nodeID

		// Set entry point if this is the initial place
		if place.Type == "initial" {
			workflow.EntryPoint = nodeID
		}
	}

	// Convert transitions to agent nodes
	for _, transition := range net.Transitions {
		nodeID := fmt.Sprintf("node_%d", nodeCounter)
		nodeCounter++

		// Determine agent type using semantic search and classifications
		agentType := wc.determineAgentType(&transition)

		node := LangGraphNode{
			ID:        nodeID,
			Type:      "agent",
			Label:     transition.Label,
			AgentType: agentType,
			Config: map[string]any{
				"job_name": transition.Label,
				"properties": transition.Properties,
			},
			Properties: map[string]any{
				"transition_id": transition.ID,
				"subprocess_count": len(transition.SubProcesses),
			},
		}

		// Add SQL subprocesses as embedded config
		if len(transition.SubProcesses) > 0 {
			sqlQueries := []string{}
			for _, subProcess := range transition.SubProcesses {
				if subProcess.Type == "sql" {
					sqlQueries = append(sqlQueries, subProcess.Content)
				}
			}
			if len(sqlQueries) > 0 {
				node.Config["sql_queries"] = sqlQueries
				node.SQLQuery = strings.Join(sqlQueries, "; ")
			}
		}

		workflow.Nodes = append(workflow.Nodes, node)
		nodeMap[transition.ID] = nodeID
	}

	// Convert arcs to edges
	for _, arc := range net.Arcs {
		sourceID, sourceExists := nodeMap[arc.Source]
		targetID, targetExists := nodeMap[arc.Target]

		if !sourceExists || !targetExists {
			continue
		}

		edge := LangGraphEdge{
			Source: sourceID,
			Target: targetID,
			Properties: map[string]any{
				"arc_id":     arc.ID,
				"arc_type":   arc.Type,
				"weight":     arc.Weight,
				"original_arc": arc.Properties,
			},
		}

		// Add condition if arc has properties
		if conditionName, ok := arc.Properties["condition_name"].(string); ok {
			edge.Condition = fmt.Sprintf("condition_met('%s')", conditionName)
		}

		workflow.Edges = append(workflow.Edges, edge)
	}

	return workflow
}

// ConvertPetriNetToAgentFlow converts a Petri net to an AgentFlow/LangFlow workflow.
func (wc *WorkflowConverter) ConvertPetriNetToAgentFlow(net *PetriNet) *AgentFlowWorkflow {
	workflow := &AgentFlowWorkflow{
		Name:        fmt.Sprintf("AgentFlow: %s", net.Name),
		Description: fmt.Sprintf("Generated from Petri net: %s", net.ID),
		Nodes:       []AgentFlowNode{},
		Edges:       []AgentFlowEdge{},
		Metadata: map[string]any{
			"source":         "petri_net",
			"petri_net_id":   net.ID,
			"place_count":    len(net.Places),
			"transition_count": len(net.Transitions),
		},
	}

	nodeMap := make(map[string]string) // Petri net ID -> AgentFlow node ID
	nodeCounter := 0
	xPos := 100.0
	yPos := 100.0
	yStep := 150.0

	// Convert places to conditional nodes
	for _, place := range net.Places {
		nodeID := fmt.Sprintf("cond_%d", nodeCounter)
		nodeCounter++

		var nodeType string
		var data map[string]any

		if place.Type == "initial" {
			nodeType = "EntryNode"
			data = map[string]any{
				"type": "entry",
				"label": place.Label,
			}
		} else {
			nodeType = "ConditionalNode"
			data = map[string]any{
				"type": "conditional",
				"label": place.Label,
				"condition": fmt.Sprintf("check_condition('%s')", place.Label),
				"place_id": place.ID,
				"place_type": place.Type,
			}
		}

		node := AgentFlowNode{
			ID:   nodeID,
			Type: nodeType,
			Data: data,
			Position: map[string]float64{
				"x": xPos,
				"y": yPos,
			},
		}

		workflow.Nodes = append(workflow.Nodes, node)
		nodeMap[place.ID] = nodeID
		yPos += yStep
	}

	// Convert transitions to agent nodes
	xPos = 400.0
	yPos = 100.0

	for _, transition := range net.Transitions {
		nodeID := fmt.Sprintf("agent_%d", nodeCounter)
		nodeCounter++

		// Determine agent type using semantic search and classifications
		agentType := wc.determineAgentType(&transition)
		
		// Map agent type to AgentFlow node type
		switch agentType {
		case "sql":
			agentType = "SQLAgent"
		case "python":
			agentType = "PythonAgent"
		case "lookup":
			agentType = "LookupAgent"
		case "etl":
			agentType = "ETLAgent"
		default:
			agentType = "GenericAgent"
		}

		data := map[string]any{
			"type":       agentType,
			"label":      transition.Label,
			"job_name":   transition.Label,
			"properties": transition.Properties,
		}

		// Add SQL subprocesses
		if len(transition.SubProcesses) > 0 {
			sqlQueries := []string{}
			sqlAgents := []map[string]any{}
			
			for i, subProcess := range transition.SubProcesses {
				if subProcess.Type == "sql" {
					sqlQueries = append(sqlQueries, subProcess.Content)
					
					// Create embedded SQL agent
					sqlAgentID := fmt.Sprintf("sql_agent_%d_%d", nodeCounter-1, i)
					sqlAgent := AgentFlowNode{
						ID:   sqlAgentID,
						Type: "SQLAgent",
						Data: map[string]any{
							"type":      "SQLAgent",
							"label":     fmt.Sprintf("%s SQL %d", transition.Label, i+1),
							"sql_query": subProcess.Content,
						},
						Position: map[string]float64{
							"x": xPos + 200,
							"y": yPos + float64(i*50),
						},
					}
					workflow.Nodes = append(workflow.Nodes, sqlAgent)
					sqlAgents = append(sqlAgents, map[string]any{
						"id":   sqlAgentID,
						"sql":  subProcess.Content,
					})
					
					// Connect SQL agent to parent agent
					workflow.Edges = append(workflow.Edges, AgentFlowEdge{
						ID:     fmt.Sprintf("edge_%s_%s", nodeID, sqlAgentID),
						Source: nodeID,
						Target: sqlAgentID,
						Type:   "default",
					})
				}
			}
			
			if len(sqlQueries) > 0 {
				data["sql_queries"] = sqlQueries
				data["sql_agents"] = sqlAgents
			}
		}

		node := AgentFlowNode{
			ID:   nodeID,
			Type: agentType,
			Data: data,
			Position: map[string]float64{
				"x": xPos,
				"y": yPos,
			},
		}

		workflow.Nodes = append(workflow.Nodes, node)
		nodeMap[transition.ID] = nodeID
		yPos += yStep
	}

	// Convert arcs to edges
	edgeCounter := 0
	for _, arc := range net.Arcs {
		sourceID, sourceExists := nodeMap[arc.Source]
		targetID, targetExists := nodeMap[arc.Target]

		if !sourceExists || !targetExists {
			continue
		}

		edgeType := "default"
		if arc.Type == "place_to_transition" {
			edgeType = "conditional"
		}

		edge := AgentFlowEdge{
			ID:     fmt.Sprintf("edge_%d", edgeCounter),
			Source: sourceID,
			Target: targetID,
			Type:   edgeType,
		}

		if conditionName, ok := arc.Properties["condition_name"].(string); ok {
			edge.Condition = conditionName
		}

		workflow.Edges = append(workflow.Edges, edge)
		edgeCounter++
	}

	return workflow
}

// determineAgentType determines agent type using semantic search and classifications
func (wc *WorkflowConverter) determineAgentType(transition *Transition) string {
	// First, check transition properties for explicit type
	if cmd, ok := transition.Properties["command"].(string); ok {
		if strings.Contains(strings.ToUpper(cmd), "SQL") {
			return "sql"
		} else if strings.Contains(strings.ToUpper(cmd), "PYTHON") {
			return "python"
		} else if strings.Contains(strings.ToUpper(cmd), "SHELL") {
			return "shell"
		}
	}

	// Use semantic search to find related tables and determine type
	if wc.useSemanticSearch && wc.extractServiceURL != "" {
		agentType := wc.discoverAgentTypeViaSemantic(transition)
		if agentType != "" {
			return agentType
		}
	}

	// Default fallback
	return "generic"
}

// discoverAgentTypeViaSemantic uses semantic search to discover agent type
func (wc *WorkflowConverter) discoverAgentTypeViaSemantic(transition *Transition) string {
	// Search for tables related to this transition
	query := fmt.Sprintf("tables for %s", transition.Label)
	
	searchPayload := map[string]any{
		"query":           query,
		"artifact_type":   "table",
		"limit":           5,
		"use_semantic":    true,
		"use_hybrid_search": true,
	}
	
	payloadJSON, err := json.Marshal(searchPayload)
	if err != nil {
		wc.logger.Printf("failed to marshal search payload: %v", err)
		return ""
	}
	
	searchURL := fmt.Sprintf("%s/knowledge-graph/search", wc.extractServiceURL)
	resp, err := http.Post(searchURL, "application/json", bytes.NewReader(payloadJSON))
	if err != nil {
		wc.logger.Printf("semantic search failed: %v", err)
		return ""
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return ""
	}
	
	var searchResult struct {
		Results []struct {
			ArtifactID string            `json:"artifact_id"`
			Metadata   map[string]any    `json:"metadata"`
			Score      float64           `json:"score"`
		} `json:"results"`
	}
	
	if err := json.NewDecoder(resp.Body).Decode(&searchResult); err != nil {
		return ""
	}
	
	// Analyze results to determine agent type
	// If results contain transaction tables, likely SQL agent
	// If results contain reference tables, might be lookup agent
	for _, result := range searchResult.Results {
		if metadata := result.Metadata; metadata != nil {
			if classification, ok := metadata["table_classification"].(string); ok {
				switch classification {
				case "transaction":
					return "sql"
				case "reference":
					return "lookup"
				case "staging":
					return "etl"
				}
			}
		}
	}
	
	return ""
}

// RouteByClassification routes workflow based on table classifications
func (wc *WorkflowConverter) RouteByClassification(tableName string, extractServiceURL string) string {
	if extractServiceURL == "" {
		return "default"
	}
	
	// Query classification for this table
	queryPayload := map[string]any{
		"query": fmt.Sprintf(`
			MATCH (n)
			WHERE n.type = 'table' AND n.label = $table_name
			RETURN n.props.table_classification AS classification
		`),
		"params": map[string]any{
			"table_name": tableName,
		},
	}
	
	payloadJSON, err := json.Marshal(queryPayload)
	if err != nil {
		return "default"
	}
	
	queryURL := fmt.Sprintf("%s/knowledge-graph/query", extractServiceURL)
	resp, err := http.Post(queryURL, "application/json", bytes.NewReader(payloadJSON))
	if err != nil {
		return "default"
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return "default"
	}
	
	var queryResult struct {
		Data []map[string]any `json:"data"`
	}
	
	if err := json.NewDecoder(resp.Body).Decode(&queryResult); err != nil {
		return "default"
	}
	
	if len(queryResult.Data) > 0 {
		if classification, ok := queryResult.Data[0]["classification"].(string); ok {
			switch classification {
			case "transaction":
				return "transaction_handler"
			case "reference":
				return "reference_handler"
			case "staging":
				return "staging_handler"
			case "test":
				return "test_handler"
			}
		}
	}
	
	return "default"
}

// ToJSON converts a LangGraph workflow to JSON.
func (lg *LangGraphWorkflow) ToJSON() (string, error) {
	data, err := json.MarshalIndent(lg, "", "  ")
	if err != nil {
		return "", fmt.Errorf("marshal langgraph workflow: %w", err)
	}
	return string(data), nil
}

// ToJSON converts an AgentFlow workflow to JSON.
func (af *AgentFlowWorkflow) ToJSON() (string, error) {
	data, err := json.MarshalIndent(af, "", "  ")
	if err != nil {
		return "", fmt.Errorf("marshal agentflow workflow: %w", err)
	}
	return string(data), nil
}

