package workflow

import (
	"encoding/json"
	"fmt"
	"log"
	"strings"

	"github.com/plturrell/aModels/services/extract/pkg/graph"
	"github.com/plturrell/aModels/services/extract/pkg/integrations"
)

// PetriNet represents a Petri net structure for workflow modeling.
// Petri nets consist of Places (states), Transitions (actions), and Arcs (connections).
type PetriNet struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	Places      []Place           `json:"places"`
	Transitions []Transition      `json:"transitions"`
	Arcs        []Arc             `json:"arcs"`
	Metadata    map[string]any    `json:"metadata,omitempty"`
}

// Place represents a place (state/condition) in a Petri net.
type Place struct {
	ID          string         `json:"id"`
	Label       string         `json:"label"`
	Type        string         `json:"type"`        // condition, state, resource, etc.
	InitialTokens int          `json:"initial_tokens,omitempty"`
	Properties  map[string]any `json:"properties,omitempty"`
}

// Transition represents a transition (action/event) in a Petri net.
type Transition struct {
	ID          string         `json:"id"`
	Label       string         `json:"label"`
	Type        string         `json:"type"`        // job, sql, operation, etc.
	Properties  map[string]any `json:"properties,omitempty"`
	SubProcesses []SubProcess   `json:"sub_processes,omitempty"` // SQL statements, etc.
}

// Arc represents an arc (connection) in a Petri net.
type Arc struct {
	ID          string         `json:"id"`
	Source      string         `json:"source"`      // Place ID or Transition ID
	Target      string         `json:"target"`      // Transition ID or Place ID
	Type        string         `json:"type"`        // place_to_transition, transition_to_place
	Weight      int            `json:"weight,omitempty"` // Token weight (default: 1)
	Properties  map[string]any `json:"properties,omitempty"`
}

// SubProcess represents a mini subprocess within a transition (e.g., SQL statement).
type SubProcess struct {
	ID          string         `json:"id"`
	Type        string         `json:"type"`        // sql, operation, etc.
	Label       string         `json:"label"`
	Content     string         `json:"content,omitempty"`
	Properties  map[string]any `json:"properties,omitempty"`
}

// PetriNetConverter converts Control-M jobs and SQL statements into Petri nets.
type PetriNetConverter struct {
	logger *log.Logger
}

// NewPetriNetConverter creates a new Petri net converter.
func NewPetriNetConverter(logger *log.Logger) *PetriNetConverter {
	return &PetriNetConverter{logger: logger}
}

// ConvertControlMToPetriNet converts Control-M jobs into a Petri net.
func (pnc *PetriNetConverter) ConvertControlMToPetriNet(
	jobs []integrations.ControlMJob,
	sqlQueries map[string][]string, // jobName -> SQL queries
) *PetriNet {
	net := &PetriNet{
		ID:          "controlm_petri_net",
		Name:        "Control-M Workflow Petri Net",
		Places:      []Place{},
		Transitions: []Transition{},
		Arcs:        []Arc{},
		Metadata: map[string]any{
			"source": "controlm",
			"job_count": len(jobs),
		},
	}

	placeMap := make(map[string]string) // condition name -> place ID
	transitionMap := make(map[string]string) // job name -> transition ID

	// Create places for all conditions (InCond and OutCond)
	for _, job := range jobs {
		for _, inCond := range job.InConds {
			condName := inCond.Name
			if _, exists := placeMap[condName]; !exists {
				placeID := fmt.Sprintf("place:cond:%s", sanitizeID(condName))
				placeMap[condName] = placeID
				net.Places = append(net.Places, Place{
					ID:    placeID,
					Label: condName,
					Type:  "condition",
					Properties: map[string]any{
						"condition_type": "input",
						"condition_name": condName,
						"odate":          inCond.ODate,
						"sign":           inCond.Sign,
						"and_or":         inCond.AndOr,
					},
				})
			}
		}

		for _, outCond := range job.OutConds {
			condName := outCond.Name
			if _, exists := placeMap[condName]; !exists {
				placeID := fmt.Sprintf("place:cond:%s", sanitizeID(condName))
				placeMap[condName] = placeID
				net.Places = append(net.Places, Place{
					ID:    placeID,
					Label: condName,
					Type:  "condition",
					Properties: map[string]any{
						"condition_type": "output",
						"condition_name": condName,
						"odate":          outCond.ODate,
						"sign":           outCond.Sign,
						"type":           outCond.Type,
					},
				})
			}
		}
	}

	// Create transitions for all jobs
	for _, job := range jobs {
		transitionID := fmt.Sprintf("transition:job:%s", sanitizeID(job.JobName))
		transitionMap[job.JobName] = transitionID

		// Extract SQL subprocesses for this job
		subProcesses := []SubProcess{}
		if sqls, exists := sqlQueries[job.JobName]; exists {
			for i, sql := range sqls {
				subProcesses = append(subProcesses, SubProcess{
					ID:   fmt.Sprintf("subprocess:%s:sql:%d", sanitizeID(job.JobName), i),
					Type: "sql",
					Label: fmt.Sprintf("SQL %d", i+1),
					Content: sql,
					Properties: map[string]any{
						"sql_query": sql,
						"order":     i,
					},
				})
			}
		}

		// Also check if command contains SQL
		if strings.Contains(strings.ToUpper(job.Command), "SQL") ||
			strings.Contains(strings.ToUpper(job.Command), "SELECT") ||
			strings.Contains(strings.ToUpper(job.Command), "INSERT") ||
			strings.Contains(strings.ToUpper(job.Command), "UPDATE") {
			subProcesses = append(subProcesses, SubProcess{
				ID:   fmt.Sprintf("subprocess:%s:command", sanitizeID(job.JobName)),
				Type: "sql",
				Label: "Command SQL",
				Content: job.Command,
				Properties: map[string]any{
					"command": job.Command,
					"source":  "command",
				},
			})
		}

		transition := Transition{
			ID:          transitionID,
			Label:       job.JobName,
			Type:        "job",
			Properties:   job.Properties(),
			SubProcesses: subProcesses,
		}

		net.Transitions = append(net.Transitions, transition)

		// Create arcs: InCond (places) -> Job (transition)
		for _, inCond := range job.InConds {
			placeID := placeMap[inCond.Name]
			arcID := fmt.Sprintf("arc:%s:%s", placeID, transitionID)
			net.Arcs = append(net.Arcs, Arc{
				ID:     arcID,
				Source: placeID,
				Target: transitionID,
				Type:   "place_to_transition",
				Weight: 1,
				Properties: map[string]any{
					"condition_name": inCond.Name,
					"and_or":          inCond.AndOr,
				},
			})
		}

		// Create arcs: Job (transition) -> OutCond (places)
		for _, outCond := range job.OutConds {
			placeID := placeMap[outCond.Name]
			arcID := fmt.Sprintf("arc:%s:%s", transitionID, placeID)
			net.Arcs = append(net.Arcs, Arc{
				ID:     arcID,
				Source: transitionID,
				Target: placeID,
				Type:   "transition_to_place",
				Weight: 1,
				Properties: map[string]any{
					"condition_name": outCond.Name,
				},
			})
		}
	}

	// Add initial places (jobs without InConds can start immediately)
	initialPlaceID := "place:initial"
	net.Places = append(net.Places, Place{
		ID:    initialPlaceID,
		Label: "Initial State",
		Type:  "initial",
		InitialTokens: 1,
		Properties: map[string]any{
			"initial": true,
		},
	})

	// Connect initial place to jobs without InConds
	for _, job := range jobs {
		if len(job.InConds) == 0 {
			transitionID := transitionMap[job.JobName]
			arcID := fmt.Sprintf("arc:%s:%s", initialPlaceID, transitionID)
			net.Arcs = append(net.Arcs, Arc{
				ID:     arcID,
				Source: initialPlaceID,
				Target: transitionID,
				Type:   "place_to_transition",
				Weight: 1,
				Properties: map[string]any{
					"initial": true,
				},
			})
		}
	}

	return net
}

// ConvertSQLToPetriNetSubprocess converts SQL statements into a mini Petri net subprocess.
func (pnc *PetriNetConverter) ConvertSQLToPetriNetSubprocess(
	sql string,
	subprocessID string,
) *PetriNet {
	net := &PetriNet{
		ID:          subprocessID,
		Name:        fmt.Sprintf("SQL Subprocess: %s", subprocessID),
		Places:      []Place{},
		Transitions: []Transition{},
		Arcs:        []Arc{},
		Metadata: map[string]any{
			"source": "sql",
			"type":   "subprocess",
		},
	}

	// Parse SQL to extract table operations
	// Create places for input tables
	// Create transitions for SQL operations
	// Create places for output tables

	// For now, create a simple structure
	// Input place
	inputPlaceID := "place:input"
	net.Places = append(net.Places, Place{
		ID:    inputPlaceID,
		Label: "Input",
		Type:  "input",
		Properties: map[string]any{
			"sql": sql,
		},
	})

	// SQL operation transition
	transitionID := "transition:sql_op"
	net.Transitions = append(net.Transitions, Transition{
		ID:    transitionID,
		Label: "SQL Operation",
		Type:  "sql",
		Properties: map[string]any{
			"sql_query": sql,
		},
	})

	// Output place
	outputPlaceID := "place:output"
	net.Places = append(net.Places, Place{
		ID:    outputPlaceID,
		Label: "Output",
		Type:  "output",
		Properties: map[string]any{
			"sql": sql,
		},
	})

	// Arcs
	net.Arcs = append(net.Arcs, Arc{
		ID:     "arc:input:op",
		Source: inputPlaceID,
		Target: transitionID,
		Type:   "place_to_transition",
		Weight: 1,
	})

	net.Arcs = append(net.Arcs, Arc{
		ID:     "arc:op:output",
		Source: transitionID,
		Target: outputPlaceID,
		Type:   "transition_to_place",
		Weight: 1,
	})

	return net
}

// PetriNetToGraphNodes converts a Petri net into knowledge graph nodes and edges.
func (pnc *PetriNetConverter) PetriNetToGraphNodes(net *PetriNet) ([]graph.Node, []graph.Edge, string) {
	nodes := []graph.Node{}
	edges := []graph.Edge{}
	rootID := fmt.Sprintf("petri_net:%s", net.ID)

	// Root node for the Petri net
	nodes = append(nodes, graph.Node{
		ID:    rootID,
		Type:  "petri_net",
		Label: net.Name,
		Props: map[string]any{
			"petri_net_id": net.ID,
			"name":         net.Name,
			"place_count":  len(net.Places),
			"transition_count": len(net.Transitions),
			"arc_count":    len(net.Arcs),
			"metadata":     net.Metadata,
		},
	})

	// Places as nodes
	for _, place := range net.Places {
		placeNodeID := fmt.Sprintf("petri_place:%s", place.ID)
		nodes = append(nodes, graph.Node{
			ID:    placeNodeID,
			Type:  "petri_place",
			Label: place.Label,
			Props: map[string]any{
				"place_id":        place.ID,
				"place_type":      place.Type,
				"initial_tokens":  place.InitialTokens,
				"properties":      place.Properties,
			},
		})

		// Connect place to Petri net root
		edges = append(edges, graph.Edge{
			SourceID: rootID,
			TargetID: placeNodeID,
			Label:    "HAS_PLACE",
		})
	}

	// Transitions as nodes
	for _, transition := range net.Transitions {
		transitionNodeID := fmt.Sprintf("petri_transition:%s", transition.ID)
		nodes = append(nodes, graph.Node{
			ID:    transitionNodeID,
			Type:  "petri_transition",
			Label: transition.Label,
			Props: map[string]any{
				"transition_id":   transition.ID,
				"transition_type": transition.Type,
				"properties":      transition.Properties,
				"subprocess_count": len(transition.SubProcesses),
			},
		})

		// Connect transition to Petri net root
		edges = append(edges, graph.Edge{
			SourceID: rootID,
			TargetID: transitionNodeID,
			Label:    "HAS_TRANSITION",
		})

		// Subprocesses as nodes
		for _, subProcess := range transition.SubProcesses {
			subProcessNodeID := fmt.Sprintf("petri_subprocess:%s", subProcess.ID)
			nodes = append(nodes, graph.Node{
				ID:    subProcessNodeID,
				Type:  "petri_subprocess",
				Label: subProcess.Label,
				Props: map[string]any{
					"subprocess_id":   subProcess.ID,
					"subprocess_type": subProcess.Type,
					"content":         subProcess.Content,
					"properties":       subProcess.Properties,
				},
			})

			// Connect subprocess to transition
			edges = append(edges, graph.Edge{
				SourceID: transitionNodeID,
				TargetID: subProcessNodeID,
				Label:    "HAS_SUBPROCESS",
			})
		}
	}

	// Arcs as edges
	for _, arc := range net.Arcs {
		sourceNodeID := ""
		targetNodeID := ""

		// Find source node
		if strings.HasPrefix(arc.Source, "place:") {
			sourceNodeID = fmt.Sprintf("petri_place:%s", arc.Source)
		} else if strings.HasPrefix(arc.Source, "transition:") {
			sourceNodeID = fmt.Sprintf("petri_transition:%s", arc.Source)
		}

		// Find target node
		if strings.HasPrefix(arc.Target, "place:") {
			targetNodeID = fmt.Sprintf("petri_place:%s", arc.Target)
		} else if strings.HasPrefix(arc.Target, "transition:") {
			targetNodeID = fmt.Sprintf("petri_transition:%s", arc.Target)
		}

		if sourceNodeID != "" && targetNodeID != "" {
			edges = append(edges, graph.Edge{
				SourceID: sourceNodeID,
				TargetID: targetNodeID,
				Label:    "PETRI_ARC",
				Props: map[string]any{
					"arc_id":     arc.ID,
					"arc_type":   arc.Type,
					"weight":     arc.Weight,
					"properties": arc.Properties,
				},
			})
		}
	}

	return nodes, edges, rootID
}

// ExportPetriNetToCatalog exports Petri net to catalog format.
func (pnc *PetriNetConverter) ExportPetriNetToCatalog(net *PetriNet) map[string]any {
	return map[string]any{
		"id":          net.ID,
		"name":        net.Name,
		"type":        "petri_net",
		"places":      net.Places,
		"transitions": net.Transitions,
		"arcs":        net.Arcs,
		"metadata":    net.Metadata,
		"statistics": map[string]any{
			"place_count":      len(net.Places),
			"transition_count": len(net.Transitions),
			"arc_count":        len(net.Arcs),
		},
	}
}

// Helper functions

func sanitizeID(id string) string {
	// Replace special characters with underscores
	id = strings.ReplaceAll(id, " ", "_")
	id = strings.ReplaceAll(id, "-", "_")
	id = strings.ReplaceAll(id, ".", "_")
	id = strings.ReplaceAll(id, ":", "_")
	id = strings.ReplaceAll(id, "/", "_")
	return id
}

// ToJSON converts a Petri net to JSON string.
func (pn *PetriNet) ToJSON() (string, error) {
	data, err := json.MarshalIndent(pn, "", "  ")
	if err != nil {
		return "", fmt.Errorf("marshal petri net: %w", err)
	}
	return string(data), nil
}

