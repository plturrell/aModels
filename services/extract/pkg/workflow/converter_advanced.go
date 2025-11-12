package workflow

import (
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"time"
)

// AdvancedWorkflowConverter extends WorkflowConverter with multi-agent support,
// dynamic agent spawning, parallel execution, and checkpointing.
type AdvancedWorkflowConverter struct {
	*WorkflowConverter
	enableParallelExecution bool
	enableCheckpointing     bool
	enableDynamicSpawning   bool
	maxParallelAgents       int
	checkpointInterval      time.Duration
}

// NewAdvancedWorkflowConverter creates a new advanced workflow converter.
func NewAdvancedWorkflowConverter(logger *log.Logger) *AdvancedWorkflowConverter {
	return &AdvancedWorkflowConverter{
		WorkflowConverter:       NewWorkflowConverter(logger),
		enableParallelExecution: os.Getenv("ENABLE_PARALLEL_WORKFLOWS") == "true",
		enableCheckpointing:     os.Getenv("ENABLE_WORKFLOW_CHECKPOINTING") == "true",
		enableDynamicSpawning:   os.Getenv("ENABLE_DYNAMIC_AGENT_SPAWNING") == "true",
		maxParallelAgents:       parseIntEnv(os.Getenv("MAX_PARALLEL_AGENTS"), 10),
		checkpointInterval:      parseDurationEnv(os.Getenv("WORKFLOW_CHECKPOINT_INTERVAL"), 30*time.Second),
	}
}

// ConvertPetriNetToAdvancedLangGraph converts a Petri net to an advanced LangGraph workflow
// with multi-agent support, parallel execution, and checkpointing.
func (awc *AdvancedWorkflowConverter) ConvertPetriNetToAdvancedLangGraph(net *PetriNet) *AdvancedLangGraphWorkflow {
	workflow := &AdvancedLangGraphWorkflow{
		LangGraphWorkflow: *awc.ConvertPetriNetToLangGraph(net),
		ParallelBranches:  []ParallelBranch{},
		Checkpoints:       []Checkpoint{},
		AgentGroups:       []AgentGroup{},
	}

	if workflow.Metadata == nil {
		workflow.Metadata = make(map[string]any)
	}
	workflow.Metadata["advanced_features"] = map[string]any{
		"parallel_execution": awc.enableParallelExecution,
		"checkpointing":      awc.enableCheckpointing,
		"dynamic_spawning":   awc.enableDynamicSpawning,
	}

	// Identify parallel branches (transitions that can run concurrently)
	if awc.enableParallelExecution {
		parallelBranches := awc.identifyParallelBranches(net)
		workflow.ParallelBranches = parallelBranches
	}

	// Add checkpoints for state persistence
	if awc.enableCheckpointing {
		checkpoints := awc.createCheckpoints(net, workflow)
		workflow.Checkpoints = checkpoints
	}

	// Group agents by type for coordination
	agentGroups := awc.groupAgentsByType(workflow.Nodes)
	workflow.AgentGroups = agentGroups

	// Add dynamic agent spawning nodes if enabled
	if awc.enableDynamicSpawning {
		spawnNodes := awc.createDynamicSpawnNodes(net)
		workflow.Nodes = append(workflow.Nodes, spawnNodes...)
	}

	return workflow
}

// AdvancedLangGraphWorkflow extends LangGraphWorkflow with advanced features.
type AdvancedLangGraphWorkflow struct {
	LangGraphWorkflow
	ParallelBranches []ParallelBranch `json:"parallel_branches,omitempty"`
	Checkpoints      []Checkpoint     `json:"checkpoints,omitempty"`
	AgentGroups      []AgentGroup     `json:"agent_groups,omitempty"`
}

// ParallelBranch represents a branch that can execute in parallel.
type ParallelBranch struct {
	ID          string   `json:"id"`
	Name        string   `json:"name"`
	NodeIDs     []string `json:"node_ids"`
	JoinNodeID  string   `json:"join_node_id,omitempty"`
	Condition   string   `json:"condition,omitempty"`
	MaxParallel int      `json:"max_parallel,omitempty"`
}

// Checkpoint represents a state checkpoint in the workflow.
type Checkpoint struct {
	ID          string         `json:"id"`
	NodeID      string         `json:"node_id"`
	StateKeys   []string       `json:"state_keys"`
	Description string         `json:"description"`
	Config      map[string]any `json:"config,omitempty"`
}

// AgentGroup represents a group of agents that work together.
type AgentGroup struct {
	ID          string   `json:"id"`
	Name        string   `json:"name"`
	AgentIDs    []string `json:"agent_ids"`
	AgentType   string   `json:"agent_type"`
	Coordinator string   `json:"coordinator,omitempty"`
}

// identifyParallelBranches identifies transitions that can execute in parallel.
func (awc *AdvancedWorkflowConverter) identifyParallelBranches(net *PetriNet) []ParallelBranch {
	branches := []ParallelBranch{}

	// Find transitions that share the same input place (can run in parallel)
	placeToTransitions := make(map[string][]string)
	for _, arc := range net.Arcs {
		if arc.Type == "place_to_transition" {
			placeID := arc.Source
			transitionID := arc.Target
			placeToTransitions[placeID] = append(placeToTransitions[placeID], transitionID)
		}
	}

	branchCounter := 0
	for placeID, transitionIDs := range placeToTransitions {
		if len(transitionIDs) > 1 {
			// Multiple transitions from same place = parallel branch
			branchID := fmt.Sprintf("parallel_branch_%d", branchCounter)
			branchCounter++

			// Convert transition IDs to node IDs
			nodeIDs := []string{}
			for _, transitionID := range transitionIDs {
				// Find corresponding node ID (would need node mapping)
				nodeID := fmt.Sprintf("node_%s", transitionID)
				nodeIDs = append(nodeIDs, nodeID)
			}

			branch := ParallelBranch{
				ID:          branchID,
				Name:        fmt.Sprintf("Parallel Branch from Place %s", placeID),
				NodeIDs:     nodeIDs,
				MaxParallel: awc.maxParallelAgents,
			}

			// Create join node for this branch
			joinNodeID := fmt.Sprintf("join_%s", branchID)
			branch.JoinNodeID = joinNodeID

			branches = append(branches, branch)
		}
	}

	return branches
}

// createCheckpoints creates checkpoint nodes for state persistence.
func (awc *AdvancedWorkflowConverter) createCheckpoints(
	net *PetriNet,
	workflow *AdvancedLangGraphWorkflow,
) []Checkpoint {
	checkpoints := []Checkpoint{}

	// Create checkpoint before major transitions
	checkpointCounter := 0
	for i, node := range workflow.Nodes {
		if node.Type == "agent" && i%5 == 0 { // Every 5th agent node
			checkpointID := fmt.Sprintf("checkpoint_%d", checkpointCounter)
			checkpointCounter++

			checkpoint := Checkpoint{
				ID:          checkpointID,
				NodeID:      node.ID,
				StateKeys:   []string{"workflow_state", "agent_results", "error_count"},
				Description: fmt.Sprintf("Checkpoint before %s", node.Label),
				Config: map[string]any{
					"interval":  awc.checkpointInterval.String(),
					"auto_save": true,
				},
			}

			checkpoints = append(checkpoints, checkpoint)
		}
	}

	return checkpoints
}

// groupAgentsByType groups agents by their type for coordination.
func (awc *AdvancedWorkflowConverter) groupAgentsByType(nodes []LangGraphNode) []AgentGroup {
	agentGroups := make(map[string][]string)
	agentTypes := make(map[string]string)

	for _, node := range nodes {
		if node.Type == "agent" {
			agentType := node.AgentType
			if agentType == "" {
				agentType = "generic"
			}

			if _, exists := agentGroups[agentType]; !exists {
				agentGroups[agentType] = []string{}
			}
			agentGroups[agentType] = append(agentGroups[agentType], node.ID)
			agentTypes[node.ID] = agentType
		}
	}

	groups := []AgentGroup{}
	groupCounter := 0

	for agentType, agentIDs := range agentGroups {
		if len(agentIDs) > 0 {
			groupID := fmt.Sprintf("agent_group_%d", groupCounter)
			groupCounter++

			group := AgentGroup{
				ID:        groupID,
				Name:      fmt.Sprintf("%s Agents", strings.ToUpper(agentType)),
				AgentIDs:  agentIDs,
				AgentType: agentType,
			}

			// Assign coordinator if multiple agents
			if len(agentIDs) > 1 {
				group.Coordinator = agentIDs[0] // First agent as coordinator
			}

			groups = append(groups, group)
		}
	}

	return groups
}

// createDynamicSpawnNodes creates nodes for dynamic agent spawning.
func (awc *AdvancedWorkflowConverter) createDynamicSpawnNodes(net *PetriNet) []LangGraphNode {
	spawnNodes := []LangGraphNode{}

	// Create spawn node for each transition that might need dynamic spawning
	for i, transition := range net.Transitions {
		// Check if transition needs dynamic spawning (e.g., based on data volume)
		if awc.shouldSpawnDynamically(&transition) {
			spawnNodeID := fmt.Sprintf("spawn_%d", i)

			spawnNode := LangGraphNode{
				ID:    spawnNodeID,
				Type:  "spawn",
				Label: fmt.Sprintf("Spawn Agents for %s", transition.Label),
				Config: map[string]any{
					"target_transition": transition.ID,
					"spawn_condition":   "data_volume > threshold",
					"max_agents":        awc.maxParallelAgents,
					"agent_type":        awc.determineAgentType(&transition),
				},
				Properties: map[string]any{
					"dynamic_spawning": true,
					"spawn_strategy":   "load_balanced",
				},
			}

			spawnNodes = append(spawnNodes, spawnNode)
		}
	}

	return spawnNodes
}

// shouldSpawnDynamically determines if a transition should use dynamic agent spawning.
func (awc *AdvancedWorkflowConverter) shouldSpawnDynamically(transition *Transition) bool {
	// Check properties for spawning hints
	if props := transition.Properties; props != nil {
		if dataVolume, ok := props["data_volume"].(float64); ok {
			return dataVolume > 1000 // Spawn if data volume > 1000
		}
		if requiresParallel, ok := props["requires_parallel"].(bool); ok {
			return requiresParallel
		}
	}

	return false
}

func parseDurationEnv(envVar string, defaultValue time.Duration) time.Duration {
	if envVar == "" {
		return defaultValue
	}
	if duration, err := time.ParseDuration(envVar); err == nil {
		return duration
	}
	return defaultValue
}
