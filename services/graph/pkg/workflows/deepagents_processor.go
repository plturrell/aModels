package workflows

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/langchain-ai/langgraph-go/pkg/stategraph"
)

// DeepAgentsProcessorOptions configures the deep agents processing workflow.
type DeepAgentsProcessorOptions struct {
	DeepAgentsServiceURL string // URL to DeepAgents service (e.g., "http://deepagents-service:9004")
}

// DeepAgentsRequest represents a request to the deep agent.
type DeepAgentsRequest struct {
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream,omitempty"`
	Config   map[string]any `json:"config,omitempty"`
}

// Message represents a chat message.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// DeepAgentsResponse represents the response from the deep agent.
type DeepAgentsResponse struct {
	Messages []Message     `json:"messages"`
	Result   any           `json:"result,omitempty"`
}

var deepagentsHTTPClient = &http.Client{
	Timeout: 300 * time.Second, // Deep agents can take longer
}

// RunDeepAgentNode returns a node that runs a deep agent.
func RunDeepAgentNode(deepagentsServiceURL string) stategraph.NodeFunc {
	return wrapStateFunc(func(ctx context.Context, state map[string]any) (map[string]any, error) {
		// Extract deep agent request from state
		var agentRequest DeepAgentsRequest

		// Try to get from state directly
		if req, ok := state["deepagents_request"].(map[string]any); ok {
			// Parse messages
			if msgs, ok := req["messages"].([]any); ok {
				agentRequest.Messages = make([]Message, 0, len(msgs))
				for _, msgIface := range msgs {
					if msgMap, ok := msgIface.(map[string]any); ok {
						role, _ := msgMap["role"].(string)
						content, _ := msgMap["content"].(string)
						agentRequest.Messages = append(agentRequest.Messages, Message{
							Role:    role,
							Content: content,
						})
					}
				}
			}
			if stream, ok := req["stream"].(bool); ok {
				agentRequest.Stream = stream
			}
			if config, ok := req["config"].(map[string]any); ok {
				agentRequest.Config = config
			}
		} else {
			// Try to construct from individual fields
			if msg, ok := state["message"].(string); ok {
				agentRequest.Messages = []Message{
					{Role: "user", Content: msg},
				}
			}
		}

		if len(agentRequest.Messages) == 0 {
			log.Println("No deep agent messages provided; skipping execution")
			return state, nil
		}

		if deepagentsServiceURL == "" || deepagentsServiceURL == "offline" {
			deepagentsServiceURL = os.Getenv("DEEPAGENTS_SERVICE_URL")
			if deepagentsServiceURL == "" {
				deepagentsServiceURL = "http://deepagents-service:9004"
			}
		}

		log.Printf("Running deep agent with %d messages", len(agentRequest.Messages))

		// Enrich messages with knowledge graph context if available
		enrichedMessages := make([]Message, len(agentRequest.Messages))
		copy(enrichedMessages, agentRequest.Messages)

		// Add knowledge graph context to the last user message if available
		if kgIface, ok := state["knowledge_graph"].(map[string]any); ok {
			if len(enrichedMessages) > 0 {
				lastMsg := &enrichedMessages[len(enrichedMessages)-1]
				if lastMsg.Role == "user" {
					kgSummary := formatKnowledgeGraphSummary(kgIface)
					if kgSummary != "" {
						lastMsg.Content = fmt.Sprintf("%s\n\nKnowledge Graph Context:\n%s", lastMsg.Content, kgSummary)
					}
				}
			}
		}

		// Prepare request
		requestBody := map[string]any{
			"messages": enrichedMessages,
			"stream":   agentRequest.Stream,
		}
		if agentRequest.Config != nil {
			requestBody["config"] = agentRequest.Config
		}

		body, err := json.Marshal(requestBody)
		if err != nil {
			return nil, fmt.Errorf("marshal deep agent request: %w", err)
		}

		// Call DeepAgents service
		endpoint := strings.TrimRight(deepagentsServiceURL, "/") + "/invoke"
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
		if err != nil {
			return nil, fmt.Errorf("build deep agent request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := deepagentsHTTPClient.Do(req)
		if err != nil {
			return nil, fmt.Errorf("request deep agent: %w", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			bodyBytes, _ := readBodyBytes(resp.Body, 4096)
			return nil, fmt.Errorf("deep agent request failed with status %s: %s",
				resp.Status, strings.TrimSpace(string(bodyBytes)))
		}

		var agentResponse DeepAgentsResponse
		if err := json.NewDecoder(resp.Body).Decode(&agentResponse); err != nil {
			return nil, fmt.Errorf("decode deep agent response: %w", err)
		}

		// Store results in state
		newState := make(map[string]any, len(state)+4)
		for k, v := range state {
			newState[k] = v
		}
		newState["deepagents_result"] = agentResponse
		newState["deepagents_messages"] = agentResponse.Messages
		newState["deepagents_success"] = true
		newState["deepagents_executed_at"] = time.Now().Format(time.RFC3339)

		// Extract text from last assistant message
		if len(agentResponse.Messages) > 0 {
			lastMsg := agentResponse.Messages[len(agentResponse.Messages)-1]
			if lastMsg.Role == "assistant" {
				newState["deepagents_text"] = lastMsg.Content
			}
		}

		log.Printf("Deep agent executed successfully: %d messages returned", len(agentResponse.Messages))

		return newState, nil
	})
}

// formatKnowledgeGraphSummary formats knowledge graph data for inclusion in agent context.
func formatKnowledgeGraphSummary(kg map[string]any) string {
	var parts []string

	if nodes, ok := kg["nodes"].([]any); ok {
		parts = append(parts, fmt.Sprintf("Nodes: %d", len(nodes)))
	}
	if edges, ok := kg["edges"].([]any); ok {
		parts = append(parts, fmt.Sprintf("Edges: %d", len(edges)))
	}
	if quality, ok := kg["quality"].(map[string]any); ok {
		if score, ok := quality["score"].(float64); ok {
			parts = append(parts, fmt.Sprintf("Quality Score: %.2f", score))
		}
		if level, ok := quality["level"].(string); ok {
			parts = append(parts, fmt.Sprintf("Quality Level: %s", level))
		}
	}

	if len(parts) == 0 {
		return ""
	}

	return strings.Join(parts, "\n")
}

// readBodyBytes reads up to limit bytes from response body.
func readBodyBytes(body io.Reader, limit int64) ([]byte, error) {
	if limit <= 0 {
		limit = 4096
	}
	limitedReader := io.LimitReader(body, limit)
	return io.ReadAll(limitedReader)
}

// NewDeepAgentsProcessorWorkflow creates a workflow that processes requests using deep agents.
func NewDeepAgentsProcessorWorkflow(opts DeepAgentsProcessorOptions) (*stategraph.CompiledStateGraph, error) {
	deepagentsServiceURL := opts.DeepAgentsServiceURL
	if deepagentsServiceURL == "" {
		deepagentsServiceURL = os.Getenv("DEEPAGENTS_SERVICE_URL")
		if deepagentsServiceURL == "" {
			deepagentsServiceURL = "http://deepagents-service:9004"
		}
	}

	nodes := map[string]stategraph.NodeFunc{
		"run_deep_agent": RunDeepAgentNode(deepagentsServiceURL),
	}

	edges := []EdgeSpec{}

	return BuildGraph("run_deep_agent", "run_deep_agent", nodes, edges)
}

