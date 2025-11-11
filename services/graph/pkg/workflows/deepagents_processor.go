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
	"github.com/langchain-ai/langgraph-go/pkg/integration"
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
	Messages         []Message   `json:"messages"`
	StructuredOutput map[string]any `json:"structured_output,omitempty"`
	ValidationErrors []string    `json:"validation_errors,omitempty"`
	Result           any         `json:"result,omitempty"`
}

var deepagentsHTTPClient = &http.Client{
	Timeout: 300 * time.Second, // Deep agents can take longer
}

// RunDeepAgentNode returns a node that runs a deep agent.
//
// This node:
// - Extracts agent request from state
// - Enriches messages with knowledge graph context (if available)
// - Calls DeepAgents service HTTP API
// - Uses retry logic for resilience
// - Logs operation with correlation ID and duration
//
// State Input:
//   - deepagents_request.messages: Chat messages for the agent
//   - deepagents_request.stream: Whether to stream responses
//   - deepagents_request.config: Optional agent configuration
//   - knowledge_graph: Optional KG context to enrich messages
//
// State Output:
//   - deepagents_result: Agent response with messages and result
//   - deepagents_messages: Agent response messages
//   - deepagents_text: Extracted text from last assistant message
//   - deepagents_success: Execution success flag
//   - deepagents_executed_at: Execution timestamp
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

		// Start logged operation with correlation ID
		op := integration.StartOperation(ctx, log.Default(), "deepagents.invoke")
		defer op.End(nil)

		op.Log("Running deep agent with %d messages", len(agentRequest.Messages))

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

		// Prepare request - check if structured output is requested
		useStructured := false
		var responseFormat map[string]any
		if config, ok := state["deepagents_config"].(map[string]any); ok {
			if rf, ok := config["response_format"].(map[string]any); ok {
				useStructured = true
				responseFormat = rf
			} else if rfStr, ok := config["response_format"].(string); ok && rfStr == "json" {
				useStructured = true
				responseFormat = map[string]any{"type": "json"}
			}
		}

		requestBody := map[string]any{
			"messages": enrichedMessages,
			"stream":   agentRequest.Stream,
		}
		if agentRequest.Config != nil {
			requestBody["config"] = agentRequest.Config
		}
		if useStructured && responseFormat != nil {
			requestBody["response_format"] = responseFormat
		}

		// Call DeepAgents service with retry logic
		endpoint := strings.TrimRight(deepagentsServiceURL, "/")
		if useStructured {
			endpoint += "/invoke/structured"
		} else {
			endpoint += "/invoke"
		}
		
		var agentResponse DeepAgentsResponse
		err := integration.RetryWithBackoffResult(
			ctx,
			integration.DefaultRetryConfig(),
			log.Default(),
			"deepagents.invoke.execute",
			func() (DeepAgentsResponse, error) {
				body, err := json.Marshal(requestBody)
				if err != nil {
					return agentResponse, fmt.Errorf("marshal deep agent request: %w", err)
				}

				req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
				if err != nil {
					return agentResponse, fmt.Errorf("build deep agent request: %w", err)
				}
				req.Header.Set("Content-Type", "application/json")

				startTime := time.Now()
				resp, err := deepagentsHTTPClient.Do(req)
				duration := time.Since(startTime)
				
				if err != nil {
					integration.LogHTTPRequest(ctx, log.Default(), "POST", endpoint, 0, duration, err)
					return agentResponse, fmt.Errorf("request deep agent: %w", err)
				}
				defer resp.Body.Close()

				integration.LogHTTPRequest(ctx, log.Default(), "POST", endpoint, resp.StatusCode, duration, nil)

				if resp.StatusCode != http.StatusOK {
					bodyBytes, _ := readBodyBytes(resp.Body, 4096)
					err := fmt.Errorf("deep agent request failed with status %s: %s",
						resp.Status, strings.TrimSpace(string(bodyBytes)))
					// Check if retryable
					if integration.IsRetryableHTTPStatus(resp.StatusCode) {
						return agentResponse, err
					}
					// Non-retryable error
					return agentResponse, fmt.Errorf("%w (non-retryable)", err)
				}

				var response DeepAgentsResponse
				if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
					return agentResponse, fmt.Errorf("decode deep agent response: %w", err)
				}
				return response, nil
			},
		)
		if err != nil {
			op.End(err)
			return nil, fmt.Errorf("execute deep agent: %w", err)
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

		// Extract structured output if available
		if agentResponse.StructuredOutput != nil {
			newState["deepagents_structured"] = agentResponse.StructuredOutput
		}

		// Extract text from last assistant message
		if len(agentResponse.Messages) > 0 {
			lastMsg := agentResponse.Messages[len(agentResponse.Messages)-1]
			if lastMsg.Role == "assistant" {
				newState["deepagents_text"] = lastMsg.Content
			}
		}

		op.Log("Deep agent executed successfully: %d messages returned", len(agentResponse.Messages))
		op.End(nil)

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

