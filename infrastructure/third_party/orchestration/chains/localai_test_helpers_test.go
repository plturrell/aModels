package chains

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/llms/localai"
)

// newTestLocalAILLM spins up a lightweight HTTP server that mimics the AI Platform
// runtime and returns the provided responses sequentially.
func newTestLocalAILLM(t *testing.T, responses ...string) (*localai.LLM, func()) {
	t.Helper()

	if len(responses) == 0 {
		responses = []string{"test-response"}
	}

	var (
		mu  sync.Mutex
		idx int
	)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/runtime/models":
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"domains": []map[string]any{},
			})
		case r.Method == http.MethodPost && r.URL.Path == "/runtime/infer":
			mu.Lock()
			current := responses[idx]
			if idx < len(responses)-1 {
				idx++
			}
			mu.Unlock()

			payload := map[string]any{
				"id":      "test-response",
				"object":  "chat.completion",
				"created": time.Now().Unix(),
				"model":   "localai-test",
				"choices": []map[string]any{
					{
						"index": 0,
						"message": map[string]any{
							"role":    "assistant",
							"content": current,
						},
						"finish_reason": "stop",
					},
				},
				"usage": map[string]int{
					"prompt_tokens":     0,
					"completion_tokens": 0,
					"total_tokens":      0,
				},
			}

			w.Header().Set("Content-Type", "application/json")
			require.NoError(t, json.NewEncoder(w).Encode(payload))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))

	llm, err := localai.New(
		localai.WithBaseURL(server.URL),
		localai.WithHTTPClient(server.Client()),
		localai.WithInferencePath("/runtime/infer"),
		localai.WithDomainsPath("/runtime/models"),
	)
	require.NoError(t, err)

	return llm, server.Close
}

