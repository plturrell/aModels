package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/storage"
)

type (
	memoryHanaCache struct {
		base    storage.HANACache
		mu      sync.Mutex
		entries map[string]*storage.CacheEntry
		hits    int
		sets    int
	}

	memorySemanticCache struct {
		base    storage.SemanticCache
		mu      sync.Mutex
		entries map[string]*storage.SemanticCacheEntry
		hits    int
		vector  bool
	}

	cacheHarness struct {
		server   *VaultGemmaServer
		http     *httptest.Server
		hana     *memoryHanaCache
		semantic *memorySemanticCache
	}
)

func newMemoryHanaCache() *memoryHanaCache {
	return &memoryHanaCache{entries: make(map[string]*storage.CacheEntry)}
}

func (m *memoryHanaCache) GenerateCacheKey(prompt, model, domain string, temperature float64, maxTokens int, topP float64, topK int) string {
	return (&m.base).GenerateCacheKey(prompt, model, domain, temperature, maxTokens, topP, topK)
}

func (m *memoryHanaCache) Get(_ context.Context, key string) (*storage.CacheEntry, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if entry, ok := m.entries[key]; ok {
		m.hits++
		clone := *entry
		return &clone, nil
	}
	return nil, errors.New("miss")
}

func (m *memoryHanaCache) Set(_ context.Context, entry *storage.CacheEntry) error {
	if entry == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	clone := *entry
	m.entries[entry.CacheKey] = &clone
	m.sets++
	return nil
}

func newMemorySemanticCache() *memorySemanticCache {
	return &memorySemanticCache{entries: make(map[string]*storage.SemanticCacheEntry)}
}

func (m *memorySemanticCache) GenerateCacheKey(prompt, model, domain string, temperature float64, maxTokens int, topP float64, topK int) string {
	return (&m.base).GenerateCacheKey(prompt, model, domain, temperature, maxTokens, topP, topK)
}

func (m *memorySemanticCache) GenerateSemanticHash(prompt string) string {
	return (&m.base).GenerateSemanticHash(prompt)
}

func (m *memorySemanticCache) FindSemanticSimilar(_ context.Context, prompt, model, domain string, _ float64, _ int) ([]*storage.SemanticCacheEntry, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	lower := strings.ToLower(prompt)
	for _, entry := range m.entries {
		if strings.Contains(strings.ToLower(entry.Prompt), lower) || strings.Contains(lower, strings.ToLower(entry.Prompt)) {
			m.hits++
			clone := *entry
			return []*storage.SemanticCacheEntry{&clone}, nil
		}
		if entry.Model == model && entry.Domain == domain {
			m.hits++
			clone := *entry
			return []*storage.SemanticCacheEntry{&clone}, nil
		}
	}
	return nil, nil
}

func (m *memorySemanticCache) Set(_ context.Context, entry *storage.SemanticCacheEntry) error {
	if entry == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	clone := *entry
	m.entries[entry.CacheKey] = &clone
	return nil
}

func (m *memorySemanticCache) CreateTables(context.Context) error { return nil }

func (m *memorySemanticCache) VectorSearchEnabled() bool { return m != nil && m.vector }

func newCacheHarness(t *testing.T, enableSemantic bool) *cacheHarness {
	t.Helper()
	srv := newTestServer()
	hana := newMemoryHanaCache()
	srv.hanaCache = hana
	var semantic *memorySemanticCache
	if enableSemantic {
		semantic = newMemorySemanticCache()
		semantic.vector = true
		srv.semanticCache = semantic
	} else {
		srv.semanticCache = nil
	}
	srv.DisableEnhancedInference()

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", srv.HandleChat)
	mux.HandleFunc("/v1/chat/completions/stream", srv.HandleStreamingChat)
	mux.HandleFunc("/v1/chat/completions/function-calling", srv.HandleFunctionCalling)

	ts := httptest.NewServer(mux)
	return &cacheHarness{server: srv, http: ts, hana: hana, semantic: semantic}
}

func (h *cacheHarness) Close() {
	if h.http != nil {
		h.http.Close()
	}
}

func (h *cacheHarness) post(path string, payload map[string]any) (time.Duration, []byte, int, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return 0, nil, 0, err
	}
	start := time.Now()
	resp, err := http.Post(h.http.URL+path, "application/json", bytes.NewReader(body))
	if err != nil {
		return 0, nil, 0, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, nil, resp.StatusCode, err
	}
	return time.Since(start), data, resp.StatusCode, nil
}

func TestCacheReuseChatSampling(t *testing.T) {
	harness := newCacheHarness(t, false)
	defer harness.Close()

	payload := map[string]any{
		"model":       "auto",
		"messages":    []map[string]string{{"role": "user", "content": "ping cache"}},
		"temperature": 0.3,
		"max_tokens":  32,
		"top_p":       0.7,
		"top_k":       40,
	}

	missDuration, _, status, err := harness.post("/v1/chat/completions", payload)
	if err != nil || status != http.StatusOK {
		t.Fatalf("first chat request failed: status=%d err=%v", status, err)
	}
	hitDuration, _, status, err := harness.post("/v1/chat/completions", payload)
	if err != nil || status != http.StatusOK {
		t.Fatalf("second chat request failed: status=%d err=%v", status, err)
	}

	if harness.hana.hits != 1 {
		t.Fatalf("expected one cache hit, got %d", harness.hana.hits)
	}
	if hitDuration > missDuration {
		t.Fatalf("expected cached request to be faster: miss=%s hit=%s", missDuration, hitDuration)
	}

	payload["top_p"] = 0.8
	_, _, _, _ = harness.post("/v1/chat/completions", payload)
	if harness.hana.hits != 1 {
		t.Fatalf("expected top_p change to bypass cache")
	}
}

func TestCacheReuseStreaming(t *testing.T) {
	harness := newCacheHarness(t, false)
	defer harness.Close()

	payload := map[string]any{
		"model":       "auto",
		"messages":    []map[string]string{{"role": "user", "content": "stream ping"}},
		"temperature": 0.3,
		"max_tokens":  32,
		"top_p":       0.75,
		"top_k":       30,
		"stream":      true,
	}

	missDuration, _, status, err := harness.post("/v1/chat/completions/stream", payload)
	if err != nil || status != http.StatusOK {
		t.Fatalf("stream miss failed: status=%d err=%v", status, err)
	}
	hitDuration, _, status, err := harness.post("/v1/chat/completions/stream", payload)
	if err != nil || status != http.StatusOK {
		t.Fatalf("stream hit failed: status=%d err=%v", status, err)
	}

	if harness.hana.hits < 1 {
		t.Fatalf("expected cache hit for streaming path")
	}
	if hitDuration > missDuration {
		t.Fatalf("expected faster cache hit for streaming: miss=%s hit=%s", missDuration, hitDuration)
	}
}

func TestCacheReuseFunctionCalling(t *testing.T) {
	harness := newCacheHarness(t, false)
	defer harness.Close()

	payload := map[string]any{
		"model":       "auto",
		"messages":    []map[string]any{{"role": "user", "content": "call a tool"}},
		"max_tokens":  32,
		"temperature": 0.25,
		"top_p":       0.65,
		"top_k":       25,
	}

	missDuration, _, status, err := harness.post("/v1/chat/completions/function-calling", payload)
	if err != nil || status != http.StatusOK {
		t.Fatalf("function-calling miss failed: status=%d err=%v", status, err)
	}
	hitDuration, _, status, err := harness.post("/v1/chat/completions/function-calling", payload)
	if err != nil || status != http.StatusOK {
		t.Fatalf("function-calling hit failed: status=%d err=%v", status, err)
	}

	if harness.hana.hits < 1 {
		t.Fatalf("expected cache hit for function calling")
	}
	if hitDuration > missDuration {
		t.Fatalf("expected faster cache hit: miss=%s hit=%s", missDuration, hitDuration)
	}
}

func TestSemanticCacheReuse(t *testing.T) {
	harness := newCacheHarness(t, true)
	defer harness.Close()

	basePayload := map[string]any{
		"model":       "auto",
		"max_tokens":  32,
		"temperature": 0.2,
		"top_p":       0.6,
		"top_k":       20,
	}

	payload1 := map[string]any{}
	for k, v := range basePayload {
		payload1[k] = v
	}
	payload1["messages"] = []map[string]any{{"role": "user", "content": "semantic cache seed"}}

	missDuration, _, status, err := harness.post("/v1/chat/completions", payload1)
	if err != nil || status != http.StatusOK {
		t.Fatalf("semantic miss failed: status=%d err=%v", status, err)
	}

	payload2 := map[string]any{}
	for k, v := range basePayload {
		payload2[k] = v
	}
	payload2["messages"] = []map[string]any{{"role": "user", "content": "semantic cache seed please"}}

	hitDuration, _, status, err := harness.post("/v1/chat/completions", payload2)
	if err != nil || status != http.StatusOK {
		t.Fatalf("semantic hit failed: status=%d err=%v", status, err)
	}

	if harness.semantic == nil {
		t.Fatalf("semantic cache not initialised")
	}
	if harness.semantic.hits < 1 {
		t.Fatalf("expected semantic cache to service similar prompt")
	}
	if harness.hana.hits != 0 {
		t.Fatalf("expected hana cache miss for semantic match")
	}
	if hitDuration > missDuration {
		t.Fatalf("expected semantic cache response to be faster: miss=%s hit=%s", missDuration, hitDuration)
	}
}
