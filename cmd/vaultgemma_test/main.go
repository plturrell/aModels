package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

// ChatMessage represents a message in the conversation
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatRequest represents the request to LocalAI
type ChatRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	Temperature float64       `json:"temperature,omitempty"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
}

// ChatResponse represents the response from LocalAI
type ChatResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index   int `json:"index"`
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// VaultGemmaClient handles communication with LocalAI VaultGemma
type VaultGemmaClient struct {
	BaseURL    string
	HTTPClient *http.Client
}

// NewVaultGemmaClient creates a new client
func NewVaultGemmaClient(baseURL string) *VaultGemmaClient {
	return &VaultGemmaClient{
		BaseURL: baseURL,
		HTTPClient: &http.Client{
			Timeout: 60 * time.Second,
		},
	}
}

// Chat sends a chat completion request
func (c *VaultGemmaClient) Chat(prompt string) (*ChatResponse, error) {
	request := ChatRequest{
		Model: "vaultgemma",
		Messages: []ChatMessage{
			{
				Role:    "user",
				Content: prompt,
			},
		},
		Temperature: 0.7,
		MaxTokens:   1024,
	}

	jsonData, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %v", err)
	}

	url := c.BaseURL + "/v1/chat/completions"
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %v", err)
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %v", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(body))
	}

	var chatResp ChatResponse
	if err := json.Unmarshal(body, &chatResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %v", err)
	}

	return &chatResp, nil
}

// CheckHealth checks if LocalAI server is running
func (c *VaultGemmaClient) CheckHealth() error {
	url := c.BaseURL + "/v1/models"
	resp, err := c.HTTPClient.Get(url)
	if err != nil {
		return fmt.Errorf("failed to connect to LocalAI: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("LocalAI returned status %d", resp.StatusCode)
	}

	return nil
}

func main() {
	// Get LocalAI URL from environment or use default
	localAIURL := os.Getenv("LOCALAI_URL")
	if localAIURL == "" {
		localAIURL = "http://localhost:8080"
	}

	fmt.Println("🚀 VaultGemma Test Client")
	fmt.Println("=" + string(make([]byte, 50)))
	fmt.Printf("LocalAI URL: %s\n\n", localAIURL)

	client := NewVaultGemmaClient(localAIURL)

	// Check if server is running
	fmt.Print("Checking LocalAI server... ")
	if err := client.CheckHealth(); err != nil {
		fmt.Printf("❌ FAILED\n")
		fmt.Printf("Error: %v\n", err)
		fmt.Println("\nMake sure LocalAI is running:")
		fmt.Println("  cd ../agenticAiETH_layer4_LocalAI")
		fmt.Println("  ./local-ai --models-path ./models")
		os.Exit(1)
	}
	fmt.Println("✅ OK")

	// Test with "hello" prompt
	fmt.Println("\n📝 Sending test prompt: 'hello'")
	fmt.Println("-" + string(make([]byte, 50)))

	startTime := time.Now()
	response, err := client.Chat("hello")
	duration := time.Since(startTime)

	if err != nil {
		fmt.Printf("❌ Error: %v\n", err)
		os.Exit(1)
	}

	// Display response
	fmt.Printf("\n✅ Response received in %.2f seconds\n\n", duration.Seconds())

	if len(response.Choices) > 0 {
		content := response.Choices[0].Message.Content
		fmt.Println("🤖 VaultGemma Response:")
		fmt.Println("-" + string(make([]byte, 50)))
		fmt.Println(content)
		fmt.Println("-" + string(make([]byte, 50)))
	}

	// Display usage stats
	fmt.Printf("\n📊 Token Usage:\n")
	fmt.Printf("  Prompt tokens:     %d\n", response.Usage.PromptTokens)
	fmt.Printf("  Completion tokens: %d\n", response.Usage.CompletionTokens)
	fmt.Printf("  Total tokens:      %d\n", response.Usage.TotalTokens)

	// Additional test prompts
	fmt.Println("\n\n🧪 Running additional tests...")

	testPrompts := []string{
		"What is blockchain?",
		"Explain differential privacy in simple terms",
		"How can AI help with smart contract security?",
	}

	for i, prompt := range testPrompts {
		fmt.Printf("\n[Test %d/%d] Prompt: %s\n", i+1, len(testPrompts), prompt)

		resp, err := client.Chat(prompt)
		if err != nil {
			fmt.Printf("  ❌ Error: %v\n", err)
			continue
		}

		if len(resp.Choices) > 0 {
			content := resp.Choices[0].Message.Content
			// Show first 100 chars
			if len(content) > 100 {
				content = content[:100] + "..."
			}
			fmt.Printf("  ✅ Response: %s\n", content)
			fmt.Printf("  📊 Tokens: %d\n", resp.Usage.TotalTokens)
		}
	}

	fmt.Println("\n✅ All tests completed successfully!")
}
