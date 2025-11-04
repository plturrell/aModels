package langchainservice

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"strings"
	"time"
)

// Client wraps access to the Python LangChain service.
type Client struct {
	baseURL    string
	httpClient *http.Client
}

// NewClient creates a client for the LangChain service.
func NewClient(baseURL string, opts ...ClientOption) *Client {
	c := &Client{
		baseURL: strings.TrimRight(baseURL, "/"),
		httpClient: &http.Client{
			Timeout: 15 * time.Second,
		},
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// ClientOption customises a Client.
type ClientOption func(*Client)

// WithHTTPClient overrides the underlying http.Client.
func WithHTTPClient(client *http.Client) ClientOption {
	return func(c *Client) {
		c.httpClient = client
	}
}

type healthResponse struct {
    Status    string
    Documents int
    Meta      map[string]any
}

type apiError struct {
    Message string `json:"message"`
    Code    int    `json:"code"`
}

type healthEnvelope struct {
    Status string `json:"status"`
    Data   struct {
        ServiceStatus string `json:"service_status"`
        Documents     int    `json:"documents"`
    } `json:"data"`
    Meta  map[string]any `json:"meta"`
    Error *apiError       `json:"error,omitempty"`
}

// Health checks the service health endpoint.
func (c *Client) Health(ctx context.Context) (*healthResponse, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/health", nil)
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, wrapNetErr(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status %d", resp.StatusCode)
	}

    var env healthEnvelope
    if err := json.NewDecoder(resp.Body).Decode(&env); err != nil {
        return nil, fmt.Errorf("decode response: %w", err)
    }
    if env.Status != "ok" {
        if env.Error != nil {
            return nil, fmt.Errorf("langchain service error (%d): %s", env.Error.Code, env.Error.Message)
        }
        return nil, fmt.Errorf("langchain service returned status %s", env.Status)
    }
    return &healthResponse{
        Status:    env.Data.ServiceStatus,
        Documents: env.Data.Documents,
        Meta:      env.Meta,
    }, nil
}

// ChainRequest represents the body sent to /chains/run.
type ChainRequest struct {
	Prompt string `json:"prompt"`
}

// ChainResponse captures the service output.
type ChainResponse struct {
    Result  string
    Sources []Source
    Meta    map[string]any
}

type Source struct {
    Source string `json:"source"`
    DocID  string `json:"doc_id"`
}

type chainEnvelope struct {
    Status string `json:"status"`
    Data   struct {
        Result  string   `json:"result"`
        Sources []Source `json:"sources"`
    } `json:"data"`
    Meta  map[string]any `json:"meta"`
    Error *apiError       `json:"error,omitempty"`
}

// RunChain executes the retrieval chain exposed by the LangChain service.
func (c *Client) RunChain(ctx context.Context, prompt string) (*ChainResponse, error) {
	if strings.TrimSpace(prompt) == "" {
		return nil, errors.New("prompt must not be empty")
	}

	payload, err := json.Marshal(ChainRequest{Prompt: prompt})
	if err != nil {
		return nil, fmt.Errorf("encode request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/chains/run", bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, wrapNetErr(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status %d", resp.StatusCode)
	}

    var env chainEnvelope
    if err := json.NewDecoder(resp.Body).Decode(&env); err != nil {
        return nil, fmt.Errorf("decode response: %w", err)
    }
    if env.Status != "ok" {
        if env.Error != nil {
            return nil, fmt.Errorf("langchain service error (%d): %s", env.Error.Code, env.Error.Message)
        }
        return nil, fmt.Errorf("langchain service returned status %s", env.Status)
    }
    return &ChainResponse{
        Result:  env.Data.Result,
        Sources: env.Data.Sources,
        Meta:    env.Meta,
    }, nil
}

func wrapNetErr(err error) error {
	var netErr net.Error
	if errors.As(err, &netErr) {
		return fmt.Errorf("langchain service network error: %w", netErr)
	}
	return err
}
