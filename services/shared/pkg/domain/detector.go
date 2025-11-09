package domain

import (
    "context"
    "encoding/json"
    "errors"
    "fmt"
    "io"
    "log"
    "net/http"
    "strings"
    "sync"
    "time"
)

const (
    domainsEndpoint     = "/v1/domains"
    defaultHTTPTimeout  = 15 * time.Second
    llmRequestTimeout   = 10 * time.Second // exported for consumers via helper
)

// ErrNoDomains is returned when no domains are available after a sync.
var ErrNoDomains = errors.New("domain: no domain configurations available")

// DomainConfig represents a domain configuration returned by LocalAI.
type DomainConfig struct {
    Name     string   `json:"name"`
    AgentID  string   `json:"agent_id"`
    Keywords []string `json:"keywords"`
    Tags     []string `json:"tags"`
    Layer    string   `json:"layer"`
    Team     string   `json:"team"`
}

// Detector loads domain configurations from LocalAI and performs keyword-based detection.
type Detector struct {
    localaiURL    string
    logger        *log.Logger
    httpClient    *http.Client
    mu            sync.RWMutex
    domainConfigs map[string]DomainConfig
    lastSync      time.Time
}

// NewDetector creates a detector. Detection is disabled when localaiURL is empty.
func NewDetector(localaiURL string, logger *log.Logger) *Detector {
    localaiURL = strings.TrimSpace(localaiURL)
    if logger == nil {
        logger = log.New(io.Discard, "", log.LstdFlags)
    }

    detector := &Detector{
        localaiURL:    strings.TrimSuffix(localaiURL, "/"),
        logger:        logger,
        httpClient:    &http.Client{Timeout: defaultHTTPTimeout},
        domainConfigs: make(map[string]DomainConfig),
    }

    if localaiURL == "" {
        detector.logger.Println("domain.Detector: LOCALAI_URL not provided; domain detection disabled")
    }

    return detector
}

// SetHTTPClient allows consumers to override the HTTP client (useful for tests).
func (d *Detector) SetHTTPClient(client *http.Client) {
    d.mu.Lock()
    defer d.mu.Unlock()
    if client == nil {
        d.httpClient = &http.Client{Timeout: defaultHTTPTimeout}
        return
    }
    d.httpClient = client
}

// LastSyncAt returns the last successful sync time.
func (d *Detector) LastSyncAt() time.Time {
    d.mu.RLock()
    defer d.mu.RUnlock()
    return d.lastSync
}

// DomainCount returns the number of cached domains.
func (d *Detector) DomainCount() int {
    d.mu.RLock()
    defer d.mu.RUnlock()
    return len(d.domainConfigs)
}

// Domains returns a copy of the cached domain configurations.
func (d *Detector) Domains() map[string]DomainConfig {
    d.mu.RLock()
    defer d.mu.RUnlock()

    copy := make(map[string]DomainConfig, len(d.domainConfigs))
    for k, v := range d.domainConfigs {
        copy[k] = v
    }
    return copy
}

// Config returns a domain configuration by ID when available.
func (d *Detector) Config(domainID string) (DomainConfig, bool) {
    d.mu.RLock()
    defer d.mu.RUnlock()
    cfg, ok := d.domainConfigs[domainID]
    return cfg, ok
}

// LoadDomains fetches domain configurations from LocalAI.
func (d *Detector) LoadDomains(ctx context.Context) error {
    if d.localaiURL == "" {
        return nil
    }

    ctx, cancel := context.WithTimeout(ctx, defaultHTTPTimeout)
    defer cancel()

    req, err := http.NewRequestWithContext(ctx, http.MethodGet, d.localaiURL+domainsEndpoint, nil)
    if err != nil {
        return fmt.Errorf("domain: create request: %w", err)
    }

    d.mu.RLock()
    client := d.httpClient
    d.mu.RUnlock()

    resp, err := client.Do(req)
    if err != nil {
        return fmt.Errorf("domain: fetch domains: %w", err)
    }
    defer resp.Body.Close()

    if resp.StatusCode != http.StatusOK {
        body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
        return fmt.Errorf("domain: fetch domains unexpected status %d: %s", resp.StatusCode, string(body))
    }

    payload, err := io.ReadAll(resp.Body)
    if err != nil {
        return fmt.Errorf("domain: read response: %w", err)
    }

    configs, err := parseDomainPayload(payload)
    if err != nil {
        return err
    }

    if len(configs) == 0 {
        return ErrNoDomains
    }

    d.mu.Lock()
    d.domainConfigs = configs
    d.lastSync = time.Now()
    d.mu.Unlock()

    d.logger.Printf("domain.Detector: loaded %d domain configurations", len(configs))
    return nil
}

// DetectDomain returns the best matching domain and agent ID for the provided text.
func (d *Detector) DetectDomain(text string) (domainID string, agentID string) {
    d.mu.RLock()
    configs := d.domainConfigs
    d.mu.RUnlock()

    if len(configs) == 0 || strings.TrimSpace(text) == "" {
        return "", ""
    }

    textLower := strings.ToLower(text)
    bestScore := 0
    for id, cfg := range configs {
        if cfg.AgentID == "" {
            continue
        }

        score := 0
        for _, keyword := range cfg.Keywords {
            keyword = strings.TrimSpace(keyword)
            if keyword == "" {
                continue
            }
            if strings.Contains(textLower, strings.ToLower(keyword)) {
                score += 2 // keywords have higher weight
            }
        }

        for _, tag := range cfg.Tags {
            tag = strings.TrimSpace(tag)
            if tag == "" {
                continue
            }
            if strings.Contains(textLower, strings.ToLower(tag)) {
                score++
            }
        }

        if score > bestScore {
            bestScore = score
            domainID = id
            agentID = cfg.AgentID
        }
    }

    return domainID, agentID
}

// parseDomainPayload handles multiple legacy response envelopes used by LocalAI.
func parseDomainPayload(payload []byte) (map[string]DomainConfig, error) {
    type domainRecord struct {
        ID     string       `json:"id"`
        Config DomainConfig `json:"config"`
    }

    var envelope map[string]json.RawMessage
    if err := json.Unmarshal(payload, &envelope); err != nil {
        return nil, fmt.Errorf("domain: decode envelope: %w", err)
    }

    configs := make(map[string]DomainConfig)

    switch {
    case envelope == nil:
        return configs, nil
    case envelope["data"] != nil:
        var data []domainRecord
        if err := json.Unmarshal(envelope["data"], &data); err != nil {
            return nil, fmt.Errorf("domain: decode data records: %w", err)
        }
        for _, item := range data {
            if item.ID == "" {
                continue
            }
            configs[item.ID] = item.Config
        }
    case envelope["domains"] != nil:
        var domainMap map[string]struct {
            Config DomainConfig `json:"config"`
        }
        if err := json.Unmarshal(envelope["domains"], &domainMap); err != nil {
            return nil, fmt.Errorf("domain: decode map records: %w", err)
        }
        for id, item := range domainMap {
            if id == "" {
                continue
            }
            configs[id] = item.Config
        }
    default:
        return nil, fmt.Errorf("domain: unrecognised domain payload structure")
    }

    return configs, nil
}

// LLMRequestTimeout exposes the recommended timeout for downstream LocalAI calls.
func LLMRequestTimeout() time.Duration {
    return llmRequestTimeout
}
