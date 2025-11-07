package agents

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"
)

// JobProcessor handles async processing of Perplexity requests.
type JobProcessor struct {
	tracker      *RequestTracker
	pipeline     *PerplexityPipeline
	jobs         map[string]*Job
	mu           sync.RWMutex
	logger       *log.Logger
	httpClient   *http.Client
	webhookQueue chan WebhookNotification
}

// Job represents an async processing job.
type Job struct {
	RequestID   string
	Query       map[string]interface{}
	Status      JobStatus
	Context     context.Context
	CancelFunc  context.CancelFunc
	StartedAt   time.Time
	CompletedAt *time.Time
	Error       error
	WebhookURL  string
}

// JobStatus represents the status of a job.
type JobStatus string

const (
	JobStatusQueued    JobStatus = "queued"
	JobStatusRunning   JobStatus = "running"
	JobStatusCompleted JobStatus = "completed"
	JobStatusFailed     JobStatus = "failed"
	JobStatusCancelled JobStatus = "cancelled"
)

// WebhookNotification represents a webhook notification.
type WebhookNotification struct {
	RequestID string
	Status    string
	URL       string
	Payload   map[string]interface{}
}

// NewJobProcessor creates a new job processor.
func NewJobProcessor(tracker *RequestTracker, pipeline *PerplexityPipeline, logger *log.Logger) *JobProcessor {
	processor := &JobProcessor{
		tracker:      tracker,
		pipeline:     pipeline,
		jobs:         make(map[string]*Job),
		logger:       logger,
		httpClient:   &http.Client{Timeout: 30 * time.Second},
		webhookQueue: make(chan WebhookNotification, 100),
	}

	// Start webhook processor
	go processor.processWebhooks()

	return processor
}

// SubmitJob submits a job for async processing.
func (jp *JobProcessor) SubmitJob(requestID string, query map[string]interface{}, webhookURL string) error {
	jp.mu.Lock()
	defer jp.mu.Unlock()

	// Create context with cancellation
	ctx, cancel := context.WithCancel(context.Background())

	job := &Job{
		RequestID:  requestID,
		Query:      query,
		Status:     JobStatusQueued,
		Context:    ctx,
		CancelFunc: cancel,
		StartedAt:  time.Now(),
		WebhookURL: webhookURL,
	}

	jp.jobs[requestID] = job

	// Start processing in background
	go jp.processJob(job)

	return nil
}

// processJob processes a job in the background.
func (jp *JobProcessor) processJob(job *Job) {
	jp.mu.Lock()
	job.Status = JobStatusRunning
	jp.mu.Unlock()

	// Update tracker status
	jp.tracker.UpdateStatus(job.RequestID, RequestStatusProcessing)

	// Process documents
	_, err := jp.pipeline.ProcessDocumentsWithTracking(job.Context, job.RequestID, job.Query)

	jp.mu.Lock()
	now := time.Now()
	job.CompletedAt = &now
	job.Error = err

	if err != nil {
		job.Status = JobStatusFailed
		jp.tracker.UpdateStatus(job.RequestID, RequestStatusFailed)
	} else {
		job.Status = JobStatusCompleted
		// Check if partial success
		if request, exists := jp.tracker.GetRequest(job.RequestID); exists {
			if request.Statistics != nil && request.Statistics.DocumentsFailed > 0 {
				jp.tracker.UpdateStatus(job.RequestID, RequestStatusPartial)
			} else {
				jp.tracker.UpdateStatus(job.RequestID, RequestStatusCompleted)
			}
		}
	}
	jp.mu.Unlock()

	// Send webhook notification if configured
	if job.WebhookURL != "" {
		jp.sendWebhookNotification(job)
	}

	if jp.logger != nil {
		if err != nil {
			jp.logger.Printf("Job %s completed with error: %v", job.RequestID, err)
		} else {
			jp.logger.Printf("Job %s completed successfully", job.RequestID)
		}
	}
}

// CancelJob cancels a running job.
func (jp *JobProcessor) CancelJob(requestID string) error {
	jp.mu.Lock()
	defer jp.mu.Unlock()

	job, exists := jp.jobs[requestID]
	if !exists {
		return fmt.Errorf("job %s not found", requestID)
	}

	if job.Status != JobStatusRunning && job.Status != JobStatusQueued {
		return fmt.Errorf("job %s cannot be cancelled (status: %s)", requestID, job.Status)
	}

	// Cancel context
	job.CancelFunc()

	// Update status
	job.Status = JobStatusCancelled
	jp.tracker.UpdateStatus(requestID, RequestStatusFailed)

	if jp.logger != nil {
		jp.logger.Printf("Job %s cancelled", requestID)
	}

	return nil
}

// GetJob retrieves a job by ID.
func (jp *JobProcessor) GetJob(requestID string) (*Job, bool) {
	jp.mu.RLock()
	defer jp.mu.RUnlock()

	job, exists := jp.jobs[requestID]
	return job, exists
}

// sendWebhookNotification sends a webhook notification.
func (jp *JobProcessor) sendWebhookNotification(job *Job) {
	request, exists := jp.tracker.GetRequest(job.RequestID)
	if !exists {
		return
	}

	payload := map[string]interface{}{
		"request_id": job.RequestID,
		"status":     string(job.Status),
		"query":      job.Query["query"],
		"timestamp":  time.Now().UTC().Format(time.RFC3339),
	}

	if request.Statistics != nil {
		payload["statistics"] = request.Statistics
	}

	if job.Error != nil {
		payload["error"] = job.Error.Error()
	}

	notification := WebhookNotification{
		RequestID: job.RequestID,
		Status:    string(job.Status),
		URL:       job.WebhookURL,
		Payload:   payload,
	}

	select {
	case jp.webhookQueue <- notification:
	default:
		if jp.logger != nil {
			jp.logger.Printf("Webhook queue full, dropping notification for %s", job.RequestID)
		}
	}
}

// processWebhooks processes webhook notifications.
func (jp *JobProcessor) processWebhooks() {
	for notification := range jp.webhookQueue {
		jp.sendWebhook(notification)
	}
}

// sendWebhook sends a webhook HTTP request.
func (jp *JobProcessor) sendWebhook(notification WebhookNotification) {
	payloadJSON, err := json.Marshal(notification.Payload)
	if err != nil {
		if jp.logger != nil {
			jp.logger.Printf("Failed to marshal webhook payload: %v", err)
		}
		return
	}

	req, err := http.NewRequest("POST", notification.URL, strings.NewReader(string(payloadJSON)))
	if err != nil {
		if jp.logger != nil {
			jp.logger.Printf("Failed to create webhook request: %v", err)
		}
		return
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "Perplexity-Integration/1.0")

	resp, err := jp.httpClient.Do(req)
	if err != nil {
		if jp.logger != nil {
			jp.logger.Printf("Failed to send webhook to %s: %v", notification.URL, err)
		}
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		if jp.logger != nil {
			jp.logger.Printf("Webhook sent successfully to %s", notification.URL)
		}
	} else {
		if jp.logger != nil {
			jp.logger.Printf("Webhook returned status %d for %s", resp.StatusCode, notification.URL)
		}
	}
}

