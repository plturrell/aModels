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

// DMSJobProcessor handles async processing of DMS requests.
type DMSJobProcessor struct {
	tracker      *RequestTracker
	pipeline     *DMSPipeline
	jobs         map[string]*Job
	mu           sync.RWMutex
	logger       *log.Logger
	httpClient   *http.Client
	webhookQueue chan WebhookNotification
}

// NewDMSJobProcessor creates a new DMS job processor.
func NewDMSJobProcessor(tracker *RequestTracker, pipeline *DMSPipeline, logger *log.Logger) *DMSJobProcessor {
	processor := &DMSJobProcessor{
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
func (jp *DMSJobProcessor) SubmitJob(requestID string, query map[string]interface{}, webhookURL string) error {
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
func (jp *DMSJobProcessor) processJob(job *Job) {
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
func (jp *DMSJobProcessor) CancelJob(requestID string) error {
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
func (jp *DMSJobProcessor) GetJob(requestID string) (*Job, bool) {
	jp.mu.RLock()
	defer jp.mu.RUnlock()

	job, exists := jp.jobs[requestID]
	return job, exists
}

// sendWebhookNotification sends a webhook notification.
func (jp *DMSJobProcessor) sendWebhookNotification(job *Job) {
	request, exists := jp.tracker.GetRequest(job.RequestID)
	if !exists {
		return
	}

	payload := map[string]interface{}{
		"request_id": job.RequestID,
		"status":     string(job.Status),
		"timestamp":  time.Now().UTC().Format(time.RFC3339),
	}

	if request.Statistics != nil {
		payload["statistics"] = request.Statistics
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
			jp.logger.Printf("Webhook queue full, dropping notification for job %s", job.RequestID)
		}
	}
}

// processWebhooks processes webhook notifications.
func (jp *DMSJobProcessor) processWebhooks() {
	for notification := range jp.webhookQueue {
		jp.sendWebhook(notification)
	}
}

// sendWebhook sends a webhook HTTP request.
func (jp *DMSJobProcessor) sendWebhook(notification WebhookNotification) {
	payload, err := json.Marshal(notification.Payload)
	if err != nil {
		if jp.logger != nil {
			jp.logger.Printf("Failed to marshal webhook payload: %v", err)
		}
		return
	}

	req, err := http.NewRequest(http.MethodPost, notification.URL, strings.NewReader(string(payload)))
	if err != nil {
		if jp.logger != nil {
			jp.logger.Printf("Failed to create webhook request: %v", err)
		}
		return
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := jp.httpClient.Do(req)
	if err != nil {
		if jp.logger != nil {
			jp.logger.Printf("Failed to send webhook: %v", err)
		}
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		if jp.logger != nil {
			jp.logger.Printf("Webhook returned error status %d", resp.StatusCode)
		}
	}
}

// Start starts the job processor.
func (jp *DMSJobProcessor) Start() {
	// Already started in constructor
}

