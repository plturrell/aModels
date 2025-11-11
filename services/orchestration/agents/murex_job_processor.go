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

// MurexJobProcessor handles async processing of Murex trade requests.
type MurexJobProcessor struct {
	tracker      *RequestTracker
	pipeline     *MurexPipeline
	jobs         map[string]*Job
	mu           sync.RWMutex
	logger       *log.Logger
	httpClient   *http.Client
	webhookQueue chan WebhookNotification
}

// NewMurexJobProcessor creates a new Murex job processor.
func NewMurexJobProcessor(tracker *RequestTracker, pipeline *MurexPipeline, logger *log.Logger) *MurexJobProcessor {
	processor := &MurexJobProcessor{
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
func (jp *MurexJobProcessor) SubmitJob(requestID string, query map[string]interface{}, webhookURL string) error {
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
func (jp *MurexJobProcessor) processJob(job *Job) {
	jp.mu.Lock()
	job.Status = JobStatusRunning
	jp.mu.Unlock()

	// Update tracker status
	jp.tracker.UpdateStatus(job.RequestID, RequestStatusProcessing)

	// Process trades
	_, err := jp.pipeline.ProcessTradesWithTracking(job.Context, job.RequestID, job.Query)

	jp.mu.Lock()
	if err != nil {
		job.Status = JobStatusFailed
		job.Error = err.Error()
		jp.tracker.UpdateStatus(job.RequestID, RequestStatusFailed)
	} else {
		job.Status = JobStatusCompleted
		job.CompletedAt = time.Now()
		jp.tracker.UpdateStatus(job.RequestID, RequestStatusCompleted)
	}
	jp.mu.Unlock()

	// Send webhook notification if configured
	if job.WebhookURL != "" {
		jp.sendWebhookNotification(job)
	}
}

// CancelJob cancels a running job.
func (jp *MurexJobProcessor) CancelJob(requestID string) error {
	jp.mu.Lock()
	defer jp.mu.Unlock()

	job, ok := jp.jobs[requestID]
	if !ok {
		return fmt.Errorf("job not found: %s", requestID)
	}

	if job.Status == JobStatusCompleted || job.Status == JobStatusFailed {
		return fmt.Errorf("job already %s", job.Status)
	}

	// Cancel context
	if job.CancelFunc != nil {
		job.CancelFunc()
	}

	job.Status = JobStatusCancelled
	job.CompletedAt = time.Now()
	jp.tracker.UpdateStatus(requestID, RequestStatusCancelled)

	return nil
}

// GetJob returns job information.
func (jp *MurexJobProcessor) GetJob(requestID string) (*Job, error) {
	jp.mu.RLock()
	defer jp.mu.RUnlock()

	job, ok := jp.jobs[requestID]
	if !ok {
		return nil, fmt.Errorf("job not found: %s", requestID)
	}

	return job, nil
}

// sendWebhookNotification sends a webhook notification.
func (jp *MurexJobProcessor) sendWebhookNotification(job *Job) {
	request := jp.tracker.GetRequest(job.RequestID)
	if request == nil {
		return
	}

	notification := WebhookNotification{
		RequestID: job.RequestID,
		Status:    string(job.Status),
		Timestamp: time.Now(),
		Statistics: request.Statistics,
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
func (jp *MurexJobProcessor) processWebhooks() {
	for notification := range jp.webhookQueue {
		// Get job to find webhook URL
		jp.mu.RLock()
		job, ok := jp.jobs[notification.RequestID]
		jp.mu.RUnlock()

		if !ok || job.WebhookURL == "" {
			continue
		}

		// Send webhook
		payload, _ := json.Marshal(notification)
		req, err := http.NewRequest(http.MethodPost, job.WebhookURL, strings.NewReader(string(payload)))
		if err != nil {
			if jp.logger != nil {
				jp.logger.Printf("Failed to create webhook request: %v", err)
			}
			continue
		}

		req.Header.Set("Content-Type", "application/json")

		resp, err := jp.httpClient.Do(req)
		if err != nil {
			if jp.logger != nil {
				jp.logger.Printf("Failed to send webhook: %v", err)
			}
			continue
		}
		resp.Body.Close()

		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			if jp.logger != nil {
				jp.logger.Printf("Webhook sent successfully to %s", job.WebhookURL)
			}
		} else {
			if jp.logger != nil {
				jp.logger.Printf("Webhook returned status %d", resp.StatusCode)
			}
		}
	}
}

// Start starts the job processor (for compatibility with other processors).
func (jp *MurexJobProcessor) Start() {
	// Webhook processor is already running
	if jp.logger != nil {
		jp.logger.Printf("Murex job processor started")
	}
}

