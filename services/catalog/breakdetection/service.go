package breakdetection

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"github.com/google/uuid"
)

// BreakDetectionService is the main service for break detection
type BreakDetectionService struct {
	db              *sql.DB
	baselineManager *BaselineManager
	financeDetector  *FinanceDetector
	capitalDetector  *CapitalDetector
	liquidityDetector *LiquidityDetector
	regulatoryDetector *RegulatoryDetector
	analysisService  *BreakAnalysisService
	enrichmentService *EnrichmentService
	ruleGenerator   *RuleGeneratorService
	searchService   *BreakSearchService
	aiAnalysisService *AIAnalysisService
	logger          *log.Logger
}

// NewBreakDetectionService creates a new break detection service
func NewBreakDetectionService(
	db *sql.DB,
	baselineManager *BaselineManager,
	financeDetector *FinanceDetector,
	capitalDetector *CapitalDetector,
	liquidityDetector *LiquidityDetector,
	regulatoryDetector *RegulatoryDetector,
	analysisService *BreakAnalysisService,
	enrichmentService *EnrichmentService,
	ruleGenerator *RuleGeneratorService,
	searchService *BreakSearchService,
	aiAnalysisService *AIAnalysisService,
	logger *log.Logger,
) *BreakDetectionService {
	return &BreakDetectionService{
		db:               db,
		baselineManager:  baselineManager,
		financeDetector:  financeDetector,
		capitalDetector:  capitalDetector,
		liquidityDetector: liquidityDetector,
		regulatoryDetector: regulatoryDetector,
		analysisService:  analysisService,
		enrichmentService: enrichmentService,
		ruleGenerator:    ruleGenerator,
		searchService:   searchService,
		aiAnalysisService: aiAnalysisService,
		logger:           logger,
	}
}

// DetectBreaks performs break detection for a system
func (s *BreakDetectionService) DetectBreaks(ctx context.Context, req *DetectionRequest) (*DetectionResult, error) {
	runID := fmt.Sprintf("run-%s-%s-%d", req.SystemName, req.DetectionType, time.Now().Unix())
	
	// Create detection run record
	run, err := s.createDetectionRun(ctx, req, runID)
	if err != nil {
		return nil, fmt.Errorf("failed to create detection run: %w", err)
	}
	
	// Get baseline
	baseline, err := s.baselineManager.GetBaseline(ctx, req.BaselineID)
	if err != nil {
		return s.failDetectionRun(ctx, runID, fmt.Errorf("failed to get baseline: %w", err))
	}
	
	// Perform detection based on type
	var breaks []*Break
	switch req.DetectionType {
	case DetectionTypeFinance:
		if s.financeDetector == nil {
			return s.failDetectionRun(ctx, runID, fmt.Errorf("finance detector not initialized"))
		}
		breaks, err = s.financeDetector.DetectBreaks(ctx, baseline, req.Configuration)
		if err != nil {
			return s.failDetectionRun(ctx, runID, fmt.Errorf("finance break detection failed: %w", err))
		}
	case DetectionTypeCapital:
		if s.capitalDetector == nil {
			return s.failDetectionRun(ctx, runID, fmt.Errorf("capital detector not initialized"))
		}
		breaks, err = s.capitalDetector.DetectBreaks(ctx, baseline, req.Configuration)
		if err != nil {
			return s.failDetectionRun(ctx, runID, fmt.Errorf("capital break detection failed: %w", err))
		}
	case DetectionTypeLiquidity:
		if s.liquidityDetector == nil {
			return s.failDetectionRun(ctx, runID, fmt.Errorf("liquidity detector not initialized"))
		}
		breaks, err = s.liquidityDetector.DetectBreaks(ctx, baseline, req.Configuration)
		if err != nil {
			return s.failDetectionRun(ctx, runID, fmt.Errorf("liquidity break detection failed: %w", err))
		}
	case DetectionTypeRegulatory:
		if s.regulatoryDetector == nil {
			return s.failDetectionRun(ctx, runID, fmt.Errorf("regulatory detector not initialized"))
		}
		breaks, err = s.regulatoryDetector.DetectBreaks(ctx, baseline, req.Configuration)
		if err != nil {
			return s.failDetectionRun(ctx, runID, fmt.Errorf("regulatory break detection failed: %w", err))
		}
	default:
		return s.failDetectionRun(ctx, runID, fmt.Errorf("unknown detection type: %s", req.DetectionType))
	}
	
	// Store breaks and enrich with Deep Research
	for _, b := range breaks {
		b.RunID = runID
		
		// Enrich break with Deep Research (if services available)
		if s.enrichmentService != nil {
			enrichment, err := s.enrichmentService.EnrichBreakContext(ctx, b)
			if err != nil {
				if s.logger != nil {
					s.logger.Printf("Warning: Failed to enrich break %s: %v", b.BreakID, err)
				}
			} else {
				b.SemanticEnrichment = enrichment
			}
		}
		
		// Perform root cause analysis (if service available)
		if s.analysisService != nil {
			rootCause, err := s.analysisService.AnalyzeRootCause(ctx, b)
			if err != nil {
				if s.logger != nil {
					s.logger.Printf("Warning: Failed to analyze root cause for break %s: %v", b.BreakID, err)
				}
			} else {
				b.RootCauseAnalysis = rootCause
				
				// Generate recommendations based on root cause
				recommendations, err := s.analysisService.GenerateRecommendations(ctx, rootCause, b)
				if err == nil && len(recommendations) > 0 {
					b.Recommendations = recommendations
				}
			}
		}
		
		// Search for similar breaks (if service available)
		if s.searchService != nil {
			similarBreaks, err := s.searchService.SearchSimilarBreaks(ctx, b, 5, 0.7)
			if err != nil {
				if s.logger != nil {
					s.logger.Printf("Warning: Failed to search for similar breaks for %s: %v", b.BreakID, err)
				}
			} else {
				b.SimilarBreaks = similarBreaks
			}
			
			// Index break for future searches
			if err := s.searchService.IndexBreak(ctx, b); err != nil {
				if s.logger != nil {
					s.logger.Printf("Warning: Failed to index break %s: %v", b.BreakID, err)
				}
			}
		}
		
		// AI analysis (if service available)
		if s.aiAnalysisService != nil {
			// Generate AI description
			aiDescription, err := s.aiAnalysisService.GenerateBreakDescription(ctx, b)
			if err == nil && aiDescription != "" {
				b.AIDescription = aiDescription
			}
			
			// Categorize break
			aiCategory, err := s.aiAnalysisService.CategorizeBreak(ctx, b)
			if err == nil && aiCategory != "" {
				b.AICategory = aiCategory
			}
			
			// Calculate priority score
			priorityScore, err := s.aiAnalysisService.CalculatePriorityScore(ctx, b)
			if err == nil {
				b.AIPriorityScore = priorityScore
			}
		}
		
		if err := s.storeBreak(ctx, b); err != nil {
			if s.logger != nil {
				s.logger.Printf("Warning: Failed to store break %s: %v", b.BreakID, err)
			}
		}
	}
	
	// Update run with results
	result := &DetectionResult{
		RunID:               runID,
		Status:              RunStatusCompleted,
		TotalBreaksDetected: len(breaks),
		TotalRecordsChecked: s.calculateRecordsChecked(breaks),
		Breaks:              breaks,
		ResultSummary: map[string]interface{}{
			"severity_counts": s.countBreaksBySeverity(breaks),
			"break_type_counts": s.countBreaksByType(breaks),
		},
	}
	
	if err := s.completeDetectionRun(ctx, runID, result); err != nil {
		if s.logger != nil {
			s.logger.Printf("Warning: Failed to update detection run: %v", err)
		}
	}
	
	if s.logger != nil {
		s.logger.Printf("Break detection completed: run_id=%s, breaks=%d", runID, len(breaks))
	}
	
	return result, nil
}

// createDetectionRun creates a new detection run record
func (s *BreakDetectionService) createDetectionRun(ctx context.Context, req *DetectionRequest, runID string) (*DetectionRun, error) {
	id := uuid.New().String()
	
	configJSON := []byte("{}")
	if req.Configuration != nil {
		var err error
		configJSON, err = json.Marshal(req.Configuration)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal configuration: %w", err)
		}
	}
	
	query := `
		INSERT INTO break_detection_runs (
			id, run_id, system_name, baseline_id, detection_type,
			status, configuration, created_by, workflow_instance_id
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
		RETURNING started_at
	`
	
	var startedAt time.Time
	err := s.db.QueryRowContext(ctx, query,
		id, runID, string(req.SystemName), req.BaselineID, string(req.DetectionType),
		string(RunStatusRunning), configJSON, req.CreatedBy, req.WorkflowInstanceID,
	).Scan(&startedAt)
	
	if err != nil {
		return nil, fmt.Errorf("failed to create detection run: %w", err)
	}
	
	return &DetectionRun{
		ID:                 id,
		RunID:              runID,
		SystemName:         req.SystemName,
		BaselineID:         req.BaselineID,
		DetectionType:      req.DetectionType,
		Status:             RunStatusRunning,
		StartedAt:          startedAt,
		Configuration:      req.Configuration,
		CreatedBy:          req.CreatedBy,
		WorkflowInstanceID: req.WorkflowInstanceID,
	}, nil
}

// failDetectionRun marks a detection run as failed
func (s *BreakDetectionService) failDetectionRun(ctx context.Context, runID string, err error) (*DetectionResult, error) {
	query := `
		UPDATE break_detection_runs
		SET status = $1, completed_at = NOW(), error_message = $2
		WHERE run_id = $3
	`
	
	_, updateErr := s.db.ExecContext(ctx, query, string(RunStatusFailed), err.Error(), runID)
	if updateErr != nil && s.logger != nil {
		s.logger.Printf("Warning: Failed to update failed detection run: %v", updateErr)
	}
	
	return nil, err
}

// completeDetectionRun marks a detection run as completed
func (s *BreakDetectionService) completeDetectionRun(ctx context.Context, runID string, result *DetectionResult) error {
	resultSummaryJSON, err := json.Marshal(result.ResultSummary)
	if err != nil {
		return fmt.Errorf("failed to marshal result summary: %w", err)
	}
	
	query := `
		UPDATE break_detection_runs
		SET status = $1, completed_at = NOW(),
		    total_breaks_detected = $2, total_records_checked = $3,
		    result_summary = $4
		WHERE run_id = $5
	`
	
	_, err = s.db.ExecContext(ctx, query,
		string(result.Status), result.TotalBreaksDetected, result.TotalRecordsChecked,
		resultSummaryJSON, runID,
	)
	
	return err
}

// storeBreak stores a detected break in the database
func (s *BreakDetectionService) storeBreak(ctx context.Context, b *Break) error {
	// Generate break ID if not set
	if b.BreakID == "" {
		b.BreakID = fmt.Sprintf("break-%s-%d", b.RunID, time.Now().UnixNano())
	}
	
	id := uuid.New().String()
	
	currentJSON, _ := json.Marshal(b.CurrentValue)
	baselineJSON, _ := json.Marshal(b.BaselineValue)
	differenceJSON, _ := json.Marshal(b.Difference)
	affectedJSON, _ := json.Marshal(b.AffectedEntities)
	semanticJSON, _ := json.Marshal(b.SemanticEnrichment)
	similarJSON, _ := json.Marshal(b.SimilarBreaks)
	
	query := `
		INSERT INTO break_detection_breaks (
			id, break_id, run_id, system_name, detection_type,
			break_type, severity, status,
			current_value, baseline_value, difference, affected_entities,
			root_cause_analysis, semantic_enrichment, recommendations,
			ai_description, ai_category, ai_priority_score,
			similar_breaks
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
	`
	
	_, err := s.db.ExecContext(ctx, query,
		id, b.BreakID, b.RunID, string(b.SystemName), string(b.DetectionType),
		string(b.BreakType), string(b.Severity), string(b.Status),
		currentJSON, baselineJSON, differenceJSON, affectedJSON,
		b.RootCauseAnalysis, semanticJSON, b.Recommendations,
		b.AIDescription, b.AICategory, b.AIPriorityScore,
		similarJSON,
	)
	
	return err
}

// GetDetectionRun retrieves a detection run by ID
func (s *BreakDetectionService) GetDetectionRun(ctx context.Context, runID string) (*DetectionRun, error) {
	// Implementation to retrieve run and associated breaks
	// This would query the database and populate the DetectionRun struct
	return nil, fmt.Errorf("not yet implemented")
}

// ListBreaks lists breaks for a system
func (s *BreakDetectionService) ListBreaks(ctx context.Context, systemName SystemName, limit int, status BreakStatus) ([]*Break, error) {
	query := `
		SELECT break_id, run_id, system_name, detection_type, break_type,
		       severity, status, current_value, baseline_value, difference,
		       affected_entities, root_cause_analysis, semantic_enrichment,
		       recommendations, ai_description, ai_category, ai_priority_score,
		       similar_breaks, detected_at, resolved_at, resolved_by
		FROM break_detection_breaks
		WHERE system_name = $1 AND status = $2
		ORDER BY detected_at DESC
		LIMIT $3
	`
	
	// Implementation would query and parse results
	return nil, fmt.Errorf("not yet implemented")
}

// Helper functions
func (s *BreakDetectionService) calculateRecordsChecked(breaks []*Break) int {
	// Count unique affected entities
	entities := make(map[string]bool)
	for _, b := range breaks {
		for _, entity := range b.AffectedEntities {
			entities[entity] = true
		}
	}
	return len(entities)
}

func (s *BreakDetectionService) countBreaksBySeverity(breaks []*Break) map[string]int {
	counts := make(map[string]int)
	for _, b := range breaks {
		counts[string(b.Severity)]++
	}
	return counts
}

func (s *BreakDetectionService) countBreaksByType(breaks []*Break) map[string]int {
	counts := make(map[string]int)
	for _, b := range breaks {
		counts[string(b.BreakType)]++
	}
	return counts
}

