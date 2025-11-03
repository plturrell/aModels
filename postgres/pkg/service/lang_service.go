package service

import (
	"context"
	"fmt"

	"google.golang.org/protobuf/types/known/structpb"
	"google.golang.org/protobuf/types/known/timestamppb"

	postgresv1 "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Postgres/pkg/gen/v1"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Postgres/pkg/models"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Postgres/pkg/repository"
)

// LangService implements the gRPC server wiring repository operations.
type LangService struct {
	postgresv1.UnimplementedPostgresLangServiceServer

	repo    *repository.OperationsRepository
	version string
}

// NewLangService constructs a LangService instance.
func NewLangService(repo *repository.OperationsRepository, version string) *LangService {
	return &LangService{repo: repo, version: version}
}

// HealthCheck validates database connectivity and reports service metadata.
func (s *LangService) HealthCheck(ctx context.Context, _ *postgresv1.HealthCheckRequest) (*postgresv1.HealthCheckResponse, error) {
	if err := s.repo.Ping(ctx); err != nil {
		return nil, fmt.Errorf("database not reachable: %w", err)
	}

	return &postgresv1.HealthCheckResponse{
		Status:  "SERVING",
		Version: s.version,
	}, nil
}

// LogLangOperation persists an incoming operation record.
func (s *LangService) LogLangOperation(ctx context.Context, req *postgresv1.LogLangOperationRequest) (*postgresv1.LogLangOperationResponse, error) {
	if req == nil || req.Operation == nil {
		return nil, fmt.Errorf("operation payload is required")
	}

	op, err := protoToModel(req.Operation)
	if err != nil {
		return nil, fmt.Errorf("convert operation: %w", err)
	}

	if err := s.repo.LogOperation(ctx, op); err != nil {
		return nil, err
	}

	protoOp, err := modelToProto(op)
	if err != nil {
		return nil, err
	}

	return &postgresv1.LogLangOperationResponse{Operation: protoOp}, nil
}

// GetLangOperation fetches a previously stored operation.
func (s *LangService) GetLangOperation(ctx context.Context, req *postgresv1.GetLangOperationRequest) (*postgresv1.GetLangOperationResponse, error) {
	if req == nil || req.Id == "" {
		return nil, fmt.Errorf("id is required")
	}

	op, err := s.repo.GetOperation(ctx, req.Id)
	if err != nil {
		return nil, err
	}

	protoOp, err := modelToProto(op)
	if err != nil {
		return nil, err
	}

	return &postgresv1.GetLangOperationResponse{Operation: protoOp}, nil
}

// ListLangOperations exposes paginated access to stored operations.
func (s *LangService) ListLangOperations(ctx context.Context, req *postgresv1.ListLangOperationsRequest) (*postgresv1.ListLangOperationsResponse, error) {
	if req == nil {
		req = &postgresv1.ListLangOperationsRequest{}
	}

	filters, err := listFiltersFromProto(req)
	if err != nil {
		return nil, err
	}

	operations, nextToken, err := s.repo.ListOperations(ctx, filters)
	if err != nil {
		return nil, err
	}

	protoOps := make([]*postgresv1.LangOperation, 0, len(operations))
	for _, op := range operations {
		protoOp, err := modelToProto(op)
		if err != nil {
			return nil, err
		}
		protoOps = append(protoOps, protoOp)
	}

	return &postgresv1.ListLangOperationsResponse{
		Operations:    protoOps,
		NextPageToken: nextToken,
	}, nil
}

// GetAnalytics returns aggregated statistics for operations.
func (s *LangService) GetAnalytics(ctx context.Context, req *postgresv1.AnalyticsRequest) (*postgresv1.AnalyticsResponse, error) {
	filters := repository.AnalyticsFilters{}
	if req != nil {
		if req.StartTime != nil {
			start := req.StartTime.AsTime()
			filters.StartTime = &start
		}
		if req.EndTime != nil {
			end := req.EndTime.AsTime()
			filters.EndTime = &end
		}
		if req.LibraryType != "" {
			filters.LibraryType = req.LibraryType
		}
	}

	summary, err := s.repo.GetAnalytics(ctx, filters)
	if err != nil {
		return nil, err
	}

	resp := &postgresv1.AnalyticsResponse{
		TotalOperations:  summary.TotalOperations,
		SuccessRate:      summary.SuccessRate,
		AverageLatencyMs: summary.AverageLatency,
		ErrorBreakdown:   summary.ErrorBreakdown,
		GeneratedAt:      timestamppb.New(summary.GeneratedAt),
	}

	for _, stat := range summary.LibraryStats {
		resp.LibraryStats = append(resp.LibraryStats, &postgresv1.LibraryStats{
			LibraryType:      stat.LibraryType,
			TotalOperations:  stat.TotalOperations,
			SuccessRate:      stat.SuccessRate,
			AverageLatencyMs: stat.AverageLatency,
			ErrorCount:       stat.ErrorCount,
		})
	}

	for _, trend := range summary.PerformanceTrend {
		resp.PerformanceTrends = append(resp.PerformanceTrends, &postgresv1.PerformanceTrend{
			Date:             timestamppb.New(trend.Date),
			LibraryType:      trend.LibraryType,
			Operations:       trend.Operations,
			SuccessRate:      trend.SuccessRate,
			AverageLatencyMs: trend.AverageLatency,
		})
	}

	return resp, nil
}

// CleanupOperations deletes stale records.
func (s *LangService) CleanupOperations(ctx context.Context, req *postgresv1.CleanupRequest) (*postgresv1.CleanupResponse, error) {
	if req == nil || req.OlderThan == nil {
		return nil, fmt.Errorf("older_than timestamp required")
	}

	result, err := s.repo.CleanupOperations(ctx, req.OlderThan.AsTime())
	if err != nil {
		return nil, err
	}

	return &postgresv1.CleanupResponse{Deleted: result.Deleted}, nil
}

func listFiltersFromProto(req *postgresv1.ListLangOperationsRequest) (repository.ListFilters, error) {
	filters := repository.ListFilters{
		LibraryType: req.LibraryType,
		SessionID:   req.SessionId,
		PageSize:    int(req.PageSize),
		PageToken:   req.PageToken,
	}

	switch req.Status {
	case postgresv1.OperationStatus_OPERATION_STATUS_RUNNING:
		filters.Status = models.OperationStatusRunning
	case postgresv1.OperationStatus_OPERATION_STATUS_SUCCESS:
		filters.Status = models.OperationStatusSuccess
	case postgresv1.OperationStatus_OPERATION_STATUS_ERROR:
		filters.Status = models.OperationStatusError
	default:
		filters.Status = models.OperationStatusUnspecified
	}

	if req.CreatedAfter != nil {
		after := req.CreatedAfter.AsTime()
		filters.CreatedAfter = &after
	}

	if req.CreatedBefore != nil {
		before := req.CreatedBefore.AsTime()
		filters.CreatedBefore = &before
	}

	if filters.PageSize == 0 {
		filters.PageSize = 50
	}

	return filters, nil
}

func protoToModel(p *postgresv1.LangOperation) (*models.LangOperation, error) {
	var (
		input  map[string]any
		output map[string]any
		err    error
	)

	if p.Input != nil {
		input = p.Input.AsMap()
	} else {
		input = make(map[string]any)
	}

	if p.Output != nil {
		output = p.Output.AsMap()
	} else {
		output = make(map[string]any)
	}

	op := &models.LangOperation{
		ID:           p.Id,
		LibraryType:  p.LibraryType,
		Operation:    p.Operation,
		Input:        input,
		Output:       output,
		Error:        p.Error,
		LatencyMs:    p.LatencyMs,
		SessionID:    p.SessionId,
		UserIDHash:   p.UserIdHash,
		PrivacyLevel: p.PrivacyLevel,
	}

	if p.CreatedAt != nil {
		op.CreatedAt = p.CreatedAt.AsTime()
	}

	if p.CompletedAt != nil {
		completed := p.CompletedAt.AsTime()
		op.CompletedAt = &completed
	}

	op.Status, err = protoStatusToModel(p.Status)
	if err != nil {
		return nil, err
	}

	return op, nil
}

func modelToProto(op *models.LangOperation) (*postgresv1.LangOperation, error) {
	input, err := structpb.NewStruct(op.Input)
	if err != nil {
		return nil, fmt.Errorf("encode input: %w", err)
	}

	output, err := structpb.NewStruct(op.Output)
	if err != nil {
		return nil, fmt.Errorf("encode output: %w", err)
	}

	protoOp := &postgresv1.LangOperation{
		Id:           op.ID,
		LibraryType:  op.LibraryType,
		Operation:    op.Operation,
		Input:        input,
		Output:       output,
		Error:        op.Error,
		LatencyMs:    op.LatencyMs,
		SessionId:    op.SessionID,
		UserIdHash:   op.UserIDHash,
		PrivacyLevel: op.PrivacyLevel,
		Status:       modelStatusToProto(op.Status),
	}

	if !op.CreatedAt.IsZero() {
		protoOp.CreatedAt = timestamppb.New(op.CreatedAt)
	}

	if op.CompletedAt != nil {
		protoOp.CompletedAt = timestamppb.New(*op.CompletedAt)
	}

	return protoOp, nil
}

func protoStatusToModel(status postgresv1.OperationStatus) (models.OperationStatus, error) {
	switch status {
	case postgresv1.OperationStatus_OPERATION_STATUS_RUNNING:
		return models.OperationStatusRunning, nil
	case postgresv1.OperationStatus_OPERATION_STATUS_SUCCESS:
		return models.OperationStatusSuccess, nil
	case postgresv1.OperationStatus_OPERATION_STATUS_ERROR:
		return models.OperationStatusError, nil
	case postgresv1.OperationStatus_OPERATION_STATUS_UNSPECIFIED:
		fallthrough
	default:
		return models.OperationStatusUnspecified, nil
	}
}

func modelStatusToProto(status models.OperationStatus) postgresv1.OperationStatus {
	switch status {
	case models.OperationStatusRunning:
		return postgresv1.OperationStatus_OPERATION_STATUS_RUNNING
	case models.OperationStatusSuccess:
		return postgresv1.OperationStatus_OPERATION_STATUS_SUCCESS
	case models.OperationStatusError:
		return postgresv1.OperationStatus_OPERATION_STATUS_ERROR
	default:
		return postgresv1.OperationStatus_OPERATION_STATUS_UNSPECIFIED
	}
}
