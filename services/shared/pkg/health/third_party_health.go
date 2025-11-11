package health

import (
	"context"
	"fmt"
	"time"

	"github.com/apache/arrow-go/v18/arrow/flight"
	elasticsearch "github.com/elastic/go-elasticsearch/v7"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// HealthCheckResult represents the result of a health check.
type HealthCheckResult struct {
	Service   string
	Status    string // "healthy", "degraded", "unhealthy"
	Message   string
	LatencyMs int64
	Error     error
}

// CheckArrowFlightHealth checks the health of an Arrow Flight server.
func CheckArrowFlightHealth(ctx context.Context, addr string, timeout time.Duration) HealthCheckResult {
	start := time.Now()
	result := HealthCheckResult{
		Service: "arrow-flight",
		Status:  "unhealthy",
	}

	if timeout == 0 {
		timeout = 5 * time.Second
	}

	checkCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	client, err := flight.NewClientWithMiddlewareCtx(
		checkCtx,
		addr,
		nil,
		nil,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		result.Error = err
		result.Message = fmt.Sprintf("Failed to connect: %v", err)
		return result
	}
	defer client.Close()

	// Try to list flights as a health check
	_, err = client.ListFlights(checkCtx, nil)
	if err != nil {
		result.Error = err
		result.Message = fmt.Sprintf("ListFlights failed: %v", err)
		return result
	}

	result.Status = "healthy"
	result.Message = "Arrow Flight server is healthy"
	result.LatencyMs = time.Since(start).Milliseconds()
	return result
}

// CheckElasticsearchHealth checks the health of an Elasticsearch cluster.
func CheckElasticsearchHealth(ctx context.Context, client *elasticsearch.Client, timeout time.Duration) HealthCheckResult {
	start := time.Now()
	result := HealthCheckResult{
		Service: "elasticsearch",
		Status:  "unhealthy",
	}

	if timeout == 0 {
		timeout = 5 * time.Second
	}

	checkCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	res, err := client.Cluster.Health(
		client.Cluster.Health.WithContext(checkCtx),
	)
	if err != nil {
		result.Error = err
		result.Message = fmt.Sprintf("Health check failed: %v", err)
		return result
	}
	defer res.Body.Close()

	if res.IsError() {
		result.Error = fmt.Errorf("elasticsearch returned error: %s", res.String())
		result.Message = "Elasticsearch cluster health check failed"
		return result
	}

	var health struct {
		Status string `json:"status"`
	}
	if err := res.Decode(&health); err != nil {
		result.Error = err
		result.Message = "Failed to decode health response"
		return result
	}

	switch health.Status {
	case "green":
		result.Status = "healthy"
		result.Message = "Elasticsearch cluster is healthy (green)"
	case "yellow":
		result.Status = "degraded"
		result.Message = "Elasticsearch cluster is degraded (yellow)"
	case "red":
		result.Status = "unhealthy"
		result.Message = "Elasticsearch cluster is unhealthy (red)"
	default:
		result.Status = "degraded"
		result.Message = fmt.Sprintf("Elasticsearch cluster status: %s", health.Status)
	}

	result.LatencyMs = time.Since(start).Milliseconds()
	return result
}

// CheckConnectionPoolHealth checks the health of a connection pool.
func CheckConnectionPoolHealth(poolType, service string, currentSize, maxSize int) HealthCheckResult {
	result := HealthCheckResult{
		Service: fmt.Sprintf("%s-pool", poolType),
		Status:  "healthy",
	}

	utilization := float64(currentSize) / float64(maxSize)
	if utilization > 0.9 {
		result.Status = "degraded"
		result.Message = fmt.Sprintf("Pool utilization high: %.1f%%", utilization*100)
	} else if utilization > 0.95 {
		result.Status = "unhealthy"
		result.Message = fmt.Sprintf("Pool near capacity: %.1f%%", utilization*100)
	} else {
		result.Message = fmt.Sprintf("Pool healthy: %d/%d connections", currentSize, maxSize)
	}

	return result
}

// AggregateHealthResults aggregates multiple health check results.
func AggregateHealthResults(results []HealthCheckResult) HealthCheckResult {
	if len(results) == 0 {
		return HealthCheckResult{
			Service: "aggregate",
			Status:  "unhealthy",
			Message: "No health checks performed",
		}
	}

	healthy := 0
	degraded := 0
	unhealthy := 0

	for _, r := range results {
		switch r.Status {
		case "healthy":
			healthy++
		case "degraded":
			degraded++
		case "unhealthy":
			unhealthy++
		}
	}

	aggregate := HealthCheckResult{
		Service: "aggregate",
	}

	if unhealthy > 0 {
		aggregate.Status = "unhealthy"
		aggregate.Message = fmt.Sprintf("%d unhealthy, %d degraded, %d healthy", unhealthy, degraded, healthy)
	} else if degraded > 0 {
		aggregate.Status = "degraded"
		aggregate.Message = fmt.Sprintf("%d degraded, %d healthy", degraded, healthy)
	} else {
		aggregate.Status = "healthy"
		aggregate.Message = fmt.Sprintf("All %d services healthy", healthy)
	}

	return aggregate
}

