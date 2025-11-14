package api

import (
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/plturrell/aModels/services/catalog/breakdetection"
)

// EnhancedBreakDetectionHandler provides enhanced API features
type EnhancedBreakDetectionHandler struct {
	*BreakDetectionHandler
	apiVersion       string
	enableRateLimit  bool
	enablePagination bool
	enableValidation bool
	enableMetrics    bool
	rateLimiter      *RateLimiter
}

// NewEnhancedBreakDetectionHandler creates an enhanced handler
func NewEnhancedBreakDetectionHandler(
	baseHandler *BreakDetectionHandler,
	apiVersion string,
	rateLimiter *RateLimiter,
) *EnhancedBreakDetectionHandler {
	if rateLimiter == nil {
		// Use default rate limiter if none provided
		rateLimiter = DefaultRateLimiter()
	}
	return &EnhancedBreakDetectionHandler{
		BreakDetectionHandler: baseHandler,
		apiVersion:            apiVersion,
		enableRateLimit:       true,
		enablePagination:      true,
		enableValidation:      true,
		enableMetrics:         true,
		rateLimiter:           rateLimiter,
	}
}

// APIResponse represents a standardized API response
type APIResponse struct {
	Success   bool          `json:"success"`
	Data      interface{}   `json:"data,omitempty"`
	Error     *APIError     `json:"error,omitempty"`
	Meta      *ResponseMeta `json:"meta,omitempty"`
	Version   string        `json:"version"`
	Timestamp time.Time     `json:"timestamp"`
}

// APIError represents an API error
type APIError struct {
	Code    string      `json:"code"`
	Message string      `json:"message"`
	Details interface{} `json:"details,omitempty"`
}

// ResponseMeta contains response metadata
type ResponseMeta struct {
	Page       int   `json:"page,omitempty"`
	PageSize   int   `json:"page_size,omitempty"`
	Total      int   `json:"total,omitempty"`
	TotalPages int   `json:"total_pages,omitempty"`
	Duration   int64 `json:"duration_ms,omitempty"`
}

// PaginationParams represents pagination parameters
type PaginationParams struct {
	Page     int
	PageSize int
	Offset   int
	Limit    int
}

// HandleDetectBreaksEnhanced handles break detection with enhanced features
func (h *EnhancedBreakDetectionHandler) HandleDetectBreaksEnhanced(w http.ResponseWriter, r *http.Request) {
	startTime := time.Now()

	// API versioning
	if h.apiVersion != "" && !h.validateAPIVersion(r) {
		h.writeErrorResponse(w, http.StatusBadRequest, "INVALID_API_VERSION",
			"API version not supported", nil)
		return
	}

	// Rate limiting (placeholder - would integrate with actual rate limiter)
	if h.enableRateLimit {
		if !h.checkRateLimit(r) {
			h.writeErrorResponse(w, http.StatusTooManyRequests, "RATE_LIMIT_EXCEEDED",
				"Rate limit exceeded. Please try again later.", nil)
			return
		}
	}

	// Parse and validate request
	ctx := r.Context()
	var req breakdetection.DetectionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		h.writeErrorResponse(w, http.StatusBadRequest, "INVALID_REQUEST",
			"Failed to decode request", map[string]interface{}{"error": err.Error()})
		return
	}

	// Enhanced validation
	if h.enableValidation {
		if err := breakdetection.ValidateDetectionRequest(&req); err != nil {
			h.writeErrorResponse(w, http.StatusBadRequest, "VALIDATION_ERROR",
				"Request validation failed", map[string]interface{}{"error": err.Error()})
			return
		}
	}

	// Perform break detection
	result, err := h.breakDetectionService.DetectBreaks(ctx, &req)
	duration := time.Since(startTime)

	if err != nil {
		h.logger.Printf("Break detection failed: %v", err)
		h.writeErrorResponse(w, http.StatusInternalServerError, "DETECTION_FAILED",
			"Break detection failed", map[string]interface{}{
				"error":       err.Error(),
				"duration_ms": duration.Milliseconds(),
			})
		return
	}

	// Validate all detected breaks
	if h.enableValidation {
		validBreaks := make([]*breakdetection.Break, 0, len(result.Breaks))
		for _, b := range result.Breaks {
			if err := breakdetection.ValidateBreak(b, breakdetection.DefaultValidationConfig()); err != nil {
				h.logger.Printf("Warning: Invalid break detected: %v", err)
				continue
			}
			validBreaks = append(validBreaks, b)
		}
		result.Breaks = validBreaks
		result.TotalBreaksDetected = len(validBreaks)
	}

	// Write success response
	h.writeSuccessResponse(w, http.StatusOK, result, &ResponseMeta{
		Duration: duration.Milliseconds(),
	})
}

// HandleListBreaksEnhanced handles listing breaks with pagination
func (h *EnhancedBreakDetectionHandler) HandleListBreaksEnhanced(w http.ResponseWriter, r *http.Request) {
	startTime := time.Now()

	// Parse pagination
	pagination := h.parsePagination(r)
	if pagination.PageSize > 1000 {
		pagination.PageSize = 1000 // Max page size
	}

	systemName := r.URL.Query().Get("system")
	if systemName == "" {
		h.writeErrorResponse(w, http.StatusBadRequest, "MISSING_PARAMETER",
			"system query parameter is required", nil)
		return
	}

	status := breakdetection.BreakStatusOpen
	if statusStr := r.URL.Query().Get("status"); statusStr != "" {
		status = breakdetection.BreakStatus(statusStr)
	}

	// Get breaks with pagination
	ctx := r.Context()
	breaks, err := h.breakDetectionService.ListBreaks(ctx,
		breakdetection.SystemName(systemName), pagination.Limit, status)
	duration := time.Since(startTime)

	if err != nil {
		h.writeErrorResponse(w, http.StatusInternalServerError, "LIST_FAILED",
			"Failed to list breaks", map[string]interface{}{"error": err.Error()})
		return
	}

	// Apply pagination
	total := len(breaks)
	totalPages := (total + pagination.PageSize - 1) / pagination.PageSize
	start := pagination.Offset
	end := start + pagination.PageSize
	if end > total {
		end = total
	}

	var paginatedBreaks []*breakdetection.Break
	if start < total {
		paginatedBreaks = breaks[start:end]
	}

	// Write paginated response
	h.writeSuccessResponse(w, http.StatusOK, paginatedBreaks, &ResponseMeta{
		Page:       pagination.Page,
		PageSize:   pagination.PageSize,
		Total:      total,
		TotalPages: totalPages,
		Duration:   duration.Milliseconds(),
	})
}

// Helper methods
func (h *EnhancedBreakDetectionHandler) validateAPIVersion(r *http.Request) bool {
	version := r.Header.Get("X-API-Version")
	if version == "" {
		version = r.URL.Query().Get("api_version")
	}
	if version == "" {
		return true // Default to latest if not specified
	}
	return version == h.apiVersion || version == "latest"
}

func (h *EnhancedBreakDetectionHandler) checkRateLimit(r *http.Request) bool {
	if !h.enableRateLimit || h.rateLimiter == nil {
		return true
	}

	// Get client IP address
	clientIP := h.getClientIP(r)

	// Default rate limits: 100 requests per minute per IP, burst of 10
	// Different endpoints can have different limits
	rps := 100.0 / 60.0 // 100 requests per minute = ~1.67 requests per second
	burst := 10

	// Get or create limiter for this IP
	ipLimiter := h.rateLimiter.GetLimiter(clientIP, rps, burst)

	// Check per-IP rate limit first
	if !ipLimiter.Allow() {
		// IP limit exceeded
		return false
	}

	// Also check global rate limit
	if !h.rateLimiter.Allow() {
		return false
	}

	return true
}

// getClientIP extracts the client IP address from the request
func (h *EnhancedBreakDetectionHandler) getClientIP(r *http.Request) string {
	// Check X-Forwarded-For header (for proxies/load balancers)
	if forwarded := r.Header.Get("X-Forwarded-For"); forwarded != "" {
		// X-Forwarded-For can contain multiple IPs, take the first one
		ips := strings.Split(forwarded, ",")
		if len(ips) > 0 {
			return strings.TrimSpace(ips[0])
		}
	}

	// Check X-Real-IP header (for nginx)
	if realIP := r.Header.Get("X-Real-IP"); realIP != "" {
		return realIP
	}

	// Fallback to RemoteAddr
	// Remove port if present (format: "IP:port")
	ip := r.RemoteAddr
	if colonIdx := strings.LastIndex(ip, ":"); colonIdx != -1 {
		ip = ip[:colonIdx]
	}

	return ip
}

func (h *EnhancedBreakDetectionHandler) parsePagination(r *http.Request) PaginationParams {
	page := 1
	if pageStr := r.URL.Query().Get("page"); pageStr != "" {
		if p, err := strconv.Atoi(pageStr); err == nil && p > 0 {
			page = p
		}
	}

	pageSize := 100
	if sizeStr := r.URL.Query().Get("page_size"); sizeStr != "" {
		if s, err := strconv.Atoi(sizeStr); err == nil && s > 0 {
			pageSize = s
		}
	}

	return PaginationParams{
		Page:     page,
		PageSize: pageSize,
		Offset:   (page - 1) * pageSize,
		Limit:    pageSize,
	}
}

func (h *EnhancedBreakDetectionHandler) writeSuccessResponse(w http.ResponseWriter,
	status int, data interface{}, meta *ResponseMeta) {
	response := APIResponse{
		Success:   true,
		Data:      data,
		Meta:      meta,
		Version:   h.apiVersion,
		Timestamp: time.Now(),
	}
	h.writeJSONResponse(w, status, response)
}

func (h *EnhancedBreakDetectionHandler) writeErrorResponse(w http.ResponseWriter,
	status int, code string, message string, details interface{}) {
	response := APIResponse{
		Success: false,
		Error: &APIError{
			Code:    code,
			Message: message,
			Details: details,
		},
		Version:   h.apiVersion,
		Timestamp: time.Now(),
	}
	h.writeJSONResponse(w, status, response)
}

func (h *EnhancedBreakDetectionHandler) writeJSONResponse(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-API-Version", h.apiVersion)
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(data); err != nil {
		h.logger.Printf("Failed to encode response: %v", err)
	}
}
