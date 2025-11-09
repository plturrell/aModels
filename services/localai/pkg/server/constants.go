// Package server constants provides centralized constants for the server package.
// This includes request timeouts, default values, HTTP headers, error codes,
// and other configuration constants used throughout the server implementation.
package server

import "time"

// Request timeouts
const (
	// RequestTimeoutDefault is the default timeout for chat completion requests
	RequestTimeoutDefault = 2 * time.Minute
	// RequestTimeoutEmbeddings is the timeout for embeddings requests
	RequestTimeoutEmbeddings = 30 * time.Second
	// RequestTimeoutStreaming is the timeout for streaming requests (longer for real-time generation)
	RequestTimeoutStreaming = 60 * time.Second
	// RequestTimeoutFunctionCalling is the timeout for function calling requests
	RequestTimeoutFunctionCalling = 30 * time.Second
)

// Default generation parameters
const (
	// DefaultMaxTokens is the default maximum number of tokens to generate
	DefaultMaxTokens = 512
	// DefaultTemperature is the default temperature for sampling
	DefaultTemperature = 0.7
	// DefaultTopP is the default top-p (nucleus) sampling parameter
	DefaultTopP = 0.9
	// DefaultTopK is the default top-k sampling parameter
	DefaultTopK = 50
)

// HTTP constants
const (
	// ContentTypeJSON is the JSON content type
	ContentTypeJSON = "application/json"
	// ContentTypeSSE is the Server-Sent Events content type
	ContentTypeSSE = "text/event-stream"
	// HeaderContentType is the Content-Type header name
	HeaderContentType = "Content-Type"
	// HeaderUserID is the X-User-ID header name
	HeaderUserID = "X-User-ID"
	// HeaderSessionID is the X-Session-ID header name
	HeaderSessionID = "X-Session-ID"
	// HeaderCacheControl is the Cache-Control header name
	HeaderCacheControl = "Cache-Control"
	// HeaderConnection is the Connection header name
	HeaderConnection = "Connection"
	// HeaderAccessControlAllowOrigin is the CORS header name
	HeaderAccessControlAllowOrigin = "Access-Control-Allow-Origin"
	// HeaderAccessControlAllowHeaders is the CORS headers header name
	HeaderAccessControlAllowHeaders = "Access-Control-Allow-Headers"
)

// Cache constants
const (
	// DefaultCacheTTL is the default time-to-live for cache entries
	DefaultCacheTTL = 24 * time.Hour
	// DefaultCleanupInterval is the default interval for cache cleanup
	DefaultCleanupInterval = 1 * time.Hour
)

// Streaming constants
const (
	// StreamingChunkSize is the number of characters per streaming chunk
	StreamingChunkSize = 50
	// StreamingChunkDelay is the delay between streaming chunks
	StreamingChunkDelay = 50 * time.Millisecond
)

// Backend type constants
const (
	// BackendTypeSafetensors is the safetensors backend type
	BackendTypeSafetensors = "safetensors"
	// BackendTypeGGUF is the GGUF backend type
	BackendTypeGGUF = "gguf"
	// BackendTypeTransformers is the HuggingFace transformers backend type
	BackendTypeTransformers = "hf-transformers"
	// BackendTypeDeepSeekOCR is the DeepSeek OCR backend type
	BackendTypeDeepSeekOCR = "deepseek-ocr"
)

// Domain constants
const (
	// DomainAuto is the auto-detect domain value
	DomainAuto = "auto"
	// DomainGeneral is the general/default domain
	DomainGeneral = "general"
	// DomainVaultGemma is the vaultgemma domain
	DomainVaultGemma = "vaultgemma"
)

// Error codes
const (
	// ErrorCodeInvalidRequest indicates an invalid request
	ErrorCodeInvalidRequest = "INVALID_REQUEST"
	// ErrorCodeModelNotFound indicates a model was not found
	ErrorCodeModelNotFound = "MODEL_NOT_FOUND"
	// ErrorCodeBackendUnavailable indicates a backend is unavailable
	ErrorCodeBackendUnavailable = "BACKEND_UNAVAILABLE"
	// ErrorCodeTimeout indicates a request timeout
	ErrorCodeTimeout = "REQUEST_TIMEOUT"
	// ErrorCodeInternalError indicates an internal server error
	ErrorCodeInternalError = "INTERNAL_ERROR"
	// ErrorCodeRateLimitExceeded indicates rate limit exceeded
	ErrorCodeRateLimitExceeded = "RATE_LIMIT_EXCEEDED"
)

// Memory conversion constants
const (
	// BytesPerKB is the number of bytes in a kilobyte
	BytesPerKB = 1024
	// BytesPerMB is the number of bytes in a megabyte
	BytesPerMB = BytesPerKB * 1024
	// BytesPerGB is the number of bytes in a gigabyte
	BytesPerGB = BytesPerMB * 1024
)

// Retry constants
const (
	// RetryMaxAttempts is the maximum number of retry attempts
	RetryMaxAttempts = 3
	// RetryBaseDelay is the base delay for exponential backoff
	RetryBaseDelay = 100 * time.Millisecond
	// RetryMaxDelay is the maximum delay for exponential backoff
	RetryMaxDelay = 5 * time.Second
	// RetryMultiplier is the multiplier for exponential backoff
	RetryMultiplier = 2.0
)

// Vision/OCR constants
const (
	// VisionDefaultTimeout is the default timeout for vision/OCR operations
	VisionDefaultTimeout = 30 * time.Second
)

// Anonymous user constants
const (
	// AnonymousUserID is the default user ID for anonymous users
	AnonymousUserID = "anonymous"
	// DefaultSessionID is the default session ID
	DefaultSessionID = "default"
)

// Error definitions
var (
	// ErrInvalidRequest indicates an invalid request
	ErrInvalidRequest = errors.New(ErrorCodeInvalidRequest)
	// ErrModelNotFound indicates a model was not found
	ErrModelNotFound = errors.New(ErrorCodeModelNotFound)
	// ErrBackendUnavailable indicates a backend is unavailable
	ErrBackendUnavailable = errors.New(ErrorCodeBackendUnavailable)
	// ErrTimeout indicates a request timeout
	ErrTimeout = errors.New(ErrorCodeTimeout)
	// ErrInternalError indicates an internal server error
	ErrInternalError = errors.New(ErrorCodeInternalError)
	// ErrRateLimitExceeded indicates rate limit exceeded
	ErrRateLimitExceeded = errors.New(ErrorCodeRateLimitExceeded)
)

