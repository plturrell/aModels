package boolq

import (
	"fmt"
)

// BoolQError represents domain-specific errors
type BoolQError struct {
	Code    string
	Message string
	Cause   error
}

func (e *BoolQError) Error() string {
	if e.Cause != nil {
		return fmt.Sprintf("[%s] %s: %v", e.Code, e.Message, e.Cause)
	}
	return fmt.Sprintf("[%s] %s", e.Code, e.Message)
}

func (e *BoolQError) Unwrap() error {
	return e.Cause
}

// Error codes
const (
	ErrCodeInvalidInput     = "INVALID_INPUT"
	ErrCodeGenerationFailed = "GENERATION_FAILED"
	ErrCodeValidationFailed = "VALIDATION_FAILED"
	ErrCodeModelUnavailable = "MODEL_UNAVAILABLE"
	ErrCodeInsufficientData = "INSUFFICIENT_DATA"
	ErrCodeTemplateError    = "TEMPLATE_ERROR"
)

// NewBoolQError creates a new error
func NewBoolQError(code, message string, cause error) *BoolQError {
	return &BoolQError{
		Code:    code,
		Message: message,
		Cause:   cause,
	}
}

// ResilientFactory wraps factory with error handling
type ResilientFactory struct {
	*GenericBoolQFactory
	ErrorHandler *ErrorHandler
}

type ErrorHandler struct {
	RetryAttempts int
	FallbackMode  bool
}

func NewResilientFactory() *ResilientFactory {
	return &ResilientFactory{
		GenericBoolQFactory: NewGenericBoolQFactory(),
		ErrorHandler: &ErrorHandler{
			RetryAttempts: 3,
			FallbackMode:  true,
		},
	}
}

func (f *ResilientFactory) MapWithRetry(data interface{}) ([]interface{}, error) {
	var lastErr error

	for attempt := 0; attempt < f.ErrorHandler.RetryAttempts; attempt++ {
		// Validate input
		if err := f.validateInput(data); err != nil {
			return nil, NewBoolQError(ErrCodeInvalidInput, "Input validation failed", err)
		}

		// Try mapping
		tasks, err := f.mapWithValidation(data)
		if err == nil {
			return tasks, nil
		}

		lastErr = err

		// If model unavailable, try fallback
		if f.ErrorHandler.FallbackMode && isModelError(err) {
			tasks, err := f.mapWithFallback(data)
			if err == nil {
				return tasks, nil
			}
		}
	}

	return nil, NewBoolQError(ErrCodeGenerationFailed, "All retry attempts failed", lastErr)
}

func (f *ResilientFactory) validateInput(data interface{}) error {
	row, ok := data.(map[string]string)
	if !ok {
		return fmt.Errorf("expected map[string]string, got %T", data)
	}

	if len(row) == 0 {
		return NewBoolQError(ErrCodeInsufficientData, "Empty input data", nil)
	}

	// Check for minimum required fields
	hasContent := false
	for _, value := range row {
		if len(value) > 10 {
			hasContent = true
			break
		}
	}

	if !hasContent {
		return NewBoolQError(ErrCodeInsufficientData, "Insufficient content in input", nil)
	}

	return nil
}

func (f *ResilientFactory) mapWithValidation(data interface{}) ([]interface{}, error) {
	row := data.(map[string]string)

	// Detect domain
	domain := f.detectDomain(row)
	adapter, exists := f.Adapters[domain]
	if !exists {
		adapter = f.Adapters["general"]
	}

	// Extract passages with validation
	passages := adapter.ExtractPassages(row)
	if len(passages) == 0 {
		return nil, NewBoolQError(ErrCodeInsufficientData, "No passages extracted", nil)
	}

	var tasks []interface{}

	for _, passage := range passages {
		// Validate passage
		if len(passage) < 20 {
			continue // Skip too short passages
		}

		// Generate questions with validation
		questionTemplates := adapter.GenerateQuestions(passage)
		if len(questionTemplates) == 0 {
			continue
		}

		for _, template := range questionTemplates {
			// Instantiate template with error handling
			question, err := f.instantiateTemplateWithValidation(template.Template, passage, row)
			if err != nil {
				continue // Skip failed templates
			}

			// Validate answer with error handling
			answer, confidence, inferenceType, err := f.validateAnswerWithErrorHandling(adapter, passage, question)
			if err != nil {
				continue
			}

			// Only include high-confidence tasks
			if confidence < 0.3 {
				continue
			}

			task := BoolQTask{
				Question:      question,
				Passage:       passage,
				Answer:        answer,
				InferenceType: inferenceType,
				Domain:        domain,
				Confidence:    confidence * template.Complexity,
			}

			tasks = append(tasks, task)
		}
	}

	if len(tasks) == 0 {
		return nil, NewBoolQError(ErrCodeGenerationFailed, "No valid tasks generated", nil)
	}

	return tasks, nil
}

func (f *ResilientFactory) instantiateTemplateWithValidation(template, passage string, data map[string]string) (string, error) {
	question := template

	// Handle {fact} placeholder
	if contains(question, "{fact}") {
		fact := extractRandomFact(passage)
		if len(fact) == 0 {
			return "", NewBoolQError(ErrCodeTemplateError, "Failed to extract fact", nil)
		}
		question = replaceString(question, "{fact}", fact)
	}

	// Handle {inference} placeholder
	if contains(question, "{inference}") {
		inference := generatePlausibleInference(passage)
		if len(inference) == 0 {
			return "", NewBoolQError(ErrCodeTemplateError, "Failed to generate inference", nil)
		}
		question = replaceString(question, "{inference}", inference)
	}

	// Validate final question
	if len(question) < 10 || !contains(question, "?") {
		return "", NewBoolQError(ErrCodeTemplateError, "Invalid question format", nil)
	}

	return question, nil
}

func (f *ResilientFactory) validateAnswerWithErrorHandling(adapter DomainAdapter, passage, question string) (bool, float64, InferenceType, error) {
	defer func() {
		if r := recover(); r != nil {
			// Handle panics in validation
		}
	}()

	answer, confidence, inferenceType := adapter.ValidateAnswer(passage, question)

	// Sanity checks
	if confidence < 0 || confidence > 1 {
		return false, 0, OtherInference, NewBoolQError(ErrCodeValidationFailed, "Invalid confidence score", nil)
	}

	return answer, confidence, inferenceType, nil
}

func (f *ResilientFactory) mapWithFallback(data interface{}) ([]interface{}, error) {
	// Use simpler, more reliable generation
	row := data.(map[string]string)

	// Generate basic tasks without complex validation
	var tasks []interface{}

	for key, value := range row {
		if len(value) > 20 {
			// Create simple yes/no question
			question := fmt.Sprintf("Does the data mention %s?", key)

			task := BoolQTask{
				Question:      question,
				Passage:       value,
				Answer:        true,
				InferenceType: Paraphrasing,
				Domain:        "general",
				Confidence:    0.6,
			}

			tasks = append(tasks, task)
		}
	}

	if len(tasks) == 0 {
		return nil, NewBoolQError(ErrCodeGenerationFailed, "Fallback generation failed", nil)
	}

	return tasks, nil
}

func isModelError(err error) bool {
	if boolqErr, ok := err.(*BoolQError); ok {
		return boolqErr.Code == ErrCodeModelUnavailable
	}
	return false
}

func replaceString(s, old, new string) string {
	result := ""
	i := 0
	for i < len(s) {
		if i+len(old) <= len(s) && s[i:i+len(old)] == old {
			result += new
			i += len(old)
		} else {
			result += string(s[i])
			i++
		}
	}
	return result
}
