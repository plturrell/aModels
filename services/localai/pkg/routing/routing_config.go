package routing

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"
	"time"
)

// RoutingRule represents a routing rule for intelligent model selection
type RoutingRule struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	Description string            `json:"description"`
	Priority    int               `json:"priority"`    // Higher number = higher priority
	Conditions  []RuleCondition   `json:"conditions"`
	Actions     []RuleAction      `json:"actions"`
	Enabled     bool              `json:"enabled"`
	CreatedAt   time.Time         `json:"created_at"`
	UpdatedAt   time.Time         `json:"updated_at"`
	Metadata    map[string]string `json:"metadata"`
}

// RuleCondition represents a condition for a routing rule
type RuleCondition struct {
	Type        string      `json:"type"`        // "query_content", "user_domain", "complexity", "time", "load"
	Field       string      `json:"field"`       // Field to check
	Operator    string      `json:"operator"`    // "equals", "contains", "greater_than", "less_than", "regex"
	Value       interface{} `json:"value"`       // Value to compare against
	Negate      bool        `json:"negate"`      // Whether to negate the condition
	Description string      `json:"description"` // Human-readable description
}

// RuleAction represents an action to take when a rule matches
type RuleAction struct {
	Type        string            `json:"type"`        // "route_to_domain", "set_priority", "add_metadata", "log_event"
	Target      string            `json:"target"`     // Target domain or value
	Parameters  map[string]string `json:"parameters"` // Action parameters
	Description string            `json:"description"` // Human-readable description
}

// RoutingConfig represents the complete routing configuration
type RoutingConfig struct {
	Rules           []RoutingRule `json:"rules"`
	DefaultDomain   string        `json:"default_domain"`
	FallbackDomain  string        `json:"fallback_domain"`
	LoadBalancing   LoadBalancingConfig `json:"load_balancing"`
	Performance     PerformanceConfig   `json:"performance"`
	Monitoring      MonitoringConfig    `json:"monitoring"`
	Version         string        `json:"version"`
	LastUpdated     time.Time     `json:"last_updated"`
}

// LoadBalancingConfig represents load balancing configuration
type LoadBalancingConfig struct {
	Enabled        bool    `json:"enabled"`
	Strategy       string  `json:"strategy"`        // "round_robin", "least_connections", "weighted", "random"
	HealthCheck    bool    `json:"health_check"`
	MaxRetries     int     `json:"max_retries"`
	RetryDelay     int     `json:"retry_delay_ms"`
	CircuitBreaker bool    `json:"circuit_breaker"`
	Threshold      float64 `json:"failure_threshold"`
}

// PerformanceConfig represents performance configuration
type PerformanceConfig struct {
	CacheEnabled     bool          `json:"cache_enabled"`
	CacheTTL         time.Duration `json:"cache_ttl"`
	MaxConcurrency   int           `json:"max_concurrency"`
	RequestTimeout   time.Duration `json:"request_timeout"`
	ResponseTimeout  time.Duration `json:"response_timeout"`
	RateLimitEnabled bool          `json:"rate_limit_enabled"`
	RateLimitRPS     int           `json:"rate_limit_rps"`
}

// MonitoringConfig represents monitoring configuration
type MonitoringConfig struct {
	Enabled         bool     `json:"enabled"`
	MetricsEnabled  bool     `json:"metrics_enabled"`
	LoggingEnabled  bool     `json:"logging_enabled"`
	AlertingEnabled bool     `json:"alerting_enabled"`
	MetricsInterval time.Duration `json:"metrics_interval"`
	LogLevel        string   `json:"log_level"`
	AlertThresholds map[string]float64 `json:"alert_thresholds"`
}

// RoutingConfigManager manages routing configurations
type RoutingConfigManager struct {
	config     *RoutingConfig
	ruleEngine *RuleEngine
}

// RuleEngine evaluates routing rules
type RuleEngine struct {
	rules []RoutingRule
}

// NewRoutingConfigManager creates a new routing config manager
func NewRoutingConfigManager() *RoutingConfigManager {
	return &RoutingConfigManager{
		config: &RoutingConfig{
			Rules:          []RoutingRule{},
			DefaultDomain:  "general",
			FallbackDomain: "general",
			LoadBalancing: LoadBalancingConfig{
				Enabled:        true,
				Strategy:       "weighted",
				HealthCheck:    true,
				MaxRetries:     3,
				RetryDelay:     1000,
				CircuitBreaker: true,
				Threshold:      0.5,
			},
			Performance: PerformanceConfig{
				CacheEnabled:     true,
				CacheTTL:         time.Hour,
				MaxConcurrency:   100,
				RequestTimeout:   time.Minute,
				ResponseTimeout:  time.Minute * 2,
				RateLimitEnabled: true,
				RateLimitRPS:     100,
			},
			Monitoring: MonitoringConfig{
				Enabled:         true,
				MetricsEnabled:  true,
				LoggingEnabled:  true,
				AlertingEnabled: true,
				MetricsInterval: time.Minute,
				LogLevel:        "info",
				AlertThresholds: map[string]float64{
					"error_rate": 0.05,
					"latency":   5000, // ms
					"throughput": 1000, // requests per minute
				},
			},
			Version:     "1.0.0",
			LastUpdated: time.Now(),
		},
		ruleEngine: &RuleEngine{},
	}
}

// LoadRoutingConfig loads routing configuration from a file
func (rcm *RoutingConfigManager) LoadRoutingConfig(configPath string) error {
	data, err := os.ReadFile(configPath)
	if err != nil {
		return fmt.Errorf("failed to read config file: %w", err)
	}

	var config RoutingConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return fmt.Errorf("failed to parse config: %w", err)
	}

	rcm.config = &config
	rcm.ruleEngine.rules = config.Rules

	return nil
}

// SaveRoutingConfig saves routing configuration to a file
func (rcm *RoutingConfigManager) SaveRoutingConfig(configPath string) error {
	rcm.config.LastUpdated = time.Now()
	
	data, err := json.MarshalIndent(rcm.config, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	if err := os.WriteFile(configPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write config: %w", err)
	}

	return nil
}

// AddRoutingRule adds a new routing rule
func (rcm *RoutingConfigManager) AddRoutingRule(rule RoutingRule) error {
	rule.ID = fmt.Sprintf("rule_%d", time.Now().UnixNano())
	rule.CreatedAt = time.Now()
	rule.UpdatedAt = time.Now()
	
	rcm.config.Rules = append(rcm.config.Rules, rule)
	rcm.ruleEngine.rules = rcm.config.Rules
	
	return nil
}

// UpdateRoutingRule updates an existing routing rule
func (rcm *RoutingConfigManager) UpdateRoutingRule(ruleID string, rule RoutingRule) error {
	for i, r := range rcm.config.Rules {
		if r.ID == ruleID {
			rule.ID = ruleID
			rule.CreatedAt = r.CreatedAt
			rule.UpdatedAt = time.Now()
			rcm.config.Rules[i] = rule
			rcm.ruleEngine.rules = rcm.config.Rules
			return nil
		}
	}
	
	return fmt.Errorf("rule not found: %s", ruleID)
}

// DeleteRoutingRule deletes a routing rule
func (rcm *RoutingConfigManager) DeleteRoutingRule(ruleID string) error {
	for i, rule := range rcm.config.Rules {
		if rule.ID == ruleID {
			rcm.config.Rules = append(rcm.config.Rules[:i], rcm.config.Rules[i+1:]...)
			rcm.ruleEngine.rules = rcm.config.Rules
			return nil
		}
	}
	
	return fmt.Errorf("rule not found: %s", ruleID)
}

// GetRoutingRules returns all routing rules
func (rcm *RoutingConfigManager) GetRoutingRules() []RoutingRule {
	return rcm.config.Rules
}

// GetRoutingRule returns a specific routing rule
func (rcm *RoutingConfigManager) GetRoutingRule(ruleID string) (*RoutingRule, error) {
	for _, rule := range rcm.config.Rules {
		if rule.ID == ruleID {
			return &rule, nil
		}
	}
	
	return nil, fmt.Errorf("rule not found: %s", ruleID)
}

// EvaluateRules evaluates routing rules against a query context
func (rcm *RoutingConfigManager) EvaluateRules(query string, userDomains []string, complexity *QueryComplexity, contextData map[string]interface{}) ([]RoutingRule, error) {
	return rcm.ruleEngine.EvaluateRules(query, userDomains, complexity, contextData)
}

// EvaluateRules evaluates which rules match the given context
func (re *RuleEngine) EvaluateRules(query string, userDomains []string, complexity *QueryComplexity, contextData map[string]interface{}) ([]RoutingRule, error) {
	var matchingRules []RoutingRule
	
	for _, rule := range re.rules {
		if !rule.Enabled {
			continue
		}
		
		matches, err := re.evaluateRule(rule, query, userDomains, complexity, contextData)
		if err != nil {
			return nil, fmt.Errorf("failed to evaluate rule %s: %w", rule.ID, err)
		}
		
		if matches {
			matchingRules = append(matchingRules, rule)
		}
	}
	
	// Sort by priority (higher priority first)
	for i := 0; i < len(matchingRules)-1; i++ {
		for j := i + 1; j < len(matchingRules); j++ {
			if matchingRules[i].Priority < matchingRules[j].Priority {
				matchingRules[i], matchingRules[j] = matchingRules[j], matchingRules[i]
			}
		}
	}
	
	return matchingRules, nil
}

// evaluateRule evaluates a single rule against the context
func (re *RuleEngine) evaluateRule(rule RoutingRule, query string, userDomains []string, complexity *QueryComplexity, contextData map[string]interface{}) (bool, error) {
	for _, condition := range rule.Conditions {
		matches, err := re.evaluateCondition(condition, query, userDomains, complexity, contextData)
		if err != nil {
			return false, err
		}
		
		if !matches {
			return false, nil
		}
	}
	
	return true, nil
}

// evaluateCondition evaluates a single condition
func (re *RuleEngine) evaluateCondition(condition RuleCondition, query string, userDomains []string, complexity *QueryComplexity, contextData map[string]interface{}) (bool, error) {
	var fieldValue interface{}
	var err error
	
	// Get the field value based on condition type
	switch condition.Type {
	case "query_content":
		fieldValue, err = re.getQueryField(condition.Field, query)
	case "user_domain":
		fieldValue, err = re.getUserDomainField(condition.Field, userDomains)
	case "complexity":
		fieldValue, err = re.getComplexityField(condition.Field, complexity)
	case "time":
		fieldValue, err = re.getTimeField(condition.Field)
	case "load":
		fieldValue, err = re.getLoadField(condition.Field, contextData)
	default:
		return false, fmt.Errorf("unknown condition type: %s", condition.Type)
	}
	
	if err != nil {
		return false, err
	}
	
	// Evaluate the condition
	result, err := re.compareValues(fieldValue, condition.Operator, condition.Value)
	if err != nil {
		return false, err
	}
	
	// Apply negation if specified
	if condition.Negate {
		result = !result
	}
	
	return result, nil
}

// getQueryField gets a field value from the query
func (re *RuleEngine) getQueryField(field string, query string) (interface{}, error) {
	switch field {
	case "content":
		return query, nil
	case "length":
		return len(query), nil
	case "word_count":
		return len(strings.Fields(query)), nil
	case "contains_sql":
		return strings.Contains(strings.ToLower(query), "select") || 
			   strings.Contains(strings.ToLower(query), "insert") ||
			   strings.Contains(strings.ToLower(query), "update") ||
			   strings.Contains(strings.ToLower(query), "delete"), nil
	case "contains_vector":
		return strings.Contains(strings.ToLower(query), "vector") ||
			   strings.Contains(strings.ToLower(query), "embedding") ||
			   strings.Contains(strings.ToLower(query), "similarity"), nil
	case "contains_blockchain":
		return strings.Contains(strings.ToLower(query), "blockchain") ||
			   strings.Contains(strings.ToLower(query), "transaction") ||
			   strings.Contains(strings.ToLower(query), "smart contract"), nil
	default:
		return nil, fmt.Errorf("unknown query field: %s", field)
	}
}

// getUserDomainField gets a field value from user domains
func (re *RuleEngine) getUserDomainField(field string, userDomains []string) (interface{}, error) {
	switch field {
	case "domains":
		return userDomains, nil
	case "domain_count":
		return len(userDomains), nil
	case "has_sql_agent":
		for _, domain := range userDomains {
			if strings.Contains(domain, "SQLAgent") {
				return true, nil
			}
		}
		return false, nil
	case "has_vector_agent":
		for _, domain := range userDomains {
			if strings.Contains(domain, "VectorProcessingAgent") {
				return true, nil
			}
		}
		return false, nil
	case "has_blockchain_agent":
		for _, domain := range userDomains {
			if strings.Contains(domain, "BlockchainAgent") {
				return true, nil
			}
		}
		return false, nil
	default:
		return nil, fmt.Errorf("unknown user domain field: %s", field)
	}
}

// getComplexityField gets a field value from complexity analysis
func (re *RuleEngine) getComplexityField(field string, complexity *QueryComplexity) (interface{}, error) {
	switch field {
	case "score":
		return complexity.Score, nil
	case "token_count":
		return complexity.TokenCount, nil
	case "domain_specific":
		return complexity.DomainSpecific, nil
	case "technical_level":
		return complexity.TechnicalLevel, nil
	case "requires_reasoning":
		return complexity.RequiresReasoning, nil
	case "context_length":
		return complexity.ContextLength, nil
	default:
		return nil, fmt.Errorf("unknown complexity field: %s", field)
	}
}

// getTimeField gets a field value from current time
func (re *RuleEngine) getTimeField(field string) (interface{}, error) {
	now := time.Now()
	
	switch field {
	case "hour":
		return now.Hour(), nil
	case "day_of_week":
		return int(now.Weekday()), nil
	case "is_weekend":
		return now.Weekday() == time.Saturday || now.Weekday() == time.Sunday, nil
	case "is_business_hours":
		hour := now.Hour()
		return hour >= 9 && hour <= 17, nil
	default:
		return nil, fmt.Errorf("unknown time field: %s", field)
	}
}

// getLoadField gets a field value from load context
func (re *RuleEngine) getLoadField(field string, contextData map[string]interface{}) (interface{}, error) {
	if contextData == nil {
		return nil, fmt.Errorf("no context data available")
	}
	
	value, exists := contextData[field]
	if !exists {
		return nil, fmt.Errorf("field not found in context: %s", field)
	}
	
	return value, nil
}

// compareValues compares two values using the specified operator
func (re *RuleEngine) compareValues(fieldValue interface{}, operator string, conditionValue interface{}) (bool, error) {
	switch operator {
	case "equals":
		return fieldValue == conditionValue, nil
	case "not_equals":
		return fieldValue != conditionValue, nil
	case "contains":
		if str, ok := fieldValue.(string); ok {
			if condStr, ok := conditionValue.(string); ok {
				return strings.Contains(strings.ToLower(str), strings.ToLower(condStr)), nil
			}
		}
		return false, nil
	case "greater_than":
		return re.compareNumbers(fieldValue, conditionValue, ">")
	case "less_than":
		return re.compareNumbers(fieldValue, conditionValue, "<")
	case "greater_than_or_equal":
		return re.compareNumbers(fieldValue, conditionValue, ">=")
	case "less_than_or_equal":
		return re.compareNumbers(fieldValue, conditionValue, "<=")
	case "regex":
		if str, ok := fieldValue.(string); ok {
			if pattern, ok := conditionValue.(string); ok {
				matched, err := regexp.MatchString(pattern, str)
				return matched, err
			}
		}
		return false, nil
	case "in":
		return re.isIn(fieldValue, conditionValue)
	case "not_in":
		result, err := re.isIn(fieldValue, conditionValue)
		return !result, err
	default:
		return false, fmt.Errorf("unknown operator: %s", operator)
	}
}

// compareNumbers compares two numeric values
func (re *RuleEngine) compareNumbers(fieldValue, conditionValue interface{}, operator string) (bool, error) {
	fieldNum, ok := re.toFloat64(fieldValue)
	if !ok {
		return false, fmt.Errorf("field value is not numeric: %v", fieldValue)
	}
	
	condNum, ok := re.toFloat64(conditionValue)
	if !ok {
		return false, fmt.Errorf("condition value is not numeric: %v", conditionValue)
	}
	
	switch operator {
	case ">":
		return fieldNum > condNum, nil
	case "<":
		return fieldNum < condNum, nil
	case ">=":
		return fieldNum >= condNum, nil
	case "<=":
		return fieldNum <= condNum, nil
	default:
		return false, fmt.Errorf("unknown numeric operator: %s", operator)
	}
}

// toFloat64 converts a value to float64
func (re *RuleEngine) toFloat64(value interface{}) (float64, bool) {
	switch v := value.(type) {
	case float64:
		return v, true
	case float32:
		return float64(v), true
	case int:
		return float64(v), true
	case int32:
		return float64(v), true
	case int64:
		return float64(v), true
	default:
		return 0, false
	}
}

// isIn checks if a value is in a list
func (re *RuleEngine) isIn(fieldValue, conditionValue interface{}) (bool, error) {
	if list, ok := conditionValue.([]interface{}); ok {
		for _, item := range list {
			if fieldValue == item {
				return true, nil
			}
		}
		return false, nil
	}
	
	return false, fmt.Errorf("condition value is not a list: %v", conditionValue)
}

// GetConfig returns the current routing configuration
func (rcm *RoutingConfigManager) GetConfig() *RoutingConfig {
	return rcm.config
}

// UpdateConfig updates the routing configuration
func (rcm *RoutingConfigManager) UpdateConfig(config *RoutingConfig) {
	rcm.config = config
	rcm.ruleEngine.rules = config.Rules
}

// ValidateConfig validates the routing configuration
func (rcm *RoutingConfigManager) ValidateConfig() error {
	if rcm.config == nil {
		return fmt.Errorf("config is nil")
	}
	
	if rcm.config.DefaultDomain == "" {
		return fmt.Errorf("default domain is required")
	}
	
	if rcm.config.FallbackDomain == "" {
		return fmt.Errorf("fallback domain is required")
	}
	
	// Validate rules
	for i, rule := range rcm.config.Rules {
		if err := rcm.validateRule(rule); err != nil {
			return fmt.Errorf("rule %d (%s) is invalid: %w", i, rule.ID, err)
		}
	}
	
	return nil
}

// validateRule validates a single routing rule
func (rcm *RoutingConfigManager) validateRule(rule RoutingRule) error {
	if rule.ID == "" {
		return fmt.Errorf("rule ID is required")
	}
	
	if rule.Name == "" {
		return fmt.Errorf("rule name is required")
	}
	
	if len(rule.Conditions) == 0 {
		return fmt.Errorf("rule must have at least one condition")
	}
	
	if len(rule.Actions) == 0 {
		return fmt.Errorf("rule must have at least one action")
	}
	
	// Validate conditions
	for i, condition := range rule.Conditions {
		if err := rcm.validateCondition(condition); err != nil {
			return fmt.Errorf("condition %d is invalid: %w", i, err)
		}
	}
	
	// Validate actions
	for i, action := range rule.Actions {
		if err := rcm.validateAction(action); err != nil {
			return fmt.Errorf("action %d is invalid: %w", i, err)
		}
	}
	
	return nil
}

// validateCondition validates a single condition
func (rcm *RoutingConfigManager) validateCondition(condition RuleCondition) error {
	if condition.Type == "" {
		return fmt.Errorf("condition type is required")
	}
	
	if condition.Field == "" {
		return fmt.Errorf("condition field is required")
	}
	
	if condition.Operator == "" {
		return fmt.Errorf("condition operator is required")
	}
	
	validTypes := []string{"query_content", "user_domain", "complexity", "time", "load"}
	if !contains(validTypes, condition.Type) {
		return fmt.Errorf("invalid condition type: %s", condition.Type)
	}
	
	validOperators := []string{"equals", "not_equals", "contains", "greater_than", "less_than", "greater_than_or_equal", "less_than_or_equal", "regex", "in", "not_in"}
	if !contains(validOperators, condition.Operator) {
		return fmt.Errorf("invalid condition operator: %s", condition.Operator)
	}
	
	return nil
}

// validateAction validates a single action
func (rcm *RoutingConfigManager) validateAction(action RuleAction) error {
	if action.Type == "" {
		return fmt.Errorf("action type is required")
	}
	
	if action.Target == "" {
		return fmt.Errorf("action target is required")
	}
	
	validTypes := []string{"route_to_domain", "set_priority", "add_metadata", "log_event"}
	if !contains(validTypes, action.Type) {
		return fmt.Errorf("invalid action type: %s", action.Type)
	}
	
	return nil
}

// contains checks if a slice contains a string
func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}
