# Intelligent Model Routing System

The Intelligent Model Routing System provides sophisticated model selection based on query complexity, domain expertise, and performance considerations. It enables automatic routing of queries to the most appropriate models while maintaining high performance and reliability.

## Overview

The system consists of several key components:

1. **IntelligentRouter**: Core routing logic with query analysis and model selection
2. **EnhancedIntelligentRouter**: Advanced routing with configuration support and load balancing
3. **RoutingConfigManager**: Manages routing rules and configuration
4. **LoadBalancer**: Handles load balancing and health monitoring
5. **PerformanceMonitor**: Tracks performance metrics and alerts

## Features

### Query Analysis
- **Complexity Analysis**: Analyzes query complexity based on multiple factors
- **Domain Detection**: Identifies domain-specific queries automatically
- **Technical Level Assessment**: Determines the technical level required
- **Reasoning Requirements**: Detects queries requiring complex reasoning

### Model Selection
- **Capability Matching**: Matches model capabilities to query requirements
- **Performance Optimization**: Considers model speed, accuracy, and cost
- **Load Balancing**: Distributes load across available models
- **Health Monitoring**: Routes away from unhealthy models

### Configuration Management
- **Rule-Based Routing**: Configurable routing rules for different scenarios
- **Priority-Based Selection**: Rules with priority-based execution
- **Dynamic Configuration**: Runtime configuration updates
- **A/B Testing Support**: Support for testing different routing strategies

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Intelligent Router                       │
├─────────────────────────────────────────────────────────────┤
│  Query Analysis  │  Model Selection  │  Performance      │
│  - Complexity    │  - Capabilities   │  - Monitoring     │
│  - Domain        │  - Load Balancing  │  - Alerts        │
│  - Technical     │  - Health Check    │  - Metrics       │
└─────────────────────────────────────────────────────────────┘
```

### Routing Flow

1. **Query Analysis**: Analyze query complexity and requirements
2. **Rule Evaluation**: Apply routing rules based on query characteristics
3. **Model Selection**: Select the most appropriate model
4. **Load Balancing**: Apply load balancing and health checks
5. **Performance Tracking**: Record metrics and monitor performance

## Usage

### Basic Setup

```go
// Create domain manager
dm := domain.NewDomainManager()
err := dm.LoadDomainConfigs("../config/domains.json")
if err != nil {
    log.Fatal("Failed to load domain configs:", err)
}

// Create intelligent router
router := NewIntelligentRouter(dm)
err = router.InitializeCapabilities()
if err != nil {
    log.Fatal("Failed to initialize capabilities:", err)
}
```

### Enhanced Router Setup

```go
// Create enhanced router
enhancedRouter := NewEnhancedIntelligentRouter(dm)
err = enhancedRouter.InitializeEnhancedRouter("../config/routing_rules.json")
if err != nil {
    log.Fatal("Failed to initialize enhanced router:", err)
}
```

### Basic Routing

```go
// Route a query
query := "How do I optimize a SQL query for better performance?"
userDomains := []string{"0x5678-SQLAgent", "0x3579-VectorProcessingAgent", "general"}

decision, err := router.RouteQuery(context.Background(), query, userDomains, nil)
if err != nil {
    log.Printf("Failed to route query: %v", err)
    return
}

// Use the routing decision
if decision.SelectedDomain != "" {
    // Route to the selected domain
    log.Printf("Selected domain: %s (confidence: %.2f)", 
        decision.SelectedDomain, decision.Confidence)
    log.Printf("Reasoning: %s", decision.Reasoning)
}
```

### Enhanced Routing

```go
// Route with enhanced features
decision, err := enhancedRouter.RouteQueryEnhanced(context.Background(), query, userDomains, contextData)
if err != nil {
    log.Printf("Failed to route query: %v", err)
    return
}

// Check for alternative routes
if decision.Confidence < 0.7 {
    log.Printf("Low confidence routing, consider alternatives:")
    for _, alt := range decision.AlternativeRoutes {
        log.Printf("  - %s (score: %.2f): %s", alt.Domain, alt.Score, alt.Reasoning)
    }
}
```

## Configuration

### Routing Rules

The system supports configurable routing rules in JSON format:

```json
{
  "rules": [
    {
      "id": "rule_sql_queries",
      "name": "SQL Query Routing",
      "description": "Route SQL-related queries to SQL Agent",
      "priority": 100,
      "conditions": [
        {
          "type": "query_content",
          "field": "contains_sql",
          "operator": "equals",
          "value": true,
          "negate": false,
          "description": "Query contains SQL keywords"
        }
      ],
      "actions": [
        {
          "type": "route_to_domain",
          "target": "0x5678-SQLAgent",
          "parameters": {
            "reason": "SQL query detected"
          },
          "description": "Route to SQL Agent"
        }
      ],
      "enabled": true
    }
  ]
}
```

### Rule Conditions

#### Query Content Conditions
- `contains_sql`: Query contains SQL keywords
- `contains_vector`: Query contains vector-related keywords
- `contains_blockchain`: Query contains blockchain keywords
- `content`: Full query content
- `length`: Query length
- `word_count`: Number of words

#### Complexity Conditions
- `score`: Overall complexity score (0.0 to 1.0)
- `token_count`: Estimated token count
- `domain_specific`: Whether query is domain-specific
- `technical_level`: Technical level required
- `requires_reasoning`: Whether query requires complex reasoning
- `context_length`: Required context length

#### User Domain Conditions
- `domains`: Available user domains
- `domain_count`: Number of available domains
- `has_sql_agent`: Whether user has access to SQL agent
- `has_vector_agent`: Whether user has access to vector agent
- `has_blockchain_agent`: Whether user has access to blockchain agent

#### Time Conditions
- `hour`: Current hour (0-23)
- `day_of_week`: Day of week (0-6)
- `is_weekend`: Whether it's weekend
- `is_business_hours`: Whether it's business hours

#### Load Conditions
- `system_load`: System load percentage
- `domain_load`: Domain-specific load
- `queue_length`: Request queue length

### Rule Actions

#### Route to Domain
```json
{
  "type": "route_to_domain",
  "target": "0x5678-SQLAgent",
  "parameters": {
    "reason": "SQL query detected"
  }
}
```

#### Set Priority
```json
{
  "type": "set_priority",
  "target": "high",
  "parameters": {
    "value": "100"
  }
}
```

#### Add Metadata
```json
{
  "type": "add_metadata",
  "target": "business_hours",
  "parameters": {
    "value": "true"
  }
}
```

#### Log Event
```json
{
  "type": "log_event",
  "target": "routing_decision",
  "parameters": {
    "level": "info"
  }
}
```

## Performance Monitoring

### Metrics Tracking

The system tracks comprehensive performance metrics:

```go
// Get performance metrics
metrics := enhancedRouter.GetPerformanceMetrics()
for domain, metric := range metrics {
    log.Printf("Domain: %s", domain)
    log.Printf("  Total Requests: %d", metric.TotalRequests)
    log.Printf("  Successful: %d", metric.SuccessfulRequests)
    log.Printf("  Failed: %d", metric.FailedRequests)
    log.Printf("  Error Rate: %.2f%%", metric.ErrorRate*100)
    log.Printf("  Average Latency: %v", metric.AverageLatency)
    log.Printf("  Throughput: %.2f req/min", metric.Throughput)
}
```

### Alerts

The system provides automatic alerting for performance issues:

```go
// Get current alerts
alerts := enhancedRouter.GetAlerts()
for _, alert := range alerts {
    log.Printf("Alert: %s - %s (Severity: %s)", 
        alert.Type, alert.Message, alert.Severity)
    log.Printf("  Domain: %s, Value: %.2f, Threshold: %.2f", 
        alert.Domain, alert.Value, alert.Threshold)
}
```

### Domain Health

Monitor domain health status:

```go
// Get domain statistics
stats := enhancedRouter.GetDomainStats()
for domain, stat := range stats {
    log.Printf("Domain: %s", domain)
    log.Printf("  Health Status: %s", stat.HealthStatus)
    log.Printf("  Total Requests: %d", stat.TotalRequests)
    log.Printf("  Error Rate: %.2f%%", stat.ErrorRate*100)
    log.Printf("  Average Latency: %v", stat.AverageLatency)
}
```

## Load Balancing

### Strategies

The system supports multiple load balancing strategies:

1. **Round Robin**: Distribute requests evenly across domains
2. **Least Connections**: Route to domain with fewest active connections
3. **Weighted**: Route based on domain weights
4. **Random**: Random selection from available domains

### Health Checks

Automatic health monitoring:

```go
// Update domain health manually
enhancedRouter.UpdateDomainHealth("0x5678-SQLAgent", "healthy")

// Record successful request
enhancedRouter.RecordRequestSuccess("0x5678-SQLAgent", time.Millisecond*100)

// Record failed request
enhancedRouter.RecordRequestFailure("0x5678-SQLAgent", "timeout")
```

### Circuit Breaker

Automatic circuit breaker for unhealthy domains:

```go
// Configure circuit breaker
config := enhancedRouter.GetConfig()
config.LoadBalancing.CircuitBreaker = true
config.LoadBalancing.Threshold = 0.5 // 50% error rate threshold
```

## Advanced Features

### Custom Routing Rules

Create custom routing rules for specific scenarios:

```go
// Add custom routing rule
rule := RoutingRule{
    ID:          "custom_rule",
    Name:        "Custom Routing Rule",
    Description: "Route based on custom criteria",
    Priority:    90,
    Conditions: []RuleCondition{
        {
            Type:        "query_content",
            Field:       "content",
            Operator:    "contains",
            Value:       "custom_keyword",
            Negate:      false,
            Description: "Query contains custom keyword",
        },
    },
    Actions: []RuleAction{
        {
            Type:        "route_to_domain",
            Target:      "custom_domain",
            Parameters:  map[string]string{"reason": "Custom rule matched"},
            Description: "Route to custom domain",
        },
    },
    Enabled: true,
}

err := enhancedRouter.configManager.AddRoutingRule(rule)
if err != nil {
    log.Printf("Failed to add custom rule: %v", err)
}
```

### Performance Optimization

Optimize routing performance:

```go
// Configure performance settings
config := enhancedRouter.GetConfig()
config.Performance.CacheEnabled = true
config.Performance.CacheTTL = time.Hour
config.Performance.MaxConcurrency = 100
config.Performance.RequestTimeout = time.Minute
config.Performance.RateLimitEnabled = true
config.Performance.RateLimitRPS = 100
```

### Monitoring Configuration

Configure monitoring and alerting:

```go
// Configure monitoring
config := enhancedRouter.GetConfig()
config.Monitoring.Enabled = true
config.Monitoring.MetricsEnabled = true
config.Monitoring.LoggingEnabled = true
config.Monitoring.AlertingEnabled = true
config.Monitoring.MetricsInterval = time.Minute
config.Monitoring.LogLevel = "info"
config.Monitoring.AlertThresholds = map[string]float64{
    "error_rate": 0.05,
    "latency":    5000,
    "throughput": 1000,
}
```

## Best Practices

### Rule Design
1. **Start Simple**: Begin with basic domain-based routing rules
2. **Test Thoroughly**: Test rules with various query types
3. **Monitor Performance**: Track rule effectiveness and performance
4. **Iterate**: Continuously improve rules based on metrics

### Performance
1. **Cache Results**: Enable caching for frequently accessed queries
2. **Monitor Metrics**: Track performance metrics continuously
3. **Set Alerts**: Configure appropriate alert thresholds
4. **Load Balance**: Use load balancing to distribute load

### Reliability
1. **Health Checks**: Implement comprehensive health monitoring
2. **Circuit Breakers**: Use circuit breakers for fault tolerance
3. **Fallbacks**: Always have fallback routing options
4. **Monitoring**: Monitor system health and performance

## Troubleshooting

### Common Issues

1. **Low Confidence Routing**
   - Check if query matches domain keywords
   - Verify model capabilities
   - Review routing rules

2. **Performance Issues**
   - Check load balancing configuration
   - Monitor domain health
   - Review performance metrics

3. **Rule Not Matching**
   - Verify rule conditions
   - Check rule priority
   - Test with sample queries

### Debugging

Enable detailed logging:

```go
// Enable debug logging
config := enhancedRouter.GetConfig()
config.Monitoring.LogLevel = "debug"
```

Monitor routing decisions:

```go
// Log routing decisions
decision, err := enhancedRouter.RouteQueryEnhanced(ctx, query, userDomains, contextData)
if err != nil {
    log.Printf("Routing failed: %v", err)
} else {
    log.Printf("Routing decision: %+v", decision)
}
```

## Conclusion

The Intelligent Model Routing System provides a powerful foundation for automatic model selection based on query analysis, domain expertise, and performance considerations. It enables sophisticated routing strategies while maintaining high performance and reliability.

The system is designed to be:
- **Intelligent**: Sophisticated query analysis and model selection
- **Configurable**: Flexible rule-based routing configuration
- **Performant**: Optimized for high-throughput scenarios
- **Reliable**: Comprehensive health monitoring and fault tolerance
- **Scalable**: Support for large-scale deployments

This makes it an essential component for building sophisticated AI systems that can automatically route queries to the most appropriate models while maintaining optimal performance and reliability.
