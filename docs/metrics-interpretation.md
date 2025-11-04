# Metrics Interpretation and Action System

This document describes how information theory metrics are interpreted and trigger actionable responses in the Extract service.

## Overview

The metrics interpretation system (`internal/processing/metrics_interpreter.go`) analyzes information theory metrics and provides:

1. **Quality Assessment**: Score (0.0-1.0) and level (excellent/good/fair/poor/critical)
2. **Issue Identification**: Specific problems detected in the data
3. **Recommendations**: Actionable suggestions for improvement
4. **Processing Decisions**: Whether to reject, warn, or proceed
5. **Strategy Selection**: Recommended processing approach

---

## Interpretation Process

### 1. Metric Analysis

The interpreter analyzes three key metrics:

- **Metadata Entropy**: Schema diversity (higher = more diverse)
- **KL Divergence**: Deviation from ideal distribution (lower = better)
- **Column Count**: Data sufficiency for meaningful metrics

### 2. Quality Score Calculation

Quality score is calculated as a weighted combination:

```go
quality_score = (entropy_score * 0.4) + (kl_score * 0.4) + (column_score * 0.2)
```

**Entropy Score:**
- `1.0` if entropy is between 1.0-4.0 (optimal range)
- `0.5` if entropy < 1.0 (low diversity)
- `0.3-1.0` if entropy > 4.0 (very high, may indicate inconsistency)

**KL Score:**
- `1.0` if KL divergence ≤ 0.5 (good)
- `0.5-1.0` if KL divergence 0.5-1.0 (warning range)
- `0.0` if KL divergence > 1.0 (critical)

**Column Score:**
- `1.0` if column count ≥ 5
- `0.0-1.0` proportional if < 5

### 3. Quality Level Assignment

Based on quality score:

| Score Range | Quality Level | Meaning |
|------------|--------------|---------|
| 0.9-1.0 | Excellent | High quality, no issues |
| 0.7-0.9 | Good | Minor issues, acceptable |
| 0.5-0.7 | Fair | Some concerns, review recommended |
| 0.3-0.5 | Poor | Significant issues, validation needed |
| 0.0-0.3 | Critical | Severe problems, reject processing |

---

## Actions Taken

### 1. Rejection (Critical Quality)

**When:** KL divergence > 1.0 or quality score < 0.3

**Action:**
```go
if interpretation.ShouldReject {
    // Return HTTP 422 with detailed error
    handlers.WriteJSON(w, http.StatusUnprocessableEntity, map[string]any{
        "error": "Graph processing rejected due to data quality issues",
        "quality_level": interpretation.QualityLevel,
        "quality_score": interpretation.QualityScore,
        "issues": interpretation.Issues,
        "recommendations": interpretation.Recommendations,
    })
    return
}
```

**Client Impact:**
- Request is rejected before graph processing
- Client receives detailed error with recommendations
- Prevents storing low-quality data

### 2. Warnings (Quality Issues)

**When:** Quality score < 0.7 or specific thresholds exceeded

**Action:**
```go
if interpretation.ShouldWarn {
    // Log warnings
    for _, issue := range interpretation.Issues {
        s.logger.Printf("WARNING: %s", issue)
    }
    // Include in response
    response["warnings"] = interpretation.Issues
}
```

**Client Impact:**
- Processing continues but with warnings
- Response includes `warnings` array
- Client can decide whether to proceed

### 3. Processing Strategy Recommendations

**Strategies:**
- `standard`: Normal processing (quality score ≥ 0.7)
- `enhanced`: Additional validation (quality score 0.5-0.7)
- `simplified`: Reduced processing (quality score 0.3-0.5)
- `skip`: Skip processing (quality score < 0.3)

**Usage:**
```go
processingFlags := processing.GetProcessingFlags(interpretation)
if processingFlags["enhanced_processing"] {
    // Apply additional validation
}
if processingFlags["simplified_processing"] {
    // Use simplified processing
}
```

---

## Response Format

Graph processing responses now include quality assessment:

```json
{
  "nodes": [...],
  "edges": [...],
  "metadata_entropy": 2.345,
  "kl_divergence": 0.123,
  "metrics": {
    "metadata_entropy": 2.345,
    "kl_divergence": 0.123,
    "actual_distribution": {...},
    "ideal_distribution": {...},
    "column_count": 45
  },
  "quality": {
    "score": 0.85,
    "level": "good",
    "issues": [],
    "recommendations": [],
    "processing_strategy": "standard",
    "needs_validation": false,
    "needs_review": false
  },
  "warnings": []
}
```

**Quality Object Fields:**
- `score`: 0.0-1.0 quality score
- `level`: "excellent", "good", "fair", "poor", or "critical"
- `issues`: Array of specific problems detected
- `recommendations`: Array of actionable suggestions
- `processing_strategy`: Recommended processing approach
- `needs_validation`: Whether additional validation is recommended
- `needs_review`: Whether human review is recommended

---

## Thresholds

Default thresholds (configurable via `MetricsThresholds`):

```go
LowEntropyThreshold:  1.0  // Warn if entropy < 1.0
HighEntropyThreshold: 4.0  // Info if entropy > 4.0
WarningKLThreshold:   0.5  // Warn if KL divergence > 0.5
ErrorKLThreshold:     1.0  // Error if KL divergence > 1.0
MinColumnCount:       5    // Minimum columns for reliable metrics
```

---

## Issue Detection

### Low Column Count

**Issue:** `"Low column count (N) - insufficient data for reliable metrics"`

**Trigger:** Column count < 5

**Recommendation:** `"Include more columns or sources to improve metric reliability"`

### Low Metadata Entropy

**Issue:** `"Low metadata entropy (X.XXX) - schema has low diversity, may indicate homogeneous data types"`

**Trigger:** Entropy < 1.0

**Recommendation:** `"Consider enriching schema with additional data types or sources"`

**Action:** `needs_review = true`

### High Metadata Entropy

**Issue:** `"High metadata entropy (X.XXX) - schema has high diversity, may indicate inconsistent data types"`

**Trigger:** Entropy > 4.0

**Recommendation:** None (informational)

**Action:** `needs_validation = true`

### High KL Divergence

**Issue:** `"High KL divergence (X.XXX) - data type distribution deviates significantly from ideal"`

**Trigger:** KL divergence > 0.5

**Recommendations:**
- `"Review data sources for type consistency and data quality issues"`
- `"Consider adjusting ideal_distribution to match actual data patterns"`

**Action:** `needs_validation = true`

### Very High KL Divergence

**Issue:** `"Very high KL divergence (X.XXX) - data type distribution is highly abnormal, data quality concerns"`

**Trigger:** KL divergence > 1.0

**Recommendations:** Same as high KL divergence

**Actions:**
- `needs_review = true`
- `should_reject = true` (critical)

---

## Client Usage Examples

### Example 1: Check Quality Before Processing

```python
response = requests.post("/graph", json=graph_data)
data = response.json()

if data.get("quality", {}).get("level") == "critical":
    print("WARNING: Data quality is critical, processing was rejected")
    print("Issues:", data["quality"]["issues"])
    print("Recommendations:", data["quality"]["recommendations"])
    return

if data.get("warnings"):
    print("Warnings:", data["warnings"])
    # Decide whether to proceed
```

### Example 2: Use Processing Strategy

```python
strategy = data["quality"]["processing_strategy"]

if strategy == "enhanced":
    # Apply additional validation
    validate_graph(data["nodes"], data["edges"])
elif strategy == "simplified":
    # Use simplified processing
    process_simplified(data["nodes"])
```

### Example 3: Track Quality Over Time

```python
# Store quality metrics for trend analysis
quality_metrics = {
    "timestamp": datetime.now(),
    "score": data["quality"]["score"],
    "level": data["quality"]["level"],
    "entropy": data["metadata_entropy"],
    "kl_divergence": data["kl_divergence"],
}
save_to_telemetry(quality_metrics)
```

---

## Future Enhancements

### 1. Adaptive Thresholds

Learn optimal thresholds from historical data:
- Adjust thresholds based on domain (e.g., financial vs. social media)
- Learn from successful vs. failed processing

### 2. Automatic Corrective Actions

Automatically apply fixes:
- Adjust ideal distribution to match actual patterns
- Suggest schema enrichment strategies
- Trigger data quality improvements

### 3. Metric-Based Routing

Route to different processors:
- High-quality data → standard processor
- Low-quality data → enhanced validator
- Critical data → human review queue

### 4. Predictive Quality Assessment

Predict quality before processing:
- Use early metrics to estimate final quality
- Stop processing early if quality is poor
- Save computational resources

---

## References

- [Information Theory Metrics](./extract-metrics.md)
- [Metrics Integration Flow](./information-theory-integration.md)
- [Metrics Interpreter Code](../services/extract/internal/processing/metrics_interpreter.go)

