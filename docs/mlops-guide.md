# MLOps Guide for agenticAiETH Training Platform

## Overview

This guide provides comprehensive documentation for the MLOps platform that transforms the Training component into a complete machine learning operations system with CPU-based LoRA fine-tuning, end-to-end automation pipelines, distributed tracing, model versioning, and comprehensive observability.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Getting Started](#getting-started)
3. [Training Configuration](#training-configuration)
4. [Automation Pipeline](#automation-pipeline)
5. [Model Registry](#model-registry)
6. [Deployment](#deployment)
7. [Monitoring and Observability](#monitoring-and-observability)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)
10. [API Reference](#api-reference)

## Architecture Overview

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   HANA Database │    │  Training Layer │    │  LocalAI Layer │
│                 │    │                 │    │                 │
│ • Training Data │◄──►│ • LoRA Training │◄──►│ • Model Serving │
│ • Results       │    │ • CPU Optimized │    │ • Inference     │
│ • Metadata      │    │ • Automation    │    │ • Caching      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Search Layer   │    │ Orchestration   │    │  Observability │
│                 │    │                 │    │                 │
│ • RAG Retrieval │    │ • Agent Workflows│   │ • Distributed   │
│ • Vector Search │    │ • Chain Execution│   │   Tracing       │
│ • Query Process │    │ • Tool Calling  │    │ • Metrics       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Key Features

- **CPU-Based LoRA Training**: Efficient fine-tuning without GPU dependencies
- **End-to-End Automation**: Automated training, validation, and deployment
- **Model Registry**: Version control and lineage tracking
- **Distributed Tracing**: OpenTelemetry integration across all components
- **Comprehensive Monitoring**: Prometheus metrics and Grafana dashboards
- **Blue-Green Deployment**: Zero-downtime model updates
- **RAG Integration**: Retrieval-augmented generation for enhanced inference

## Getting Started

### Prerequisites

- Go 1.21 or later
- HANA database access
- LocalAI server running
- Docker (for containerization)
- Kubernetes (for orchestration)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/plturrell/agenticAiETH.git
   cd agenticAiETH/agenticAiETH_layer4_Training
   ```

2. **Install dependencies**
   ```bash
   go mod download
   go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
   ```

3. **Configure environment**
   ```bash
   cp configs/training_config.yaml.example configs/training_config.yaml
   # Edit configuration file
   ```

4. **Initialize HANA schema**
   ```bash
   go run cmd/init/main.go --config configs/training_config.yaml
   ```

5. **Start services**
   ```bash
   # Start training service
   go run cmd/training-service/main.go --config configs/training_config.yaml
   
   # Start benchmark scheduler
   go run cmd/benchmark-scheduler/main.go --config configs/automation_config.yaml
   
   # Start model deployer
   go run cmd/model-deployer/main.go --config configs/deployment_config.yaml
   ```

### Quick Start

1. **Train a model**
   ```bash
   go run cmd/training-service/main.go train \
     --model-name vaultgemma-1b \
     --dataset-name instruction-tuning \
     --epochs 3 \
     --batch-size 4
   ```

2. **Run benchmarks**
   ```bash
   go run cmd/benchmark-service/main.go run \
     --models vaultgemma-1b,phi-3.5-mini \
     --benchmarks arc,boolq,hellaswag
   ```

3. **Deploy model**
   ```bash
   go run cmd/deployment-service/main.go deploy \
     --model-path models/lora-adapters \
     --environment production
   ```

## Training Configuration

### LoRA Training Parameters

```yaml
lora:
  rank: 16                    # LoRA rank (8, 16, 32)
  alpha: 32.0                 # LoRA scaling factor
  dropout: 0.1                # Dropout rate
  target_modules:             # Modules to apply LoRA to
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
```

### CPU Optimization

```yaml
cpu_optimization:
  num_workers: 4              # Number of CPU workers
  use_simd: true             # Use SIMD instructions
  use_vectorization: true     # Use vectorization
  cache_optimization: true    # Enable cache optimization
  memory_pool_size: 1024      # Memory pool size in MB
```

### Training Hyperparameters

```yaml
training:
  learning_rate: 2e-4         # Base learning rate
  batch_size: 4               # Batch size per worker
  gradient_accumulation: 4    # Gradient accumulation steps
  max_epochs: 3               # Maximum number of epochs
  warmup_steps: 100           # Number of warmup steps
  weight_decay: 0.01          # Weight decay for regularization
```

## Automation Pipeline

### Benchmark Scheduler

The benchmark scheduler automatically runs benchmarks on a schedule:

```yaml
scheduler:
  schedule_type: "cron"
  cron_expression: "0 2 * * *"  # Daily at 2 AM
  benchmarks: ["arc", "boolq", "hellaswag", "piqa", "socialiqa", "triviaqa"]
  models: ["vaultgemma-1b", "phi-3.5-mini", "granite-4.0"]
  max_concurrency: 2
```

### Model Deployment

Automated deployment with blue-green strategy:

```yaml
deployment:
  strategy: "blue_green"
  max_instances: 2
  validation_enabled: true
  min_accuracy: 0.7
  max_latency: 2s
  auto_rollback: true
```

### CI/CD Pipeline

The GitHub Actions workflow provides:

- **Automated Testing**: Unit tests, integration tests, security scans
- **Model Training**: Conditional LoRA training on schedule
- **Benchmark Execution**: Automated benchmark runs
- **Model Validation**: Pre-deployment validation
- **Model Deployment**: Blue-green and canary deployments
- **Monitoring**: Health checks and performance monitoring

## Model Registry

### Model Versioning

```go
// Register a new model
model := &registry.ModelInfo{
    Name:        "vaultgemma-1b",
    Version:     "1.0.0",
    Description: "Fine-tuned VaultGemma model",
    Tags:        []string{"instruction-tuning", "lora"},
    Metadata: map[string]interface{}{
        "base_model": "vaultgemma-1b",
        "lora_rank":  16,
        "dataset":    "instruction-tuning",
    },
    Hyperparameters: map[string]interface{}{
        "learning_rate": 2e-4,
        "batch_size":    4,
        "epochs":        3,
    },
    TrainingMetrics: map[string]interface{}{
        "final_loss": 0.15,
        "accuracy":   0.85,
    },
}

err := registry.RegisterModel(ctx, model)
```

### Model Lineage

```go
// Get model lineage
lineage, err := registry.GetModelLineage(ctx, "vaultgemma-1b")
if err != nil {
    return err
}

// Access training runs, deployments, and relationships
for _, run := range lineage.TrainingRuns {
    fmt.Printf("Training run: %s, Status: %s\n", run.ID, run.Status)
}
```

### Model Comparison

```go
// Compare two models
comparison, err := registry.CompareModels(ctx, model1, model2)
if err != nil {
    return err
}

// Access differences and similarities
for _, diff := range comparison.Differences {
    fmt.Printf("Difference in %s: %v vs %v\n", diff.Field, diff.Value1, diff.Value2)
}
```

## Deployment

### Blue-Green Deployment

1. **Deploy new instances (green)**
2. **Validate new instances**
3. **Switch traffic to green**
4. **Stop old instances (blue)**

### Canary Deployment

1. **Deploy canary instance**
2. **Gradually shift traffic (10%, 25%, 50%, 75%, 100%)**
3. **Monitor canary health**
4. **Rollback if issues detected**

### Rolling Deployment

1. **Deploy new instances one by one**
2. **Wait for each instance to be healthy**
3. **Stop old instances**

### Deployment Configuration

```yaml
deployment:
  strategy: "blue_green"
  max_instances: 2
  traffic_shift: 100
  validation_enabled: true
  validation_timeout: 5m
  min_accuracy: 0.7
  max_latency: 2s
  auto_rollback: true
  rollback_timeout: 10m
  health_check_interval: 30s
```

## Monitoring and Observability

### Distributed Tracing

OpenTelemetry integration provides end-to-end tracing:

```go
// Start a span
ctx, span := tracer.Start(ctx, "training.operation")
defer span.End()

// Add attributes
span.SetAttributes(
    attribute.String("model.name", "vaultgemma-1b"),
    attribute.String("dataset.name", "instruction-tuning"),
    attribute.Int("epoch", 1),
)

// Add events
span.AddEvent("epoch_completed", trace.WithAttributes(
    attribute.Float64("loss", 0.15),
    attribute.Float64("accuracy", 0.85),
))
```

### Metrics Collection

Prometheus metrics are automatically collected:

- **Training Metrics**: Loss, accuracy, learning rate, CPU utilization
- **Inference Metrics**: Latency, throughput, cache hit rate
- **System Metrics**: CPU usage, memory usage, disk I/O
- **Business Metrics**: Model performance, deployment status

### Grafana Dashboards

Pre-configured dashboards provide:

- **Training Dashboard**: Loss curves, accuracy trends, hyperparameter effects
- **Inference Dashboard**: Latency, throughput, error rates
- **System Dashboard**: Resource utilization, health status
- **Business Dashboard**: Model performance, deployment metrics

### Alerting

Configure alerts for:

- **Training Failures**: Failed training runs, convergence issues
- **Performance Degradation**: Accuracy drops, latency increases
- **System Issues**: High CPU usage, memory leaks, disk space
- **Deployment Issues**: Failed deployments, health check failures

## Troubleshooting

### Common Issues

#### Training Issues

**Problem**: Training fails with out of memory error
**Solution**: 
- Reduce batch size
- Enable gradient checkpointing
- Increase memory pool size
- Use gradient accumulation

**Problem**: Training converges slowly
**Solution**:
- Adjust learning rate
- Check data quality
- Verify hyperparameters
- Enable learning rate scheduling

#### Deployment Issues

**Problem**: Deployment fails validation
**Solution**:
- Check model accuracy
- Verify latency requirements
- Review validation configuration
- Check model format compatibility

**Problem**: Traffic switching fails
**Solution**:
- Verify LocalAI connectivity
- Check load balancer configuration
- Review traffic split settings
- Monitor instance health

#### Performance Issues

**Problem**: High inference latency
**Solution**:
- Enable model caching
- Optimize batch processing
- Check network latency
- Review model size

**Problem**: Low throughput
**Solution**:
- Increase batch size
- Enable parallel processing
- Check CPU utilization
- Review memory usage

### Debugging

#### Enable Debug Logging

```yaml
logging:
  level: "DEBUG"
  format: "json"
  include_timestamp: true
  include_level: true
```

#### Enable Profiling

```yaml
environment:
  debug: true
  profile: true
  profile_path: "profiles"
```

#### Check Health Status

```bash
# Check training service health
curl http://localhost:8080/health

# Check benchmark service health
curl http://localhost:8081/health

# Check deployment service health
curl http://localhost:8082/health
```

## Best Practices

### Training

1. **Start with small models**: Begin with smaller models to validate the pipeline
2. **Use appropriate batch sizes**: Balance memory usage and training speed
3. **Monitor training metrics**: Track loss, accuracy, and convergence
4. **Validate on holdout data**: Use separate validation sets
5. **Save checkpoints**: Regular checkpointing for recovery

### Deployment

1. **Test in staging**: Always test deployments in staging first
2. **Use blue-green**: Prefer blue-green over rolling deployments
3. **Monitor closely**: Watch metrics during and after deployment
4. **Have rollback plan**: Always be prepared to rollback
5. **Validate thoroughly**: Comprehensive validation before production

### Monitoring

1. **Set up alerts**: Configure alerts for critical metrics
2. **Monitor trends**: Watch for gradual performance degradation
3. **Track business metrics**: Monitor model performance in production
4. **Regular reviews**: Review metrics and logs regularly
5. **Document issues**: Keep track of issues and resolutions

### Security

1. **Secure credentials**: Use secure credential management
2. **Network security**: Implement proper network segmentation
3. **Access control**: Limit access to production systems
4. **Audit logging**: Enable comprehensive audit logging
5. **Regular updates**: Keep dependencies and systems updated

## API Reference

### Training Service API

#### Start Training

```http
POST /v1/training/start
Content-Type: application/json

{
  "model_name": "vaultgemma-1b",
  "dataset_name": "instruction-tuning",
  "config": {
    "epochs": 3,
    "batch_size": 4,
    "learning_rate": 2e-4
  }
}
```

#### Get Training Status

```http
GET /v1/training/status/{training_id}
```

#### Stop Training

```http
POST /v1/training/stop/{training_id}
```

### Benchmark Service API

#### Run Benchmarks

```http
POST /v1/benchmarks/run
Content-Type: application/json

{
  "models": ["vaultgemma-1b", "phi-3.5-mini"],
  "benchmarks": ["arc", "boolq", "hellaswag"],
  "config": {
    "parallel": 2,
    "timeout": "30m"
  }
}
```

#### Get Benchmark Results

```http
GET /v1/benchmarks/results/{run_id}
```

### Deployment Service API

#### Deploy Model

```http
POST /v1/deployment/deploy
Content-Type: application/json

{
  "model_path": "models/lora-adapters",
  "environment": "production",
  "strategy": "blue_green",
  "config": {
    "max_instances": 2,
    "validation_enabled": true
  }
}
```

#### Get Deployment Status

```http
GET /v1/deployment/status/{deployment_id}
```

#### Rollback Deployment

```http
POST /v1/deployment/rollback/{deployment_id}
```

### Model Registry API

#### Register Model

```http
POST /v1/registry/models
Content-Type: application/json

{
  "name": "vaultgemma-1b",
  "version": "1.0.0",
  "description": "Fine-tuned VaultGemma model",
  "tags": ["instruction-tuning", "lora"],
  "metadata": {
    "base_model": "vaultgemma-1b",
    "lora_rank": 16
  }
}
```

#### List Models

```http
GET /v1/registry/models?name=vaultgemma&tags=lora
```

#### Get Model

```http
GET /v1/registry/models/{name}/{version}
```

#### Compare Models

```http
POST /v1/registry/models/compare
Content-Type: application/json

{
  "model1": {
    "name": "vaultgemma-1b",
    "version": "1.0.0"
  },
  "model2": {
    "name": "vaultgemma-1b",
    "version": "1.1.0"
  }
}
```

## Conclusion

This MLOps platform provides a comprehensive solution for machine learning operations, from training to deployment. The CPU-based approach ensures broad compatibility while the automation pipeline reduces manual overhead. The observability features provide deep insights into system performance and model behavior.

For additional support or questions, please refer to the troubleshooting section or contact the development team.
