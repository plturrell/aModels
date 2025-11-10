# GPU Orchestrator Service

The GPU Orchestrator service provides intelligent GPU resource management for aModels, leveraging the unified workflow and DeepAgents logic for workload-aware allocation.

## Features

- **Dynamic GPU Allocation**: Automatically allocate GPUs based on workload requirements
- **Multi-GPU Support**: Support for parallel GPU allocation across multiple devices
- **Workload-Aware Scheduling**: Analyze workloads to determine optimal GPU allocation
- **DeepAgents Integration**: Use DeepAgents for intelligent allocation decisions
- **Unified Workflow Integration**: Coordinate GPU allocation with graph, orchestration, and AgentFlow services
- **Real-time Monitoring**: Track GPU utilization, memory, temperature, and power consumption
- **Priority-Based Scheduling**: Queue and schedule requests based on priority
- **Automatic Cleanup**: Clean up expired allocations automatically
- **Webhook Notifications**: Async callbacks when queued allocations succeed
- **Comprehensive Metrics**: Prometheus metrics for allocations, queue depth, GPU utilization, memory, temperature, and power
- **API Key Authentication**: Optional API key authentication for production security
- **YAML Configuration**: Flexible configuration via YAML files with environment variable overrides

## Architecture

- `gpu_monitor`: Discovers and monitors GPU resources via NVIDIA-SMI
- `gpu_allocator`: Manages GPU allocation and deallocation with resource reservation
- `workload_analyzer`: Analyzes workloads to determine GPU requirements
- `scheduler`: Schedules GPU allocation requests based on priority and availability
- `gpu_orchestrator`: Main orchestrator coordinating all components
- `api`: HTTP API handlers for GPU management

## API Endpoints

- `GET /healthz`: Health check
- `POST /gpu/allocate`: Allocate GPUs for a service
- `POST /gpu/release`: Release GPUs
- `GET /gpu/status`: Get GPU status
- `GET /gpu/list`: List all GPUs, allocations, and queue
- `POST /gpu/workload`: Analyze workload requirements
- `GET /metrics`: Prometheus metrics

## Configuration

The service can be configured via a YAML configuration file or environment variables.

### Configuration File

Create a `config.yaml` file (see `config.yaml.example` for a complete example):

```yaml
server:
  port: "8086"
  read_timeout: 15s
  write_timeout: 15s

services:
  deepagents_url: "http://localhost:9004"
  graph_service_url: "http://localhost:8081"

scheduler:
  queue_check_interval: 5s
  max_queue_size: 100

workload_defaults:
  inference:
    default_memory_mb: 4096
    default_priority: 7
```

### Environment Variables

Environment variables override configuration file settings:
- `CONFIG_PATH`: Path to configuration file (default: config.yaml)
- `PORT`: Service port (default: 8086)
- `DEEPAGENTS_URL`: DeepAgents service URL (default: http://localhost:9004)
- `GRAPH_SERVICE_URL`: Graph service URL (default: http://localhost:8081)

## Authentication

API key authentication can be enabled for production security:

```yaml
auth:
  enabled: true
  header_name: "X-API-Key"
  api_keys:
    "your-secret-key-1": "training-service"
    "your-secret-key-2": "inference-service"
```

When enabled, all API requests (except `/healthz` and `/metrics`) must include the API key:

```bash
curl -X POST http://localhost:8086/gpu/allocate \
  -H "X-API-Key: your-secret-key-1" \
  -H "Content-Type: application/json" \
  -d '{"service_name": "training", "workload_type": "training"}'
```

Alternative Bearer token format is also supported:

```bash
curl -X POST http://localhost:8086/gpu/allocate \
  -H "Authorization: Bearer your-secret-key-1" \
  -H "Content-Type: application/json" \
  -d '{"service_name": "training", "workload_type": "training"}'
```

## Metrics

The service exposes comprehensive Prometheus metrics at `/metrics`:

### Allocation Metrics
- `gpu_orchestrator_allocations_total` - Total allocations by service and status
- `gpu_orchestrator_allocations_active` - Currently active allocations
- `gpu_allocator_gpus_allocated_total` - Total number of allocated GPUs

### Queue Metrics
- `gpu_orchestrator_queue_depth` - Number of requests waiting in queue
- `gpu_scheduler_queue_wait_seconds` - Histogram of queue wait times
- `gpu_scheduler_scheduled_total` - Total scheduled requests by status

### GPU Resource Metrics (per GPU)
- `gpu_orchestrator_gpu_utilization_percent` - GPU utilization percentage
- `gpu_orchestrator_gpu_memory_used_mb` - GPU memory used in MB
- `gpu_orchestrator_gpu_memory_total_mb` - GPU total memory in MB
- `gpu_orchestrator_gpu_temperature_celsius` - GPU temperature
- `gpu_orchestrator_gpu_power_draw_watts` - GPU power draw in watts

### System Metrics
- `gpu_orchestrator_gpus_total` - Total number of GPUs
- `gpu_orchestrator_gpus_available` - Number of available GPUs
- `gpu_orchestrator_http_request_duration_seconds` - HTTP request duration
- `gpu_orchestrator_http_requests_total` - Total HTTP requests

## Usage

### Allocate GPUs

```bash
curl -X POST http://localhost:8086/gpu/allocate \
  -H "Content-Type: application/json" \
  -d '{
    "service_name": "training",
    "workload_type": "training",
    "workload_data": {
      "model_size": "large",
      "batch_size": 128
    }
  }'
```

### Release GPUs

```bash
curl -X POST http://localhost:8086/gpu/release \
  -H "Content-Type: application/json" \
  -d '{
    "allocation_id": "training-1234567890"
  }'
```

### Check GPU Status

```bash
curl http://localhost:8086/gpu/status
```

### Allocate with Webhook Callback

When GPUs are not immediately available, provide a webhook URL to receive a notification when allocation succeeds:

```bash
curl -X POST http://localhost:8086/gpu/allocate \
  -H "Content-Type: application/json" \
  -d '{
    "service_name": "training",
    "workload_type": "training",
    "webhook_url": "http://your-service:8080/gpu-allocated",
    "workload_data": {
      "model_size": "large"
    }
  }'
```

The webhook will receive a POST request with the following payload:
```json
{
  "status": "allocated",
  "request_id": "training-1234567890",
  "allocation": {
    "id": "training-1234567890",
    "service_name": "training",
    "gpu_ids": [0, 1],
    "allocated_at": "2024-11-10T08:30:00Z",
    "priority": 5
  },
  "submitted_at": "2024-11-10T08:25:00Z",
  "allocated_at": "2024-11-10T08:30:00Z",
  "wait_time_seconds": 300
}
```

## Workload Types

Supported workload types:
- `training`: Model training workloads
- `inference`: Model inference workloads
- `embedding`: Embedding generation workloads
- `ocr`: OCR processing workloads
- `graph_processing`: Graph processing workloads
- `generic`: Generic workloads (default)

## Requirements

- NVIDIA GPU with nvidia-smi installed
- Go 1.23.0 or later
- Docker (for containerized deployment)

## Development

```bash
cd services/gpu-orchestrator
go mod tidy
go run main.go
```

