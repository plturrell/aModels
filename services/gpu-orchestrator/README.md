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

Environment variables:
- `PORT`: Service port (default: 8086)
- `DEEPAGENTS_URL`: DeepAgents service URL (default: http://localhost:9004)
- `GRAPH_SERVICE_URL`: Graph service URL (default: http://localhost:8081)

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

