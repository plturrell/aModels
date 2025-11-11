# Extract Service

This directory contains the Extract service copied from `agenticAiETH_layer4_Extract`.

## Overview

The Extract service provides:
- OCR (Optical Character Recognition) capabilities
- Schema replication from databases
- SQL exploration and normalization
- SGMI view lineage generation
- Document embedding and persistence

Refer to the original [README.md](README.md) for detailed setup instructions.

## Quick Start

```bash
cd extract
go run main.go
```

Or via Docker:

```bash
docker build -t extract-service .
docker run -p 8081:8081 extract-service
```

## Integration

The unified browser gateway (`browser/gateway/`) includes adapters for Extract:
- `/extract/ocr` - OCR extraction
- `/extract/schema-replication` - Schema replication

The service also exposes a gRPC interface on port 9090 (default) and an HTTP/JSON interface on port 8081.
