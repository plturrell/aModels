# Postgres Lang Service

This directory contains the Postgres telemetry service copied from `agenticAiETH_layer4_Postgres`.

## Overview

The Postgres Lang Service provides:
- gRPC service for telemetry logging and querying
- FastAPI gateway for HTTP/REST access
- Database administration endpoints
- Operation analytics

Refer to the original [README.md](README.md) for detailed setup instructions.

## Quick Start

### gRPC Service

```bash
cd postgres
go run ./cmd/server/main.go
```

### FastAPI Gateway

```bash
cd postgres/gateway
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m gateway
```

## Integration

The unified browser gateway (`browser/gateway/`) includes an adapter for the Postgres data service, accessible at `/data/sql`. The gateway also aggregates telemetry at `/telemetry/recent`.
