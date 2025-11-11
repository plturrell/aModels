# AgentFlow

This directory contains the AgentFlow service copied from `agenticAiETH_layer4_AgentFlow`.

## Overview

AgentFlow acts as a bridge between the agenticAiETH project and a managed Langflow deployment. It provides:
- Go CLI for syncing and running flows
- FastAPI service for HTTP interface
- Flow catalog management
- SGMI view lineage integration

Refer to the original [README.md](README.md) for detailed setup and usage instructions.

## Quick Start

### FastAPI Service

```bash
cd agentflow/service
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn service.main:app --reload
```

### Go CLI

```bash
cd agentflow
go run ./cmd/flow-run --probe
go run ./cmd/flow-run --flow-id processes/sample_reconciliation --input 'Reconcile ledger 123'
```

## Integration

The unified browser gateway (`browser/gateway/`) includes an adapter for AgentFlow, accessible at `/agentflow/run`.
