# Quick Start Guide - Graph Service

## 🚀 Get Started in 5 Minutes

### Prerequisites
- Go 1.23 or higher
- Git
- Access to aModels mono-repository

---

## Option 1: Local Development (Recommended)

### Step 1: Enable Workspace Mode
```bash
cd /home/aModels
cp go.work.example go.work
```

### Step 2: Build the Service
```bash
cd services/graph
go build ./cmd/graph-server
```

### Step 3: Run the Server
```bash
./graph-server
```

**That's it!** The server is now running on port 8081.

---

## Option 2: Docker Build

### Step 1: Build Image
```bash
cd /home/aModels
docker build -t graph-service:latest -f services/graph/Dockerfile .
```

### Step 2: Run Container
```bash
docker run -p 8081:8081 \
  -e NEO4J_URI=bolt://localhost:7687 \
  -e EXTRACT_SERVICE_URL=http://extract-service:8081 \
  graph-service:latest
```

---

## Verify Installation

### Run Verification Script
```bash
cd /home/aModels/services/graph
./verify-dependencies.sh
```

This checks:
- ✅ Go version
- ✅ All dependencies present
- ✅ Replace directives valid
- ✅ Relative paths correct
- ✅ Build succeeds

### Manual Health Check
```bash
# After starting the server
curl http://localhost:8081/agent/catalog
```

---

## Common Operations

### Build CLI Tool
```bash
cd /home/aModels/services/graph
go build ./cmd/langgraph
./langgraph demo -input 3 -checkpoint sqlite:langgraph.dev.db -mode sync
```

### Run Tests
```bash
go test ./...
```

### Update Dependencies
```bash
go mod tidy
go mod download
```

---

## Environment Variables

### Required
```bash
export EXTRACT_SERVICE_URL=http://extract-service:8081
```

### Optional
```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=password
export MUREX_BASE_URL=http://murex-api:8080
export MUREX_API_KEY=your-api-key
export AGENTFLOW_SERVICE_URL=http://agentflow:9001
export LOCALAI_URL=http://localai:8080
export TRAINING_SERVICE_URL=http://training:8080
```

---

## API Endpoints

After starting the server, these endpoints are available:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/run` | POST | Execute graph workflow |
| `/knowledge-graph/process` | POST | Process knowledge graph |
| `/orchestration/process` | POST | Run orchestration chain |
| `/agentflow/process` | POST | Execute AgentFlow flow |
| `/deepagents/process` | POST | Deep agent analysis |
| `/gnn/query` | POST | GNN queries |
| `/gnn/hybrid-query` | POST | Hybrid KG+GNN |
| `/unified/process` | POST | Unified workflow |
| `/agent/catalog` | GET | Agent catalog |

---

## Example Request

### Simple Graph Execution
```bash
curl -X POST http://localhost:8081/run \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/path/to/data.json"
  }'
```

### Unified Workflow
```bash
curl -X POST http://localhost:8081/unified/process \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_mode": "sequential",
    "knowledge_graph_request": {
      "project_id": "test",
      "sql_queries": ["SELECT * FROM table"]
    },
    "orchestration_request": {
      "chain_name": "knowledge_graph_analyzer",
      "inputs": {
        "query": "Analyze the graph"
      }
    }
  }'
```

---

## Troubleshooting

### "module not found" Error
```bash
# Verify dependencies
cd /home/aModels/services/graph
go mod verify
go mod tidy

# Check replace directives
grep "replace" go.mod

# Ensure sibling services exist
ls -la ../catalog ../extract ../postgres
```

### Build Fails
```bash
# Clean and rebuild
go clean -cache
go mod download
go build ./cmd/graph-server
```

### Port Already in Use
```bash
# Find process using port 8081
lsof -i :8081

# Kill it if needed
kill -9 <PID>
```

---

## Next Steps

1. **Read full documentation**: `cat DEPENDENCIES.md`
2. **Explore integrations**: `cat INTEGRATION.md`
3. **Check Murex integration**: `cat MUREX_INTEGRATION.md`
4. **View examples**: Browse `python/langgraph_narrative/examples/`

---

## Getting Help

- 📖 **Dependency issues**: See `DEPENDENCIES.md`
- 🔧 **Build problems**: Run `./verify-dependencies.sh`
- 🐛 **Bugs**: Check logs and service health
- 📚 **API docs**: See `INTEGRATION.md`

---

## Performance Tips

- Use workspace mode (`go.work`) for faster builds
- Enable connection pooling for Arrow Flight
- Use Redis for distributed checkpointing
- Configure GPU orchestration for compute-intensive workflows

---

**You're now ready to use the graph service!** 🎉
