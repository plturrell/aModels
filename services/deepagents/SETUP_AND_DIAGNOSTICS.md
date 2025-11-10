# DeepAgents Service - Setup and Diagnostics

## Why DeepAgents Might Appear "Empty"

The DeepAgents service is a **Python FastAPI service** that provides AI capabilities, but it needs to be:

1. **Running** (as a service/container)
2. **Configured** (with environment variables)
3. **Enabled** (AI features are opt-in)

## Current Status Check

### 1. Is the Service Running?

```bash
# Check if service is running
curl http://localhost:9004/healthz

# Expected response:
# {"status":"ok","service":"deepagents","agent_initialized":true}
```

### 2. Are AI Features Enabled?

The catalog service AI features are **disabled by default**. To enable:

```bash
# Enable AI features in catalog service
export CATALOG_AI_DEDUPLICATION_ENABLED=true
export CATALOG_AI_VALIDATION_ENABLED=true
export CATALOG_AI_RESEARCH_ENABLED=true

# Enable AI enrichment in extract service
export EXTRACT_AI_ENRICHMENT_ENABLED=true
```

### 3. Is DeepAgents Service Configured?

The service requires these environment variables:

```bash
# Required Service URLs
EXTRACT_SERVICE_URL=http://extract-service:19080
AGENTFLOW_SERVICE_URL=http://agentflow-service:9001
GRAPH_SERVICE_URL=http://graph-service:8081

# Required: At least ONE LLM provider
LOCALAI_URL=http://localai:8081  # Recommended (local, no API keys)
# OR
ANTHROPIC_API_KEY=your_key        # Alternative
# OR
OPENAI_API_KEY=your_key          # Alternative

# Service Port
DEEPAGENTS_PORT=9004
```

## What DeepAgents Should Be Doing

### When Enabled and Running:

1. **Catalog Service Integration**:
   - **Deduplication**: Analyzes candidate data elements for duplicates
   - **Validation**: Validates data element definitions against ISO 11179
   - **Research**: Finds similar elements using Open Deep Research

2. **Extract Service Integration**:
   - **Metadata Enrichment**: Enhances extracted node metadata with AI insights

3. **Other Integrations**:
   - Knowledge graph analysis
   - Pipeline analysis
   - Data quality assessment
   - AgentFlow flow generation

## How to Start DeepAgents

### Option 1: Docker Compose (Recommended)

```bash
cd infrastructure/docker
docker-compose up deepagents
```

### Option 2: Manual Start

```bash
cd services/deepagents
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 9004
```

### Option 3: Check Docker Container

```bash
# Check if container is running
docker ps | grep deepagents

# Check logs
docker logs deepagents-service

# Start if stopped
docker start deepagents-service
```

## Diagnostic Commands

### Test DeepAgents Health

```bash
curl http://localhost:9004/healthz
```

### Test DeepAgents Invocation

```bash
curl -X POST http://localhost:9004/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello, can you help me?"}
    ]
  }'
```

### Check Agent Info

```bash
curl http://localhost:9004/agent/info
```

## Common Issues

### Issue 1: Service Not Running

**Symptoms**: 
- Health check fails
- Catalog AI features don't work
- Connection refused errors

**Solution**:
```bash
# Start the service
docker-compose up -d deepagents

# Or manually
cd services/deepagents && uvicorn main:app --host 0.0.0.0 --port 9004
```

### Issue 2: Configuration Missing

**Symptoms**:
- Service starts but agent_initialized=false
- Validation errors on startup

**Solution**:
```bash
# Set required environment variables
export EXTRACT_SERVICE_URL=http://extract-service:19080
export AGENTFLOW_SERVICE_URL=http://agentflow-service:9001
export GRAPH_SERVICE_URL=http://graph-service:8081
export LOCALAI_URL=http://localai:8081
```

### Issue 3: AI Features Disabled

**Symptoms**:
- Catalog service doesn't use AI
- No deduplication/validation happening
- Logs show "DeepAgents client disabled"

**Solution**:
```bash
# Enable AI features
export CATALOG_AI_DEDUPLICATION_ENABLED=true
export CATALOG_AI_VALIDATION_ENABLED=true
export CATALOG_AI_RESEARCH_ENABLED=true

# Restart catalog service
```

### Issue 4: LLM Provider Not Available

**Symptoms**:
- Agent fails to initialize
- "Failed to connect to LocalAI" errors

**Solution**:
```bash
# Ensure LocalAI is running
curl http://localai:8081/v1/models

# Or use alternative LLM provider
export ANTHROPIC_API_KEY=your_key
# OR
export OPENAI_API_KEY=your_key
```

## Expected Behavior When Working

### Catalog Service with AI Enabled:

1. **Bulk Registration**:
   ```bash
   POST /catalog/data-elements/bulk
   ```
   - Checks for duplicates using DeepAgents
   - Validates definitions using DeepAgents
   - Researches similar elements
   - Response includes `ai_suggestions`, `duplicates_detected`, `research_findings`

2. **Logs Should Show**:
   ```
   Catalog DeepAgents client enabled (URL: http://deepagents-service:9004)
   DeepAgents deduplication request...
   DeepAgents validation request...
   ```

### Extract Service with AI Enabled:

1. **Data Extraction**:
   - Metadata enrichment attempted for each node
   - Logs show "AI metadata enrichment enabled"
   - Elements marked with `ai_enrichment_attempted=true`

## Quick Start Checklist

- [ ] DeepAgents service is running (`curl http://localhost:9004/healthz`)
- [ ] Required environment variables are set
- [ ] At least one LLM provider is configured (LocalAI/Anthropic/OpenAI)
- [ ] AI features are enabled (`CATALOG_AI_*_ENABLED=true`)
- [ ] Catalog service can reach DeepAgents (`DEEPAGENTS_URL` is correct)
- [ ] Logs show "DeepAgents client enabled"

## Next Steps

1. **Start the service**: `docker-compose up -d deepagents`
2. **Enable AI features**: Set environment variables
3. **Test**: Make a bulk registration request and check for AI suggestions
4. **Monitor logs**: Check both catalog and deepagents service logs

