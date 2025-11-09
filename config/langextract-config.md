# langextract-api Configuration for Local Models Only

## Overview

langextract-api is an optional service for document extraction. To ensure it uses only local models, configure it to use LocalAI or disable it entirely.

## Configuration Options

### Option 1: Disable langextract-api (Recommended for LocalAI-only operation)

Set the environment variables to empty in docker-compose.yml:

```yaml
environment:
  - LANGEXTRACT_API_URL=
  - LANGEXTRACT_API_KEY=
```

This completely disables langextract-api. The extract service will handle extraction using LocalAI directly.

### Option 2: Configure langextract-api to use LocalAI

If langextract-api supports OpenAI-compatible APIs (which LocalAI provides), configure it:

```yaml
environment:
  - LANGEXTRACT_API_URL=http://localai:8081/v1
  - LANGEXTRACT_API_KEY=not-needed
```

**Note**: langextract-api may need to be configured to use the "general" domain or a specific extraction domain configured in LocalAI's `domains.json`.

### Option 3: Use langextract with Ollama (if available locally)

If you have Ollama running locally with models:

```yaml
environment:
  - LANGEXTRACT_API_URL=http://ollama:11434
  - LANGEXTRACT_API_KEY=
```

## Current Configuration

The extract service in `docker-compose.yml` is configured with:

```yaml
- LANGEXTRACT_API_URL=${LANGEXTRACT_API_URL:-}
- LANGEXTRACT_API_KEY=${LANGEXTRACT_API_KEY:-}
```

This means:
- By default (empty), langextract-api is **disabled**
- To enable, set `LANGEXTRACT_API_URL` environment variable before running docker-compose
- If enabled, ensure it points to LocalAI or another local service

## Verification

To verify langextract-api is not using external models:

1. Check extract service logs for langextract API calls
2. Ensure no external API endpoints are being called
3. If `LANGEXTRACT_API_URL` is empty, langextract-api is disabled and extraction uses LocalAI directly

## Recommendation

For pure LocalAI-only operation, **keep langextract-api disabled** (leave `LANGEXTRACT_API_URL` empty). The extract service can handle extraction tasks using LocalAI's configured domains.

