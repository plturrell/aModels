# Security Configuration Guide

This document describes how to configure authentication and rate limiting for the search service.

## Authentication

### Python Service

Authentication is controlled via environment variables:

- `AUTH_ENABLED`: Set to `true` to enable authentication (default: `false`)
- `API_KEYS`: Comma-separated list of valid API keys

**Example:**
```bash
export AUTH_ENABLED=true
export API_KEYS="key1,key2,key3"
```

**Usage:**
- API keys can be provided via `X-API-Key` header
- Or via `Authorization: Bearer <key>` header

**Example request:**
```bash
curl -H "X-API-Key: your-api-key" \
     -H "Content-Type: application/json" \
     -d '{"query": "test", "top_k": 5}' \
     http://localhost:8081/v1/search
```

### Go Service

Same configuration as Python service:

- `AUTH_ENABLED`: Set to `true` to enable authentication (default: `false`)
- `API_KEYS`: Comma-separated list of valid API keys
- `AUTH_HEADER_NAME`: Header name for API key (default: `X-API-Key`)

**Example:**
```bash
export AUTH_ENABLED=true
export API_KEYS="key1,key2,key3"
```

## Rate Limiting

### Python Service

Rate limiting is controlled via environment variables:

- `RATE_LIMIT_ENABLED`: Set to `true` to enable rate limiting (default: `true`)
- `RATE_LIMIT_PER_MINUTE`: Maximum requests per minute per client (default: `60`)

**Example:**
```bash
export RATE_LIMIT_ENABLED=true
export RATE_LIMIT_PER_MINUTE=100
```

**Note:** The Python service uses in-memory rate limiting. For production with multiple instances, consider using Redis-based rate limiting.

### Go Service

Same configuration as Python service:

- `RATE_LIMIT_ENABLED`: Set to `true` to enable rate limiting (default: `true`)
- `RATE_LIMIT_PER_MINUTE`: Maximum requests per minute per client (default: `60`)

## Docker Compose Configuration

You can configure authentication and rate limiting via environment variables in `docker-compose.yml`:

```yaml
services:
  python-service:
    environment:
      - AUTH_ENABLED=true
      - API_KEYS=your-api-key-1,your-api-key-2
      - RATE_LIMIT_ENABLED=true
      - RATE_LIMIT_PER_MINUTE=60
```

Or use a `.env` file:

```bash
# .env
AUTH_ENABLED=true
API_KEYS=your-api-key-1,your-api-key-2
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=60
```

Then run:
```bash
docker compose --env-file .env up
```

## Security Best Practices

1. **Use Strong API Keys**: Generate cryptographically secure random keys:
   ```python
   import secrets
   api_key = secrets.token_urlsafe(32)
   ```

2. **Rotate Keys Regularly**: Change API keys periodically and update all clients.

3. **Use Secrets Management**: In production, use:
   - HashiCorp Vault
   - AWS Secrets Manager
   - Kubernetes Secrets
   - Environment variables (for development only)

4. **Enable Authentication in Production**: Always set `AUTH_ENABLED=true` in production.

5. **Configure Appropriate Rate Limits**: Adjust `RATE_LIMIT_PER_MINUTE` based on your use case.

6. **Monitor Rate Limit Violations**: Log and alert on 429 responses.

7. **Use HTTPS**: Always use TLS/SSL in production. Configure reverse proxy (nginx, traefik) with SSL certificates.

## Health Endpoints

Health check endpoints (`/health`) are **always** accessible without authentication to allow monitoring systems to check service status.

## Production Deployment Checklist

- [ ] Enable authentication (`AUTH_ENABLED=true`)
- [ ] Configure strong API keys
- [ ] Enable rate limiting
- [ ] Set appropriate rate limits
- [ ] Use secrets management for API keys
- [ ] Configure HTTPS/TLS
- [ ] Set up monitoring and alerting
- [ ] Review and test security configuration
- [ ] Document API key distribution process
- [ ] Set up key rotation schedule

