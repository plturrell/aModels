# Security Guidelines

## Overview

This document outlines security considerations and best practices for deploying and using the Postgres Lang Service.

## Current Security Posture

### ⚠️ Known Security Risks

1. **Database Admin Endpoints (`/db/*`)**
   - The `/db/query` endpoint allows arbitrary SQL execution
   - Even with `allow_mutations=false`, SELECT queries can expose sensitive data
   - **Recommendation**: Disable in production or require authentication

2. **No Authentication/Authorization**
   - All endpoints are publicly accessible
   - No API key or JWT validation
   - **Recommendation**: Implement authentication before production deployment

3. **Database Credentials**
   - DSN stored in environment variables (visible in process listing)
   - **Recommendation**: Use secrets management (Vault, AWS Secrets Manager, K8s Secrets)

## Security Best Practices

### 1. Authentication Setup (Recommended)

#### API Key Authentication Example

Add to `gateway/app.py`:

```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required")
    
    # Validate against stored keys (use secrets management)
    valid_keys = os.getenv("API_KEYS", "").split(",")
    if api_key not in valid_keys:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    return api_key

# Add to endpoints:
@app.post("/operations", dependencies=[Depends(verify_api_key)])
def log_operation(...):
    ...
```

#### JWT Authentication Example

```python
from fastapi import Depends
from fastapi.security import HTTPBearer
import jwt

security = HTTPBearer()

def verify_jwt(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(
            credentials.credentials,
            os.getenv("JWT_SECRET"),
            algorithms=["HS256"]
        )
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/operations", dependencies=[Depends(verify_jwt)])
def log_operation(...):
    ...
```

### 2. Database Admin Endpoints

**Production Configuration**:

```bash
# Disable database admin endpoints in production
unset POSTGRES_LANG_DB_DSN

# Or use read-only mode
export POSTGRES_DB_ALLOW_MUTATIONS=false
export POSTGRES_DB_DEFAULT_LIMIT=100
```

**Alternative**: Remove `/db/*` endpoints entirely from production builds.

### 3. Rate Limiting

Install slowapi:
```bash
pip install slowapi
```

Add to `gateway/app.py`:
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/operations")
@limiter.limit("100/minute")
def log_operation(request: Request, ...):
    ...
```

### 4. Secrets Management

#### Using HashiCorp Vault

```go
import (
    vault "github.com/hashicorp/vault/api"
)

func getDSNFromVault() (string, error) {
    client, err := vault.NewClient(vault.DefaultConfig())
    if err != nil {
        return "", err
    }
    
    secret, err := client.Logical().Read("secret/data/postgres/dsn")
    if err != nil {
        return "", err
    }
    
    return secret.Data["data"].(map[string]interface{})["dsn"].(string), nil
}
```

#### Using Environment Variables (Current)

```bash
# Use restricted permissions
chmod 600 /etc/postgres-service/.env

# Never commit .env files
echo ".env" >> .gitignore
```

### 5. TLS/SSL Configuration

#### gRPC Server with TLS

```go
import (
    "google.golang.org/grpc/credentials"
)

func main() {
    creds, err := credentials.NewServerTLSFromFile("server.crt", "server.key")
    if err != nil {
        log.Fatal(err)
    }
    
    grpcServer := grpc.NewServer(
        grpc.Creds(creds),
        grpc.UnaryInterceptor(loggingInterceptor),
    )
    // ...
}
```

#### FastAPI Gateway with HTTPS

```bash
# Using uvicorn
uvicorn gateway:app \
    --ssl-keyfile=/path/to/key.pem \
    --ssl-certfile=/path/to/cert.pem
```

### 6. Database Connection Security

```bash
# Always use SSL mode in production
POSTGRES_DSN="postgres://user:pass@host:5432/db?sslmode=require"

# For stricter validation
POSTGRES_DSN="postgres://user:pass@host:5432/db?sslmode=verify-full&sslrootcert=/path/to/ca.crt"
```

### 7. Input Validation

Current implementation uses:
- Protobuf validation for gRPC
- Pydantic models for FastAPI

**Additional validation** can be added:

```python
from pydantic import validator, Field

class OperationLogPayload(BaseModel):
    library_type: str = Field(..., max_length=100, pattern="^[a-z0-9_]+$")
    operation: str = Field(..., max_length=200)
    
    @validator('library_type')
    def validate_library_type(cls, v):
        allowed = ['langchain', 'langgraph', 'llamaindex', 'crewai']
        if v not in allowed:
            raise ValueError(f"library_type must be one of {allowed}")
        return v
```

### 8. Audit Logging

Add audit logs for sensitive operations:

```go
func auditLog(ctx context.Context, action string, user string, details map[string]interface{}) {
    log.Info().
        Str("action", action).
        Str("user", user).
        Interface("details", details).
        Msg("audit event")
}

// In cleanup handler
auditLog(ctx, "cleanup_operations", getUserFromContext(ctx), map[string]interface{}{
    "older_than": olderThan,
    "deleted": result.Deleted,
})
```

## Security Checklist

Before deploying to production:

- [ ] Implement authentication (API keys, JWT, or OAuth)
- [ ] Disable or protect database admin endpoints
- [ ] Enable TLS for gRPC and HTTPS for gateway
- [ ] Use secrets management for database credentials
- [ ] Configure database connection with SSL mode
- [ ] Add rate limiting to prevent abuse
- [ ] Enable audit logging for all write operations
- [ ] Review and restrict CORS origins
- [ ] Set up monitoring and alerting
- [ ] Perform security scanning (grype, trivy)
- [ ] Regular dependency updates
- [ ] Implement network policies (K8s) or firewall rules

## Vulnerability Reporting

If you discover a security vulnerability, please email: security@example.com

Do not create public GitHub issues for security vulnerabilities.

## Compliance Considerations

### GDPR/Privacy

- User ID hashing: Currently uses `user_id_hash` field
- Implement data retention policies using cleanup operations
- Provide data export/deletion capabilities
- Consider field-level encryption for sensitive data

### SOC 2

- Enable audit logging for all operations
- Implement access controls and authentication
- Regular security assessments
- Incident response procedures

## Regular Security Maintenance

1. **Weekly**: Review security logs and audit trails
2. **Monthly**: Update dependencies (`go get -u`, `pip install -U`)
3. **Quarterly**: Security scanning and vulnerability assessment
4. **Annually**: Penetration testing and security audit

## References

- [OWASP API Security Top 10](https://owasp.org/www-project-api-security/)
- [gRPC Security Guide](https://grpc.io/docs/guides/auth/)
- [FastAPI Security Best Practices](https://fastapi.tiangolo.com/tutorial/security/)
