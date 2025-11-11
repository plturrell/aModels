# Security Guidelines

## Overview

The DMS service implements multiple security layers to protect sensitive document data and prevent unauthorized access.

## Authentication

### JWT Tokens

JWT (JSON Web Tokens) provide stateless authentication:

```python
from app.core.auth import create_access_token

# Create token with user info
token = create_access_token(
    data={"sub": "user_id", "username": "john@example.com"},
    expires_delta=timedelta(hours=24)
)
```

**Configuration:**
- `DMS_JWT_SECRET`: Secret key for signing tokens (MUST be changed in production)
- `DMS_JWT_ALGORITHM`: Algorithm for token signing (default: HS256)

**Best Practices:**
1. Use strong, random secrets (minimum 32 characters)
2. Rotate secrets regularly
3. Store secrets in secure vault (e.g., AWS Secrets Manager, HashiCorp Vault)
4. Never commit secrets to version control

### API Keys

API keys provide simple authentication for service-to-service communication:

**Configuration:**
- `DMS_API_KEYS`: Comma-separated list of valid API keys

**Best Practices:**
1. Generate cryptographically random keys (use `openssl rand -hex 32`)
2. Use separate keys per client/service
3. Rotate keys quarterly
4. Revoke compromised keys immediately

### Protected Endpoints

By default, all endpoints are accessible. To enforce authentication:

```bash
export DMS_REQUIRE_AUTH=true
```

To protect specific endpoints only, use the `verify_token` dependency:

```python
from app.core.auth import verify_token
from fastapi import Depends

@router.post("/admin/action")
async def admin_action(user: dict = Depends(verify_token)):
    # Only authenticated users can access
    pass
```

## Database Security

### Password Requirements

The service validates database credentials and rejects weak passwords:

**Rejected passwords:**
- `postgres`, `password`, `admin`, `root`, `123456`
- Any password under 8 characters (recommended: 16+ characters)

**Password generation:**
```bash
# Generate secure password
openssl rand -base64 24
```

### Connection Security

**Development:**
```bash
postgresql+psycopg://user:password@localhost:5432/dms
```

**Production (with SSL):**
```bash
postgresql+psycopg://user:password@db.example.com:5432/dms?sslmode=require
```

Always use SSL/TLS for database connections in production.

## Docker Security

### Secure docker-compose Setup

1. **Never use default credentials:**
```bash
# ❌ BAD
POSTGRES_PASSWORD=postgres

# ✅ GOOD
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}  # From .env file
```

2. **Use .env file with restricted permissions:**
```bash
cp .env.example .env
chmod 600 .env  # Owner read/write only
# Add .env to .gitignore
```

3. **Example secure .env:**
```bash
POSTGRES_PASSWORD=$(openssl rand -base64 24)
NEO4J_AUTH=neo4j/$(openssl rand -base64 24)
DMS_JWT_SECRET=$(openssl rand -hex 32)
DMS_API_KEYS=$(openssl rand -hex 32),$(openssl rand -hex 32)
```

### Container Hardening

1. **Run as non-root user** (add to Dockerfile):
```dockerfile
RUN addgroup --system --gid 1001 dms && \
    adduser --system --uid 1001 --ingroup dms dms
USER dms
```

2. **Read-only root filesystem:**
```yaml
services:
  dms:
    read_only: true
    tmpfs:
      - /tmp
      - /var/run
```

3. **Drop capabilities:**
```yaml
services:
  dms:
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
```

## Network Security

### HTTPS/TLS

Always use HTTPS in production. Use a reverse proxy (nginx, Traefik) for TLS termination:

```nginx
server {
    listen 443 ssl http2;
    server_name dms.example.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    
    location / {
        proxy_pass http://dms:8080;
        proxy_set_header X-Request-ID $request_id;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### CORS Configuration

For web clients, configure CORS in `main.py`:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.example.com"],  # Specific origins only
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "X-API-Key"],
)
```

## File Storage Security

### Storage Path Validation

The service validates storage paths to prevent directory traversal:

```python
# Safe: Uses UUID-based paths
storage_dir = settings.storage_root / document_id  # /data/documents/uuid
```

**Never** accept user-provided paths directly.

### File Type Validation

Implement file type validation:

```python
ALLOWED_EXTENSIONS = {".pdf", ".doc", ".docx", ".txt", ".png", ".jpg"}
ALLOWED_MIME_TYPES = {"application/pdf", "image/png", "text/plain"}

# Check extension
if Path(file.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
    raise HTTPException(400, "File type not allowed")

# Check MIME type
import magic
mime = magic.from_buffer(await file.read(2048), mime=True)
if mime not in ALLOWED_MIME_TYPES:
    raise HTTPException(400, "Invalid file content")
```

### File Size Limits

Configure max upload size:

```python
# In main.py
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"],
    max_body_size=100 * 1024 * 1024  # 100MB
)
```

## Monitoring & Auditing

### Security Logging

Enable audit logging for sensitive operations:

```python
logger.info(
    "Document uploaded: %s by user %s",
    document_id,
    user.get("username"),
    extra={
        "event": "document_upload",
        "user_id": user.get("sub"),
        "document_id": document_id,
        "ip_address": request.client.host,
    }
)
```

### Failed Authentication Tracking

Monitor failed authentication attempts:

```python
from collections import defaultdict
from datetime import datetime, timedelta

failed_attempts = defaultdict(list)

def check_rate_limit(ip_address: str) -> bool:
    """Rate limit failed authentication attempts."""
    now = datetime.utcnow()
    recent = [t for t in failed_attempts[ip_address] 
              if now - t < timedelta(minutes=15)]
    failed_attempts[ip_address] = recent
    
    if len(recent) >= 5:
        logger.warning("Rate limit exceeded for IP: %s", ip_address)
        return False
    
    return True
```

## Incident Response

### Compromised Credentials

If credentials are compromised:

1. **Rotate immediately:**
```bash
# Update secrets
export DMS_JWT_SECRET=$(openssl rand -hex 32)
export DMS_API_KEYS=$(openssl rand -hex 32)

# Restart service
docker compose restart dms
```

2. **Invalidate all tokens:**
   - JWT tokens will become invalid with new secret
   - Revoke old API keys from `DMS_API_KEYS`

3. **Audit access logs:**
```bash
# Search for suspicious activity
grep "401\|403" /var/log/dms/access.log
```

### Data Breach

If data breach is suspected:

1. Isolate affected systems
2. Preserve logs for forensic analysis
3. Notify security team and stakeholders
4. Review access logs for unauthorized access
5. Implement additional access controls

## Security Checklist

### Development
- [ ] Use `.env.example` for documentation only
- [ ] Add `.env` to `.gitignore`
- [ ] Never commit secrets to Git
- [ ] Use strong passwords locally

### Staging
- [ ] Enable authentication (`DMS_REQUIRE_AUTH=true`)
- [ ] Use secure passwords (16+ characters)
- [ ] Enable HTTPS
- [ ] Review CORS configuration
- [ ] Test authentication flows

### Production
- [ ] Use secrets management service (Vault, AWS Secrets Manager)
- [ ] Enable all authentication mechanisms
- [ ] Use TLS 1.2+ for all connections
- [ ] Implement rate limiting
- [ ] Enable audit logging
- [ ] Set up monitoring and alerts
- [ ] Regular security updates
- [ ] Conduct security audit
- [ ] Implement backup encryption
- [ ] Document incident response plan

## Reporting Security Issues

If you discover a security vulnerability, please email security@example.com with:

1. Description of the vulnerability
2. Steps to reproduce
3. Potential impact
4. Suggested fix (if any)

**Do not** open public issues for security vulnerabilities.
