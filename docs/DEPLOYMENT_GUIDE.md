# aModels Deployment Guide - SOC 2 Compliant

This guide covers deploying aModels with all security requirements for SOC 2 Type II compliance.

## Prerequisites

- Docker and Docker Compose installed
- Access to secrets management (for production)
- SSL certificates (for HTTPS in production)

## Quick Start

### 1. Generate Secure Secrets

```bash
# Generate all required secrets
./scripts/generate-secrets.sh > .env.secrets

# Review and copy values to your .env file
cat .env.secrets
```

### 2. Configure Environment Variables

Copy the example environment file and fill in values:

```bash
cp .env.example .env
# Edit .env with your actual values
```

**Required Environment Variables:**

#### JWT Authentication (Catalog Service)
```bash
JWT_SECRET_KEY=<generate-with-openssl-rand-base64-32>
JWT_TOKEN_EXPIRY=15m
JWT_REFRESH_EXPIRY=168h
```

#### Neo4j Configuration
```bash
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<strong-password>
```

#### PostgreSQL Configuration
```bash
POSTGRES_USER=postgres
POSTGRES_PASSWORD=<strong-password>
POSTGRES_DB=amodels
```

#### CORS Configuration (Production)
```bash
ENVIRONMENT=production
LOCALAI_CORS_ALLOW_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
```

### 3. Deploy with Docker Compose

```bash
cd infrastructure/docker
docker-compose -f compose.yml up -d
```

## Production Deployment Checklist

### Security Configuration

- [ ] **JWT Secret Key**: Generate and set `JWT_SECRET_KEY` (32+ bytes, base64 encoded)
- [ ] **Database Passwords**: Set strong passwords for all databases (Neo4j, PostgreSQL)
- [ ] **Environment**: Set `ENVIRONMENT=production` to enforce security checks
- [ ] **CORS Origins**: Configure specific allowed origins (no wildcards)
- [ ] **HTTPS**: Configure reverse proxy with SSL/TLS certificates
- [ ] **Secrets Management**: Use secrets manager (AWS Secrets Manager, HashiCorp Vault, etc.)

### Service-Specific Requirements

#### Catalog Service
- [ ] `JWT_SECRET_KEY` is set and secure
- [ ] `NEO4J_USERNAME` and `NEO4J_PASSWORD` are set
- [ ] `CATALOG_DATABASE_URL` uses secure credentials
- [ ] Authentication is enabled (mandatory)

#### Graph Service
- [ ] `NEO4J_USERNAME` and `NEO4J_PASSWORD` are set
- [ ] No default passwords in configuration

#### DMS Service
- [ ] `DMS_POSTGRES_DSN` is set with secure credentials
- [ ] `DMS_NEO4J_USER` and `DMS_NEO4J_PASSWORD` are set
- [ ] Password validation passes (no common defaults)

#### LocalAI Service
- [ ] `CORS_ALLOW_ORIGINS` is set with specific origins
- [ ] `ENVIRONMENT=production` is set

## Testing Authentication

### 1. Generate a JWT Token

```bash
# Using the catalog service API (requires authentication endpoint)
curl -X POST http://localhost:8084/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "password"}'
```

### 2. Use Token for Protected Endpoints

```bash
# Get token from login response
TOKEN="your-jwt-token-here"

# Make authenticated request
curl -X POST http://localhost:8084/catalog/data-elements \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "test", "type": "string"}'
```

### 3. Verify Token Expiration

```bash
# Wait 15+ minutes, then try to use the token
# Should receive 401 Unauthorized
curl -X POST http://localhost:8084/catalog/data-elements \
  -H "Authorization: Bearer $TOKEN"
```

## Verifying Security Configuration

### Check for Default Passwords

```bash
# Verify Neo4j password is not default
docker-compose exec catalog env | grep NEO4J_PASSWORD
# Should NOT show "password" or "neo4j"

# Verify PostgreSQL password is not default
docker-compose exec catalog env | grep POSTGRES_PASSWORD
# Should NOT show "postgres"
```

### Verify CORS Configuration

```bash
# Check CORS is configured (production)
docker-compose exec catalog env | grep ENVIRONMENT
# Should show: ENVIRONMENT=production

# Check CORS origins are set
docker-compose exec localai env | grep CORS_ALLOW_ORIGINS
# Should show specific origins, not "*"
```

### Verify JWT Configuration

```bash
# Check JWT secret is set
docker-compose exec catalog env | grep JWT_SECRET_KEY
# Should show a long base64 string

# Verify service starts successfully
docker-compose logs catalog | grep "JWT authentication middleware initialized"
```

## Troubleshooting

### Service Fails to Start

**Error: "JWT_SECRET_KEY environment variable is required"**
- Solution: Set `JWT_SECRET_KEY` in your `.env` file
- Generate with: `openssl rand -base64 32`

**Error: "NEO4J_PASSWORD environment variable is required"**
- Solution: Set `NEO4J_PASSWORD` in your `.env` file
- Must not be empty or use default values

**Error: "CORS_ALLOW_ORIGINS is not set"**
- Solution: Set `LOCALAI_CORS_ALLOW_ORIGINS` with specific origins
- In production, wildcard "*" is not allowed

### Authentication Issues

**401 Unauthorized on protected endpoints**
- Verify JWT token is valid and not expired
- Check token format: `Authorization: Bearer <token>`
- Verify `JWT_SECRET_KEY` matches between token generation and validation

**Token expires too quickly**
- Adjust `JWT_TOKEN_EXPIRY` (default: 15m)
- Use refresh tokens for longer sessions

## Secrets Management

### Using AWS Secrets Manager

```bash
# Retrieve secrets
aws secretsmanager get-secret-value --secret-id amodels/production \
  --query SecretString --output text | jq -r '.JWT_SECRET_KEY'
```

### Using HashiCorp Vault

```bash
# Retrieve secrets
vault kv get -field=JWT_SECRET_KEY secret/amodels/production
```

### Using Kubernetes Secrets

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: amodels-secrets
type: Opaque
stringData:
  JWT_SECRET_KEY: <base64-encoded-key>
  NEO4J_PASSWORD: <password>
  POSTGRES_PASSWORD: <password>
```

## Monitoring and Auditing

### Enable Audit Logging

All authentication events are logged. Check logs:

```bash
# View authentication audit logs
docker-compose logs catalog | grep AUDIT

# Example output:
# [AUDIT] token_generated user_id=user123 email=user@example.com
# [AUDIT] authenticated user_id=user123 ip=192.168.1.1 path=/catalog/data-elements
# [AUDIT] token_validation_failed error=token has expired
```

### Security Monitoring

Monitor for:
- Failed authentication attempts
- Token expiration events
- CORS violations
- SQL injection attempts (check error logs)

## Rollback Procedure

If security issues are detected:

1. **Immediate Actions:**
   ```bash
   # Rotate JWT secret key
   # Update .env with new JWT_SECRET_KEY
   docker-compose restart catalog
   
   # Rotate database passwords
   # Update credentials in database and .env
   docker-compose restart
   ```

2. **Revoke All Tokens:**
   - Change `JWT_SECRET_KEY` to invalidate all existing tokens
   - Users will need to re-authenticate

3. **Review Audit Logs:**
   ```bash
   docker-compose logs catalog | grep AUDIT > security-audit.log
   ```

## Additional Resources

- [SOC 2 Security Fixes Documentation](./SOC2_SECURITY_FIXES.md)
- [Environment Variables Reference](../.env.example)
- [Secret Generation Script](../scripts/generate-secrets.sh)

