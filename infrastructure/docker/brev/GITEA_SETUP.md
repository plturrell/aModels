# Gitea Setup Guide for aModels

## Overview

Gitea has been integrated into the aModels infrastructure as a Docker service. This guide will help you set up and configure Gitea for code storage and version control.

---

## Quick Start

### 1. Start Gitea Service

```bash
# From aModels root directory
cd /home/aModels/infrastructure/docker/brev

# Start Gitea (and dependencies)
docker-compose up -d postgres gitea

# Check Gitea status
docker-compose ps gitea
docker-compose logs -f gitea
```

### 2. Initial Configuration

Once Gitea is running, complete the initial setup:

1. **Open Gitea in browser:**
   ```
   http://localhost:3000
   ```

2. **First-time setup** (if not already configured):
   - The database is already configured via environment variables
   - Click **"Complete Installation"** if prompted
   - Default admin credentials will be created

3. **Create admin user:**
   ```bash
   # Create admin account from command line
   docker exec -it gitea gitea admin user create \
     --username admin \
     --password admin123 \
     --email admin@localhost \
     --admin
   ```

### 3. Configure for Extract Service

#### A. Generate API Token

1. Log in to Gitea: http://localhost:3000
2. Go to **Settings** → **Applications** → **Generate New Token**
3. Token name: `extract-service`
4. Select scopes: **All** (or at minimum: `repo`, `user`, `write:repository`)
5. Generate and **save the token**

#### B. Create Organization

```bash
# Via Gitea UI:
# 1. Click + → New Organization
# 2. Name: extract-service
# 3. Visibility: Public

# Or via API:
curl -X POST http://localhost:3000/api/v1/orgs \
  -H "Authorization: token YOUR_TOKEN_HERE" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "extract-service",
    "full_name": "Extract Service",
    "description": "Organization for extracted code repositories"
  }'
```

#### C. Configure Environment Variables

Create or update `.env` file in aModels root:

```bash
# Gitea Configuration
export GITEA_URL="http://localhost:3000"
export GITEA_TOKEN="your-api-token-from-step-3A"
export GITEA_WEBHOOK_SECRET="$(openssl rand -hex 32)"

# For Docker containers (use service name)
export GITEA_URL_DOCKER="http://gitea:3000"
```

#### D. Update Extract Service Configuration

If running Extract in Docker, update `docker-compose.yml`:

```yaml
extract:
  environment:
    - GITEA_URL=http://gitea:3000
    - GITEA_TOKEN=${GITEA_TOKEN}
    - GITEA_WEBHOOK_SECRET=${GITEA_WEBHOOK_SECRET}
```

If running natively, export environment variables:

```bash
source .env
cd services/extract
go run cmd/extract/main.go
```

---

## Verification

### Test Gitea Connection

```bash
# 1. Check Gitea health
curl http://localhost:3000/api/healthz

# 2. Test API with your token
curl http://localhost:3000/api/v1/user \
  -H "Authorization: token YOUR_TOKEN_HERE"

# 3. List repositories
curl http://localhost:3000/api/v1/user/repos \
  -H "Authorization: token YOUR_TOKEN_HERE"
```

### Test Extract Service Integration

```bash
# Test Extract → Gitea connection
curl http://localhost:8083/gitea/repositories \
  -H "X-Gitea-URL: http://localhost:3000" \
  -H "X-Gitea-Token: YOUR_TOKEN_HERE"

# Should return: [] (empty list initially)
```

### Create Test Repository

```bash
# Via Extract service
curl -X POST http://localhost:8083/gitea/repositories \
  -H "Content-Type: application/json" \
  -H "X-Gitea-URL: http://localhost:3000" \
  -H "X-Gitea-Token: YOUR_TOKEN_HERE" \
  -d '{
    "owner": "extract-service",
    "name": "test-repo",
    "description": "Test repository for Extract service",
    "private": false,
    "auto_init": true
  }'

# Verify in browser: http://localhost:3000/extract-service/test-repo
```

---

## Webhook Configuration

### 1. Configure Webhook in Gitea

For each repository that should trigger Extract processing:

1. Go to repository → **Settings** → **Webhooks** → **Add Webhook** → **Gitea**
2. **Target URL:** 
   - If Extract is in Docker: `http://extract:8083/webhooks/gitea`
   - If Extract is native: `http://localhost:8083/webhooks/gitea`
3. **HTTP Method:** `POST`
4. **POST Content Type:** `application/json`
5. **Secret:** Use the value of `GITEA_WEBHOOK_SECRET`
6. **Trigger On:** Select **Push Events**
7. **Active:** ✓ Checked
8. Click **Add Webhook**

### 2. Test Webhook

```bash
# Make a change to the repository
cd /tmp
git clone http://localhost:3000/extract-service/test-repo.git
cd test-repo
echo "# Test" > test.yaml
git add test.yaml
git commit -m "Test webhook"
git push

# Check Extract service logs
docker-compose logs -f extract
# Should see: "Webhook received from..."
```

---

## Production Configuration

### Security Settings

```yaml
# docker-compose.yml - Production settings
gitea:
  environment:
    - GITEA__service__DISABLE_REGISTRATION=true  # Disable public registration
    - GITEA__server__ROOT_URL=https://gitea.yourdomain.com/
    - GITEA__security__SECRET_KEY=${GITEA_SECRET_KEY}  # Generate with: gitea generate secret SECRET_KEY
    - GITEA__security__INTERNAL_TOKEN=${GITEA_INTERNAL_TOKEN}  # Generate with: gitea generate secret INTERNAL_TOKEN
```

### Database Backup

```bash
# Backup Gitea PostgreSQL database
docker exec postgres pg_dump -U gitea gitea > gitea_backup_$(date +%Y%m%d).sql

# Backup Gitea data volume
docker run --rm \
  -v gitea-data:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/gitea-data-$(date +%Y%m%d).tar.gz /data
```

### SSL/TLS Configuration

```yaml
# Use reverse proxy (Nginx/Traefik) for HTTPS
# Example Nginx config:
server {
    listen 443 ssl;
    server_name gitea.yourdomain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## Troubleshooting

### Gitea Won't Start

```bash
# Check logs
docker-compose logs gitea

# Common issues:
# 1. PostgreSQL not ready - wait 30s and retry
# 2. Database connection failed - check credentials
# 3. Port already in use - check: lsof -i :3000
```

### Can't Connect to Database

```bash
# Test PostgreSQL connection
docker exec -it postgres psql -U gitea -d gitea -c "SELECT 1;"

# Recreate Gitea database
docker exec -it postgres psql -U postgres <<EOF
DROP DATABASE IF EXISTS gitea;
DROP USER IF EXISTS gitea;
CREATE USER gitea WITH PASSWORD 'gitea_password';
CREATE DATABASE gitea WITH OWNER gitea;
GRANT ALL PRIVILEGES ON DATABASE gitea TO gitea;
EOF

# Restart Gitea
docker-compose restart gitea
```

### Webhook Not Triggering

```bash
# 1. Check webhook configuration in Gitea
# Repository → Settings → Webhooks → Edit

# 2. Check webhook secret matches
echo $GITEA_WEBHOOK_SECRET

# 3. Test webhook manually
curl -X POST http://localhost:8083/webhooks/gitea \
  -H "Content-Type: application/json" \
  -H "X-Gitea-Signature: sha256=$(echo -n '{}' | openssl dgst -sha256 -hmac "$GITEA_WEBHOOK_SECRET" | cut -d' ' -f2)" \
  -d '{
    "action": "push",
    "ref": "refs/heads/main",
    "repository": {"full_name": "owner/repo", "clone_url": "http://localhost:3000/owner/repo.git"},
    "commits": [{"added": ["test.yaml"]}]
  }'

# 4. Check Extract service logs
docker-compose logs -f extract | grep -i webhook
```

### Extract Service Can't Reach Gitea

```bash
# If Extract is in Docker
# Use service name: http://gitea:3000

# If Extract is native
# Use localhost: http://localhost:3000

# Test network connectivity
docker exec extract ping -c 3 gitea

# Or from host
curl http://localhost:3000/api/healthz
```

---

## Service Management

### Start/Stop Gitea

```bash
# Start
docker-compose up -d gitea

# Stop
docker-compose stop gitea

# Restart
docker-compose restart gitea

# View logs
docker-compose logs -f gitea

# Check status
docker-compose ps gitea
```

### Reset Gitea

```bash
# ⚠️ WARNING: This will delete all Gitea data!

# Stop and remove containers/volumes
docker-compose down -v gitea
docker volume rm brev_gitea-data

# Restart fresh
docker-compose up -d postgres
sleep 10
docker-compose up -d gitea
```

---

## Integration with aModels Services

### Extract Service

The Extract service automatically uses Gitea if configured:

```bash
# Example: Extract code and store in Gitea
curl -X POST http://localhost:8083/knowledge-graph \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "my-project",
    "system_id": "backend",
    "json_tables": ["/path/to/schema.json"],
    "gitea_storage": {
      "enabled": true,
      "gitea_url": "http://gitea:3000",
      "gitea_token": "'"$GITEA_TOKEN"'",
      "owner": "extract-service",
      "repo_name": "my-project-extracted-code",
      "auto_create": true
    }
  }'
```

### Orchestration Service

The orchestration service can trigger Extract → Gitea workflows:

```bash
# Create repository via orchestration
curl -X POST http://localhost:8085/api/gitea/repositories \
  -H "Content-Type: application/json" \
  -d '{
    "name": "project-repo",
    "description": "Project repository",
    "owner": "extract-service"
  }'
```

---

## Monitoring

### Health Check

```bash
# Gitea health endpoint
curl http://localhost:3000/api/healthz

# Expected: {"status":"ok"}
```

### Metrics

```bash
# View Gitea stats
curl http://localhost:3000/api/v1/nodeinfo

# Repository count
curl http://localhost:3000/api/v1/user/repos \
  -H "Authorization: token $GITEA_TOKEN" | jq length
```

---

## Additional Resources

- **Gitea Documentation:** https://docs.gitea.io/
- **API Documentation:** http://localhost:3000/api/swagger
- **Extract Service Integration:** `/home/aModels/services/extract/docs/GITEA_SECURITY_IMPROVEMENTS.md`
- **Security Guide:** `/home/aModels/services/extract/docs/GITEA_SECURITY_IMPROVEMENTS.md`

---

## Quick Reference

### Ports
- **HTTP:** 3000
- **SSH:** 2222 (mapped from container port 22)

### Credentials
- **Admin User:** Create with `docker exec` command above
- **Database:** `gitea` / `gitea_password` (PostgreSQL)
- **API Token:** Generate in Gitea UI

### Environment Variables
```bash
GITEA_URL=http://localhost:3000
GITEA_TOKEN=<your-token>
GITEA_WEBHOOK_SECRET=<random-32-char-hex>
```

### Docker Commands
```bash
docker-compose up -d gitea           # Start
docker-compose logs -f gitea         # Logs
docker-compose restart gitea         # Restart
docker exec -it gitea sh            # Shell access
```

---

## Support

For issues or questions:
1. Check logs: `docker-compose logs gitea`
2. Review documentation above
3. Check Gitea documentation: https://docs.gitea.io/
