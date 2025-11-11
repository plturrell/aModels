# Changelog

## [Unreleased] - 2024-11-11

### Added

#### Security & Authentication
- **JWT Authentication**: Full JWT token support with validation and expiration
- **API Key Authentication**: Simple API key authentication for service-to-service communication
- **Authentication Middleware**: Optional or required authentication via `DMS_REQUIRE_AUTH` flag
- **Security Dependencies**: `verify_token()` and `optional_verify_token()` for endpoint protection
- **Enhanced Password Validation**: Detects weak/default passwords in configuration
- **Environment Variables**:
  - `DMS_JWT_SECRET`: Secret key for JWT signing
  - `DMS_JWT_ALGORITHM`: JWT algorithm (default: HS256)
  - `DMS_API_KEYS`: Comma-separated list of valid API keys
  - `DMS_REQUIRE_AUTH`: Enable/disable required authentication

#### Health Checks
- **Basic Health Check**: `GET /healthz` - Always returns 200 if service is alive
- **Detailed Health Check**: `GET /healthz/detailed` - Checks all dependencies (Postgres, Redis, Neo4j)
- **Readiness Probe**: `GET /readyz` - Kubernetes readiness check
- **Liveness Probe**: `GET /livez` - Kubernetes liveness check
- **Parallel Dependency Checks**: All health checks run concurrently for fast response

#### Database Migrations
- **Alembic Setup**: Complete Alembic configuration for schema versioning
- **Initial Migration**: `001_initial_schema.py` - Creates documents and document_versions tables
- **Async Support**: Alembic configured for async SQLAlchemy engine
- **Environment-based Configuration**: Uses `DMS_POSTGRES_DSN` from environment

#### Docker & Deployment
- **Secure docker-compose**: No hardcoded credentials, uses `.env` file
- **Environment Template**: `.env.example` with secure defaults and documentation
- **Health Checks**: Container health checks for all services
- **Build Context Fix**: Corrected docker-compose build context path

#### Documentation
- **Enhanced README**: Added sections for features, authentication, migrations, health checks, API endpoints
- **Security Guide**: Comprehensive `SECURITY.md` with best practices, checklists, and incident response
- **Migration Instructions**: Step-by-step guide for running Alembic migrations
- **Authentication Examples**: JWT and API key usage examples

### Changed

#### Code Quality
- **Module-level Imports**: Moved all imports to module level in `documents.py`
- **Import Organization**: Removed inline imports from functions
- **Consistent Error Handling**: Unified `HTTPException` usage across endpoints
- **Type Annotations**: Enhanced type hints in auth module

#### Configuration
- **Postgres DSN Validation**: Enhanced to detect more weak password patterns
- **Neo4j Password Validation**: Stricter validation for production use
- **Docker Compose Variables**: All credentials now use environment variables with defaults

### Dependencies

#### Added
- `pyjwt==2.9.0`: JWT token encoding/decoding

## Security Notes

⚠️ **Breaking Changes for Production Deployments:**

1. **Default credentials are now rejected**: The service will not start with weak passwords like `postgres`, `neo4j`, etc.
2. **Authentication is now available**: Set `DMS_REQUIRE_AUTH=true` to enforce authentication on all endpoints
3. **Environment variables required**: Must use `.env` file or set environment variables for docker-compose

## Migration Guide

### From Previous Version

1. **Create .env file:**
```bash
cp .env.example .env
# Edit .env with secure credentials
```

2. **Run database migrations:**
```bash
export DMS_POSTGRES_DSN="your-secure-dsn"
alembic upgrade head
```

3. **Update docker-compose:**
```bash
# Old: Had hardcoded credentials
# New: Uses .env file
docker compose up --build
```

4. **Optional: Enable authentication:**
```bash
# Add to .env
DMS_REQUIRE_AUTH=true
DMS_JWT_SECRET=$(openssl rand -hex 32)
```

## Testing

All changes have been tested with:
- Unit tests for authentication logic
- Integration tests for health checks
- Migration testing against PostgreSQL 15

## Contributors

- System improvements based on security audit recommendations
- Authentication patterns following FastAPI best practices
- Alembic configuration based on async SQLAlchemy standards
