# SAP BTP XSUAA Integration Guide

This guide covers integrating aModels Catalog Service with SAP Business Technology Platform (BTP) using XSUAA (Extended User Account and Authentication) for user security and permissions.

## Overview

XSUAA provides:
- **OAuth 2.0 / OpenID Connect** authentication
- **User account management** via SAP Identity Provider
- **Role-based authorization** with scopes and authorities
- **Multi-tenant support** for SaaS applications
- **Integration with SAP BTP services**

## Architecture

```
┌─────────────┐
│   Client    │
│ Application│
└──────┬──────┘
       │ 1. Request with Bearer Token
       ▼
┌─────────────────────────────────┐
│   aModels Catalog Service       │
│  ┌───────────────────────────┐ │
│  │  XSUAA Middleware          │ │
│  │  - Validates JWT Token     │ │
│  │  - Extracts User Info      │ │
│  │  - Checks Scopes/Roles     │ │
│  └───────────────────────────┘ │
│  ┌───────────────────────────┐ │
│  │  Protected Endpoints       │ │
│  └───────────────────────────┘ │
└─────────────────────────────────┘
       │ 2. Validate Token
       ▼
┌─────────────────────────────────┐
│      XSUAA Service               │
│  - Token Validation             │
│  - User Information             │
│  - Scope/Role Management         │
└─────────────────────────────────┘
```

## Prerequisites

- SAP BTP account with Cloud Foundry environment
- Cloud Foundry CLI installed
- Multi-Target Application (MTA) Build Tool installed
- Access to SAP BTP Cockpit

## Configuration Files

### 1. xs-security.json

Defines security configuration including scopes, roles, and role collections.

**Location:** `services/catalog/xs-security.json`

**Key Components:**
- **Scopes**: Fine-grained permissions (e.g., `DataProduct.Read`, `DataProduct.Create`)
- **Role Templates**: Reusable role definitions with scope references
- **Role Collections**: Assignable roles for users (e.g., `CatalogViewer`, `CatalogEditor`, `CatalogAdmin`)

### 2. manifest.yml

Cloud Foundry application manifest for deployment.

**Location:** `services/catalog/manifest.yml`

**Key Settings:**
- Service bindings (XSUAA, PostgreSQL, Redis, Neo4j)
- Environment variables
- Health check configuration
- Resource limits

### 3. mta.yaml

Multi-Target Application descriptor for MTA deployment.

**Location:** `services/catalog/mta.yaml`

**Key Components:**
- Module definition (Go application)
- XSUAA service instance with security configuration
- Database service instances
- Service dependencies

## Deployment Steps

### 1. Build the Application

```bash
cd services/catalog

# Build Go application
go build -o catalog-service .

# Or use MTA build tool
mbt build
```

### 2. Deploy to SAP BTP

#### Option A: Using Cloud Foundry CLI

```bash
# Login to SAP BTP
cf login -a https://api.cf.sap.hana.ondemand.com

# Create XSUAA service instance
cf create-service xsuaa application amodels-catalog-xsuaa \
  -c xs-security.json

# Create other service instances
cf create-service postgresql v12-dev amodels-catalog-postgres
cf create-service redis cache amodels-catalog-redis

# Deploy application
cf push -f manifest.yml
```

#### Option B: Using MTA Deploy

```bash
# Build MTA archive
mbt build

# Deploy MTA
cf deploy mta_archives/amodels-catalog_1.0.0.mtar
```

### 3. Configure Role Collections

After deployment, assign role collections to users in SAP BTP Cockpit:

1. Navigate to **Security** → **Role Collections**
2. Find role collections:
   - `aModels_Catalog_Viewer`
   - `aModels_Catalog_Editor`
   - `aModels_Catalog_Admin`
3. Assign to users or user groups

## Authentication Flow

### 1. Client Obtains Token

```bash
# Get OAuth token from XSUAA
curl -X POST https://<subaccount>.authentication.sap.hana.ondemand.com/oauth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials" \
  -d "client_id=<client_id>" \
  -d "client_secret=<client_secret>"
```

### 2. Client Makes Authenticated Request

```bash
# Use token in Authorization header
curl -X GET https://amodels-catalog.cfapps.sap.hana.ondemand.com/catalog/data-elements \
  -H "Authorization: Bearer <access_token>"
```

### 3. Service Validates Token

The XSUAA middleware:
1. Extracts token from `Authorization` header
2. Validates token signature using XSUAA public key
3. Checks token expiration
4. Extracts user information and scopes
5. Adds claims to request context

## Authorization

### Scopes

Scopes define fine-grained permissions:

```go
// Check if user has specific scope
xsuaaMiddleware.RequireScope("DataProduct.Create")(handler)
```

**Available Scopes:**
- `$XSAPPNAME.Display` - View catalog data
- `$XSAPPNAME.Edit` - Edit catalog data
- `$XSAPPNAME.Admin` - Administrative access
- `$XSAPPNAME.DataProduct.Create` - Create data products
- `$XSAPPNAME.DataProduct.Read` - Read data products
- `$XSAPPNAME.DataProduct.Update` - Update data products
- `$XSAPPNAME.DataProduct.Delete` - Delete data products
- `$XSAPPNAME.BreakDetection.Read` - Read break detection
- `$XSAPPNAME.BreakDetection.Admin` - Admin break detection
- `$XSAPPNAME.Analytics.Read` - Read analytics

### Role Collections

Role collections group scopes for easier user management:

- **aModels_Catalog_Viewer**: Read-only access
- **aModels_Catalog_Editor**: Create and update access
- **aModels_Catalog_Admin**: Full administrative access

### Using Authorization in Code

```go
// In handler, check user context
claims, ok := r.Context().Value("xsuaa_claims").(*security.XSUAAClaims)
if !ok {
    http.Error(w, "Not authenticated", http.StatusUnauthorized)
    return
}

// Check scope
if !xsuaaMiddleware.HasScope(claims, "DataProduct.Create") {
    http.Error(w, "Insufficient permissions", http.StatusForbidden)
    return
}

// Access user information
userID := claims.UserID
userName := claims.UserName
email := claims.Email
scopes := claims.Scopes
```

## Environment Variables

When deployed on SAP BTP, XSUAA credentials are automatically injected via `VCAP_SERVICES`:

```json
{
  "xsuaa": [{
    "credentials": {
      "url": "https://<subaccount>.authentication.sap.hana.ondemand.com",
      "clientid": "<client_id>",
      "clientsecret": "<client_secret>",
      "xsappname": "amodels-catalog"
    }
  }]
}
```

The middleware automatically parses this configuration.

## Local Development

For local development without SAP BTP:

```bash
# Set XSUAA environment variables
export XSUAA_CLIENT_ID="your-client-id"
export XSUAA_CLIENT_SECRET="your-client-secret"
export XSUAA_URL="https://your-subaccount.authentication.sap.hana.ondemand.com"
export XSUAA_VERIFICATION_KEY="<base64-encoded-public-key>"
export XS_APP_NAME="amodels-catalog"
```

Or use a `.env` file:

```bash
XSUAA_CLIENT_ID=your-client-id
XSUAA_CLIENT_SECRET=your-client-secret
XSUAA_URL=https://your-subaccount.authentication.sap.hana.ondemand.com
XSUAA_VERIFICATION_KEY=<base64-encoded-public-key>
XS_APP_NAME=amodels-catalog
```

## Testing

### 1. Test Authentication

```bash
# Get token from XSUAA
TOKEN=$(curl -s -X POST https://<subaccount>.authentication.sap.hana.ondemand.com/oauth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials" \
  -d "client_id=<client_id>" \
  -d "client_secret=<client_secret>" | jq -r '.access_token')

# Test protected endpoint
curl -X GET https://amodels-catalog.cfapps.sap.hana.ondemand.com/catalog/data-elements \
  -H "Authorization: Bearer $TOKEN"
```

### 2. Test Authorization

```bash
# Test with different role collections
# Viewer role - should work for GET requests
# Editor role - should work for POST/PUT requests
# Admin role - should work for all requests including DELETE
```

## Troubleshooting

### Token Validation Fails

**Error:** "Invalid or expired token"

**Solutions:**
- Verify token is not expired
- Check token signature is valid
- Ensure XSUAA service is accessible
- Verify `XSUAA_VERIFICATION_KEY` is correct

### Missing Scopes

**Error:** "Insufficient scope"

**Solutions:**
- Check user's role collection assignments
- Verify scope names match `xs-security.json`
- Ensure role template includes required scope

### VCAP_SERVICES Not Found

**Error:** "XSUAA_CLIENT_ID or VCAP_SERVICES with xsuaa credentials is required"

**Solutions:**
- Verify service binding exists: `cf services`
- Check service instance is bound: `cf env amodels-catalog`
- Ensure `VCAP_SERVICES` environment variable is set

## Security Best Practices

1. **Never log tokens**: Tokens contain sensitive information
2. **Validate all scopes**: Don't trust client claims, always verify
3. **Use HTTPS**: Always use encrypted connections
4. **Rotate secrets**: Regularly rotate XSUAA client secrets
5. **Audit logging**: Log all authentication and authorization events
6. **Least privilege**: Assign minimal required scopes to users

## Integration with Other Services

### Calling Other SAP BTP Services

When calling other services, forward the XSUAA token:

```go
// Get token from request context
token := r.Header.Get("Authorization")

// Forward to downstream service
req, _ := http.NewRequest("GET", "https://other-service.cfapps.sap.hana.ondemand.com/api", nil)
req.Header.Set("Authorization", token)
```

### Service-to-Service Communication

For service-to-service calls, use client credentials flow:

```go
// Get service token using client credentials
token := getServiceToken(xsuaaConfig.ClientID, xsuaaConfig.ClientSecret)
```

## Additional Resources

- [SAP BTP XSUAA Documentation](https://help.sap.com/docs/btp/sap-business-technology-platform/security-concepts)
- [XSUAA API Reference](https://api.sap.com/api/xsuaa/overview)
- [Cloud Foundry Security](https://docs.cloudfoundry.org/concepts/security.html)
- [OAuth 2.0 Specification](https://oauth.net/2/)

