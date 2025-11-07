# XSUAA Integration with Differential Privacy and Domain Intelligence

This document explains how XSUAA authentication integrates with aModels' differential privacy and domain intelligence features.

## Overview

The integration provides:
- **Role-based domain access** via XSUAA scopes
- **Privacy-aware data access** with differential privacy based on user permissions
- **Domain intelligence routing** using user's accessible domains
- **Privacy budget management** per user and domain
- **Sensitive data protection** with automatic privacy level adjustment

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Client Request                       │
│              (with XSUAA Bearer Token)                  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              XSUAA Authentication                        │
│  - Validates JWT Token                                  │
│  - Extracts User Info, Scopes, Authorities              │
│  - Determines User's Accessible Domains                 │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│      Privacy-Domain Integration Layer                     │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Domain Access Resolution                          │  │
│  │  - Maps XSUAA scopes → domain access              │  │
│  │  - Determines privacy level per domain            │  │
│  │  - Checks restricted domain access                │  │
│  └────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Privacy Configuration                             │  │
│  │  - Calculates epsilon/delta based on access level  │  │
│  │  - Sets noise scale and max queries               │  │
│  │  - Adjusts for sensitive domains                  │  │
│  └────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Domain Intelligence Context                       │  │
│  │  - Provides user domains to routing               │  │
│  │  - Enables domain-aware query processing          │  │
│  └────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌──────────────────┐   ┌──────────────────┐
│ Domain Routing    │   │ Privacy Service  │
│ - Uses user domains │   │ - Applies noise │
│ - Routes to models  │   │ - Manages budget│
└──────────────────┘   └──────────────────┘
```

## Domain Access Model

### Scope Format

XSUAA scopes follow this format for domain-specific access:
```
$XSAPPNAME.Domain.<domain_id>.<access_level>
```

**Examples:**
- `amodels-catalog.Domain.finance.read` - Read access to finance domain
- `amodels-catalog.Domain.health.write` - Write access to health domain
- `amodels-catalog.Domain.regulatory.admin` - Admin access to regulatory domain

### Privacy Levels

Privacy levels are automatically determined based on:
1. **Domain sensitivity** (finance, health, PII = high privacy)
2. **User access level** (admin = low privacy, read = high privacy)
3. **Data classification** (restricted domains = high privacy)

**Privacy Level Mapping:**
- **Low Privacy** (ε=2.0, δ=1e-4): Admin access to non-sensitive domains
- **Medium Privacy** (ε=1.0, δ=1e-5): Standard access to general domains
- **High Privacy** (ε=0.5, δ=1e-6): Read access to sensitive/restricted domains

## Integration Points

### 1. Domain Access Resolution

```go
// Get user's accessible domains from XSUAA scopes
domainAccesses, err := privacyIntegration.GetUserDomainAccess(ctx)
// Returns: []DomainAccess with domain_id, access_level, privacy_level

// Check if user can access specific domain
canAccess, accessLevel, err := privacyIntegration.CanAccessDomain(ctx, "finance")
```

**Scope Mapping:**
- Scopes like `Domain.finance.read` → DomainAccess{domainID: "finance", accessLevel: "read"}
- General scopes like `Display` → Default domain access
- Admin scopes → All domains with admin access

### 2. Privacy Configuration

```go
// Get privacy config based on user's domain access
privacyConfig, err := privacyIntegration.GetPrivacyConfig(ctx, "finance")
// Returns: PrivacyConfig with epsilon, delta, noise_scale, max_queries

// Privacy config is automatically adjusted:
// - Admin users: 1.5x epsilon, 1.5x max_queries
// - Read-only users: 0.8x epsilon, 0.8x max_queries
// - Restricted domains: 0.7x epsilon, 0.1x delta, 0.7x max_queries
```

**Privacy Budget:**
- **Low Privacy**: 200 queries/day per domain
- **Medium Privacy**: 100 queries/day per domain
- **High Privacy**: 50 queries/day per domain

### 3. Domain Intelligence Routing

```go
// Get domain intelligence context for routing
context, err := privacyIntegration.GetDomainIntelligenceContext(ctx)
// Returns: map with user_id, domains, domain_access_levels, scopes

// Use in domain detection/routing
userDomains := context["domains"].([]string)
domainManager.DetectDomain(prompt, userDomains) // Only considers accessible domains
```

**Domain Filtering:**
- Domain intelligence only routes to domains user has access to
- Domain detection scores only accessible domains
- Fallback to default domain if no matches

### 4. Privacy-Aware Data Responses

```go
// Apply privacy to response data
privateData, err := privacyIntegration.ApplyPrivacyToResponse(ctx, "finance", data)
// Automatically:
// - Skips sensitive fields for high privacy levels
// - Adds noise metadata for numeric values
// - Includes privacy configuration in response
```

## Usage Examples

### Example 1: Domain-Specific Endpoint Protection

```go
// Protect endpoint with domain access requirement
mux.Handle("/catalog/domains/finance/data",
    xsuaaMiddleware.Middleware(
        privacyIntegration.RequireDomainAccess("finance")(
            http.HandlerFunc(handleFinanceData),
        ),
    ),
)
```

### Example 2: Privacy-Aware Query Processing

```go
func handleQuery(w http.ResponseWriter, r *http.Request) {
    ctx := r.Context()
    
    // Get user's accessible domains
    domains, _ := privacyIntegration.GetUserDomains(ctx)
    
    // Get privacy config for domain
    domainID := detectDomainFromQuery(r)
    privacyConfig, _ := privacyIntegration.GetPrivacyConfig(ctx, domainID)
    
    // Process query with privacy constraints
    result := processQueryWithPrivacy(query, privacyConfig)
    
    // Apply privacy to response
    privateResult, _ := privacyIntegration.ApplyPrivacyToResponse(ctx, domainID, result)
    
    json.NewEncoder(w).Encode(privateResult)
}
```

### Example 3: Domain Intelligence Integration

```go
func handleIntelligentRouting(w http.ResponseWriter, r *http.Request) {
    ctx := r.Context()
    
    // Get domain intelligence context
    context, _ := privacyIntegration.GetDomainIntelligenceContext(ctx)
    
    // Use in intelligent router
    decision, _ := intelligentRouter.RouteQuery(
        ctx,
        query,
        context["domains"].([]string),
        context,
    )
    
    // Decision only includes accessible domains
    json.NewEncoder(w).Encode(decision)
}
```

## XSUAA Scope Configuration

### Domain-Specific Scopes

Add to `xs-security.json`:

```json
{
  "name": "$XSAPPNAME.Domain.<domain_id>.<access_level>",
  "description": "Access to specific domain"
}
```

**Predefined Domain Scopes:**
- `Domain.finance.read/write/admin`
- `Domain.health.read/write/admin`
- `Domain.regulatory.read/write/admin`
- `Domain.pii.read/write/admin`

### Privacy Scopes

```json
{
  "name": "$XSAPPNAME.Privacy.Low",
  "description": "Access to low-privacy data (higher epsilon)"
},
{
  "name": "$XSAPPNAME.Privacy.Medium",
  "description": "Access to medium-privacy data"
},
{
  "name": "$XSAPPNAME.Privacy.High",
  "description": "Access to high-privacy data (lower epsilon)"
}
```

## Privacy Budget Management

### Per-User, Per-Domain Budgets

Privacy budgets are tracked per user and domain:

```go
// Budget consumption
cost := 1.0 / privacyConfig.MaxQueries
usedBudget := (queryCount * cost) / privacyConfig.Epsilon

if usedBudget >= 1.0 {
    return errors.New("privacy budget exceeded")
}
```

### Budget Reset

Budgets reset daily or can be reset manually:
- **Daily Reset**: Automatic at midnight UTC
- **Manual Reset**: Via admin endpoint
- **Per-Domain**: Each domain has independent budget

## Sensitive Domain Detection

Automatically detected sensitive domains:
- **finance** - Financial data
- **health** - Health information
- **pii** - Personally Identifiable Information
- **regulatory** - Regulatory/compliance data
- **compliance** - Compliance data
- **confidential** - Confidential data
- **restricted** - Restricted data

Sensitive domains automatically get:
- Higher privacy level (high)
- Stricter epsilon/delta values
- Lower query limits
- Additional access restrictions

## Audit Logging

All privacy and domain access events are logged:

```
[AUDIT] domain_access_granted user_id=user123 domain=finance access_level=read
[AUDIT] privacy_config_applied user_id=user123 domain=finance epsilon=0.5 delta=1e-6
[AUDIT] privacy_budget_consumed user_id=user123 domain=finance cost=0.01 remaining=0.99
[AUDIT] domain_access_denied user_id=user123 domain=health path=/catalog/domains/health/data
[AUDIT] sensitive_data_filtered user_id=user123 domain=finance fields_removed=3
```

## Security Considerations

1. **Scope Validation**: Always validate scopes before granting domain access
2. **Privacy Budget Enforcement**: Strictly enforce query limits per domain
3. **Sensitive Data Filtering**: Automatically filter sensitive fields for high privacy
4. **Audit Trail**: Log all privacy and domain access decisions
5. **Least Privilege**: Grant minimum required domain access
6. **Budget Monitoring**: Monitor privacy budget consumption per user/domain

## Configuration

### Environment Variables

```bash
# Enable privacy-domain integration
ENABLE_PRIVACY_DOMAIN_INTEGRATION=true

# Privacy budget reset schedule (cron format)
PRIVACY_BUDGET_RESET_SCHEDULE="0 0 * * *"  # Daily at midnight

# Default privacy level for unknown domains
DEFAULT_PRIVACY_LEVEL=medium

# Sensitive domain keywords (comma-separated)
SENSITIVE_DOMAIN_KEYWORDS=finance,health,pii,regulatory,compliance
```

## Testing

### Test Domain Access

```bash
# Test with finance domain scope
curl -X GET https://catalog.cfapps.sap.hana.ondemand.com/catalog/domains/finance/data \
  -H "Authorization: Bearer $TOKEN_WITH_FINANCE_SCOPE"

# Should return data with privacy applied
```

### Test Privacy Budget

```bash
# Make multiple queries to consume budget
for i in {1..100}; do
  curl -X GET https://catalog.cfapps.sap.hana.ondemand.com/catalog/domains/finance/data \
    -H "Authorization: Bearer $TOKEN"
done

# 101st query should fail with "privacy budget exceeded"
```

### Test Domain Intelligence

```bash
# Query with domain intelligence
curl -X POST https://catalog.cfapps.sap.hana.ondemand.com/catalog/intelligent-route \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "financial analysis", "context": {}}'

# Response should only include accessible domains
```

## Integration Flow Diagram

```
User Request
    │
    ├─► XSUAA Token Validation
    │       │
    │       ├─► Extract Scopes
    │       ├─► Extract User Info
    │       └─► Extract Authorities
    │
    ├─► Domain Access Resolution
    │       │
    │       ├─► Parse Domain Scopes
    │       ├─► Determine Access Levels
    │       └─► Check Restricted Domains
    │
    ├─► Privacy Configuration
    │       │
    │       ├─► Calculate Epsilon/Delta
    │       ├─► Set Noise Scale
    │       └─► Set Query Limits
    │
    ├─► Domain Intelligence
    │       │
    │       ├─► Filter Accessible Domains
    │       ├─► Route to Domain Models
    │       └─► Apply Domain Context
    │
    └─► Privacy-Aware Response
            │
            ├─► Apply Noise to Data
            ├─► Filter Sensitive Fields
            └─► Add Privacy Metadata
```

This integration ensures that differential privacy and domain intelligence work seamlessly with XSUAA authentication, providing secure, privacy-preserving, domain-aware access to aModels services.

