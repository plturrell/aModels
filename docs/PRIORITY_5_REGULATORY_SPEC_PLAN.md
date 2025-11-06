# Priority 5: Regulatory Reporting Spec Extraction

## Overview
Implement regulatory reporting specification extraction for MAS 610 and BCBS 239, with validation and LangExtract integration.

**Estimated Time**: 3-4 hours  
**Components**: 3 core modules

---

## Components

### 1. Regulatory Spec Extractor
**Purpose**: Extract structured specifications from regulatory documents (MAS 610, BCBS 239).

**Features**:
- Document parsing (PDF, Word, text)
- Domain-specific LangExtract prompts
- Multi-step extraction (document → sections → fields → rules)
- Structured output schemas
- Confidence scoring
- Regulatory context tagging

### 2. Spec Validation Engine
**Purpose**: Validate extracted specifications against regulatory requirements.

**Features**:
- Regulatory schema repository
- Field completeness validation
- Data type validation
- Business rule validation
- Cross-reference validation
- Version-aware validation

### 3. Regulatory Schema Repository
**Purpose**: Store and version regulatory schemas and extracted specifications.

**Features**:
- Schema versioning
- Regulatory document tracking
- Change management
- Schema mapping to data products
- Compliance reporting

---

## Architecture

```
RegulatorySpecSystem
  ├── SpecExtractor
  │   ├── MAS610Extractor
  │   ├── BCBS239Extractor
  │   └── GenericRegExtractor
  ├── ValidationEngine
  │   ├── SchemaValidator
  │   ├── CompletenessValidator
  │   └── RuleValidator
  └── SchemaRepository
      ├── SchemaStore
      ├── VersionManager
      └── ChangeTracker
```

---

## Implementation Plan

### Phase 1: Regulatory Spec Extractor (1.5 hours)
1. Create `RegulatorySpecExtractor` with LangExtract integration
2. Domain-specific prompts for MAS 610 and BCBS 239
3. Structured output schemas
4. Multi-step extraction pipeline

### Phase 2: Validation Engine (1 hour)
1. Create `ValidationEngine`
2. Regulatory schema definitions
3. Validation rules
4. Compliance checking

### Phase 3: Schema Repository (1 hour)
1. Create `RegulatorySchemaRepository`
2. Schema versioning
3. Document tracking
4. Integration with knowledge graph

### Phase 4: Integration (0.5 hours)
1. Wire all components together
2. API endpoints
3. Audit trail integration
4. Documentation

---

## Files to Create

1. `services/extract/regulatory/spec_extractor.go`
2. `services/extract/regulatory/validation_engine.go`
3. `services/extract/regulatory/schema_repository.go`
4. `services/extract/regulatory/mas610_extractor.go`
5. `services/extract/regulatory/bcbs239_extractor.go`
6. `services/extract/regulatory/integration.go`
7. `services/extract/migrations/0003_create_regulatory_specs.sql`

---

## Dependencies

- LangExtract (services/extract/langextract)
- Audit Trail (services/extract/langextract/audit_trail.go)
- Knowledge Graph (services/graph)
- Data Products (services/catalog)

