# Next Steps for Advanced Features

## Overview

Before returning to advanced features (orchestration and analytics), the following steps must be completed to enable the full pipeline with multimodal extraction and OCR capabilities.

---

## Prerequisites

These steps are required to enable:
- **Multimodal Extraction**: Image/document processing with OCR
- **DeepSeek OCR**: Advanced OCR capabilities for document analysis
- **DMS Integration**: Document Management System integration for catalog population

---

## Step 1: Deploy Stack with Multimodal Extraction

### Action Required:

Deploy the stack with the following environment variables enabled on the **extract service**:

```bash
USE_MULTIMODAL_EXTRACTION=true
USE_DEEPSEEK_OCR=true
```

### Implementation:

1. **Update docker-compose.yml** for extract service:
   ```yaml
   extract:
     environment:
       - USE_MULTIMODAL_EXTRACTION=${USE_MULTIMODAL_EXTRACTION:-true}
       - USE_DEEPSEEK_OCR=${USE_DEEPSEEK_OCR:-true}
   ```

2. **Ensure DeepSeek Model Weights Available**:
   - Follow repository documentation for DeepSeek model setup
   - Verify model weights are in the correct location
   - Ensure extract service can access model files

3. **Restart Extract Service**:
   ```bash
   docker compose -f infrastructure/docker/brev/docker-compose.yml restart extract
   ```

### Verification:

- Check extract service logs for multimodal extraction initialization
- Verify DeepSeek OCR model loading
- Test multimodal extraction endpoint with sample document/image

---

## Step 2: Configure DMS Integration

### Action Required:

Set the following environment variables in **production (or GPU) environments**:

```bash
DMS_EXTRACT_URL=<extract-service-url>
DMS_CATALOG_URL=<catalog-service-url>
```

### Implementation:

1. **Update Environment Configuration**:
   - Set `DMS_EXTRACT_URL` to extract service endpoint (e.g., `http://extract-service:8082`)
   - Set `DMS_CATALOG_URL` to catalog service endpoint (e.g., `http://catalog:8084`)

2. **Rerun Documents Through DMS**:
   - Process existing documents through Document Management System
   - OCR summaries should populate automatically
   - Catalog IDs should be assigned to extracted entities

3. **Verify DMS Integration**:
   - Check that documents are being processed
   - Verify OCR summaries are being generated
   - Confirm catalog IDs are being assigned

### Expected Results:

- Documents processed through DMS have OCR summaries
- Extracted entities have catalog IDs
- Data flows from extract → catalog automatically

---

## Step 3: Update GPU Server Configuration

### Action Required:

Update the GPU server's compose/env configuration and restart services.

### Implementation:

1. **Update docker-compose.yml on GPU Server**:
   ```yaml
   extract:
     environment:
       - USE_MULTIMODAL_EXTRACTION=true
       - USE_DEEPSEEK_OCR=true
       - DMS_EXTRACT_URL=${DMS_EXTRACT_URL:-http://extract-service:8082}
       - DMS_CATALOG_URL=${DMS_CATALOG_URL:-http://catalog:8084}
   ```

2. **Update Environment Variables**:
   - Set `USE_MULTIMODAL_EXTRACTION=true`
   - Set `USE_DEEPSEEK_OCR=true`
   - Configure `DMS_EXTRACT_URL` and `DMS_CATALOG_URL`

3. **Restart Services**:
   ```bash
   docker compose -f infrastructure/docker/brev/docker-compose.yml restart extract catalog
   ```

4. **Verify Enriched Pipeline**:
   - Check that training-shell reflects enriched pipeline
   - Verify Documents module surfaces OCR synopsis
   - Confirm catalog integration is working

### Expected Results:

- Services restart with new configuration
- Training shell shows enriched pipeline
- Documents module displays OCR synopsis
- Catalog IDs are populated for extracted entities

---

## Verification Checklist

- [ ] Extract service deployed with `USE_MULTIMODAL_EXTRACTION=true`
- [ ] Extract service deployed with `USE_DEEPSEEK_OCR=true`
- [ ] DeepSeek model weights available and accessible
- [ ] `DMS_EXTRACT_URL` configured in production/GPU environment
- [ ] `DMS_CATALOG_URL` configured in production/GPU environment
- [ ] Documents rerun through DMS
- [ ] OCR summaries populating correctly
- [ ] Catalog IDs assigned to extracted entities
- [ ] GPU server compose/env updated
- [ ] Services restarted successfully
- [ ] Training shell reflects enriched pipeline
- [ ] Documents module shows OCR synopsis

---

## Next Steps After Completion

Once these steps are complete:

1. **Return to Advanced Features**:
   - Complete graph-server deployment (resolve dependency issues)
   - Verify orchestration endpoints are working
   - Test full automation pipeline

2. **End-to-End Testing**:
   - Test multimodal extraction with OCR
   - Verify DMS → Extract → Catalog flow
   - Test enriched pipeline in training shell

3. **Production Deployment**:
   - Deploy to production with all features enabled
   - Monitor performance and resource usage
   - Validate end-to-end workflows

---

## Notes

- **DeepSeek Model Weights**: Ensure model weights are downloaded and placed in the correct location per repository documentation
- **DMS Integration**: May require additional configuration depending on DMS implementation
- **GPU Resources**: DeepSeek OCR may require GPU resources - verify GPU availability
- **Performance**: Monitor resource usage when enabling multimodal extraction and OCR

---

**Status:** ⏸️ **Pending - Next Steps Required**  
**Created:** 2025-11-06  
**Priority:** High - Required before advanced features work

