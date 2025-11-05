# SAP Business Data Cloud Inbound Source Service

This service provides an inbound source integration for SAP Business Data Cloud, allowing aModels to extract data products, intelligent applications, and schemas from SAP Business Data Cloud formations.

## Architecture

Based on the [SAP Business Data Cloud Architecture](https://architecture.learning.sap.com/docs/ref-arch/f5b6b597a6/1) and [SAP Business Data Cloud Cockpit](https://learning.sap.com/courses/introducing-sap-business-data-cloud/exploring-and-deploying-objects-with-sap-business-data-cloud-cockpit), this service integrates with:

- **SAP Business Data Cloud Cockpit**: For discovering and deploying data products and intelligent applications
- **SAP Datasphere**: For extracting schemas from managed spaces
- **SAP HANA Data Lake** (Foundation Service): For extracting schemas from the Foundation Service layer
- **Formations**: For managing component bindings and data sources

## Features

- **Data Product Discovery**: List and retrieve details of available data products
- **Intelligent Application Discovery**: List and retrieve details of intelligent applications
- **Schema Extraction**: Extract schemas from both SAP Datasphere spaces and SAP HANA Data Lake
- **Formation Management**: Get formation details including components and data sources
- **Graph Conversion**: Convert SAP BDC schemas to aModels knowledge graph format

## Configuration

Set the following environment variables:

```bash
# Required
SAP_BDC_BASE_URL=https://your-bdc-instance.com
SAP_BDC_API_TOKEN=your-api-token
SAP_BDC_FORMATION_ID=your-formation-id

# Optional
SAP_DATASPHERE_URL=https://your-datasphere-instance.com
PORT=8083
```

## API Endpoints

### `GET /healthz`
Health check endpoint.

### `POST /extract`
Extract data and schema from SAP Business Data Cloud.

**Request Body:**
```json
{
  "formation_id": "formation-123",
  "source_system": "SAP S/4HANA Cloud",
  "data_product_id": "product-456",
  "space_id": "space-789",
  "database": "HANADB",
  "include_views": true,
  "options": {}
}
```

**Response:**
```json
{
  "success": true,
  "formation": {...},
  "data_products": [...],
  "schema": {
    "database": "HANADB",
    "schema": "PUBLIC",
    "tables": [...],
    "views": [...]
  }
}
```

### `GET /data-products`
List all available data products in the formation.

### `GET /intelligent-applications`
List all available intelligent applications.

### `GET /formation`
Get formation details including components and data sources.

## Integration with Extract Service

The SAP BDC service can be integrated with the Extract service to automatically extract schemas and convert them to knowledge graphs:

1. The Extract service calls the SAP BDC service `/extract` endpoint
2. SAP BDC service extracts schema from SAP Datasphere or SAP HANA Data Lake
3. Schema is converted to aModels graph format (nodes and edges)
4. Graph is returned to Extract service for processing

## References

- [SAP Business Data Cloud Architecture](https://architecture.learning.sap.com/docs/ref-arch/f5b6b597a6/1)
- [SAP Business Data Cloud Cockpit](https://learning.sap.com/courses/introducing-sap-business-data-cloud/exploring-and-deploying-objects-with-sap-business-data-cloud-cockpit)

