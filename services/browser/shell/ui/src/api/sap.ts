/**
 * SAP BDC API Client
 * 
 * Client for interacting with SAP Business Data Cloud endpoints
 */

import { fetchJSON } from './client';

const API_BASE = import.meta.env.VITE_GATEWAY_URL || 'http://localhost:8000';

export interface SAPBDCConnectionConfig {
  base_url: string;
  api_token: string;
  formation_id: string;
  datasphere_url?: string;
}

export interface SAPBDCExtractRequest {
  formation_id: string;
  source_system: string;
  data_product_id?: string;
  space_id?: string;
  database?: string;
  include_views?: boolean;
  options?: Record<string, any>;
}

export interface SAPBDCExtractResponse {
  success: boolean;
  formation?: any;
  data_products?: SAPDataProduct[];
  schema?: SAPSchema;
  metadata?: Record<string, any>;
  error?: string;
}

export interface SAPDataProduct {
  id: string;
  name: string;
  description?: string;
  version?: string;
  status?: string;
  metadata?: Record<string, any>;
}

export interface SAPIntelligentApplication {
  id: string;
  name: string;
  description?: string;
  type?: string;
  status?: string;
}

export interface SAPFormation {
  id: string;
  name: string;
  components?: any[];
  data_sources?: any[];
  metadata?: Record<string, any>;
}

export interface SAPSchema {
  database: string;
  schema: string;
  tables: SAPTable[];
  views?: SAPView[];
}

export interface SAPTable {
  name: string;
  columns: SAPColumn[];
  primary_keys?: string[];
  foreign_keys?: SAPForeignKey[];
  metadata?: Record<string, any>;
}

export interface SAPView {
  name: string;
  definition?: string;
  columns: SAPColumn[];
  metadata?: Record<string, any>;
}

export interface SAPColumn {
  name: string;
  type: string;
  nullable?: boolean;
  default?: any;
  comment?: string;
  metadata?: Record<string, any>;
}

export interface SAPForeignKey {
  column: string;
  referenced_table: string;
  referenced_column: string;
}

/**
 * Extract data and schema from SAP Business Data Cloud
 */
export async function extractFromSAPBDC(
  request: SAPBDCExtractRequest
): Promise<SAPBDCExtractResponse> {
  return fetchJSON<SAPBDCExtractResponse>('/sap-bdc/extract', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * List all available data products in the formation
 */
export async function listSAPDataProducts(): Promise<SAPDataProduct[]> {
  try {
    const response = await fetchJSON<SAPDataProduct[] | { data_products: SAPDataProduct[] }>(
      '/v2/sap-bdc/data-products'
    );
    // Handle both response formats
    if (Array.isArray(response)) {
      return response;
    }
    return (response as any).data_products || [];
  } catch (error) {
    // Fallback to v1 endpoint
    try {
      const response = await fetchJSON<SAPDataProduct[] | { data_products: SAPDataProduct[] }>(
        '/sap-bdc/data-products'
      );
      if (Array.isArray(response)) {
        return response;
      }
      return (response as any).data_products || [];
    } catch {
      throw error;
    }
  }
}

/**
 * List all available intelligent applications
 */
export async function listSAPIntelligentApplications(): Promise<SAPIntelligentApplication[]> {
  try {
    const response = await fetchJSON<SAPIntelligentApplication[] | { applications: SAPIntelligentApplication[] }>(
      '/sap-bdc/intelligent-applications'
    );
    if (Array.isArray(response)) {
      return response;
    }
    return (response as any).applications || [];
  } catch (error) {
    throw error;
  }
}

/**
 * Get formation details including components and data sources
 */
export async function getSAPFormation(): Promise<SAPFormation> {
  return fetchJSON<SAPFormation>('/sap-bdc/formation');
}

/**
 * Test SAP BDC connection
 */
export async function testSAPConnection(config: SAPBDCConnectionConfig): Promise<{ success: boolean; message: string }> {
  try {
    // Validate configuration
    if (!config.base_url) {
      return {
        success: false,
        message: 'SAP BDC Base URL is required. Please enter the base URL of your SAP Business Data Cloud instance.',
      };
    }
    if (!config.api_token) {
      return {
        success: false,
        message: 'API Token is required. Generate this from your SAP BDC Cockpit under API Management.',
      };
    }
    if (!config.formation_id) {
      return {
        success: false,
        message: 'Formation ID is required. You can find this in the SAP BDC Cockpit.',
      };
    }

    // Try to get formation as a connection test
    const formation = await getSAPFormation();
    return {
      success: true,
      message: `Connected successfully. Formation: ${formation.name || formation.id}`,
    };
  } catch (error) {
    let message = 'Connection failed';
    if (error instanceof Error) {
      if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
        message = `Cannot reach SAP BDC service. Please ensure:
1. The SAP BDC service is running and configured
2. The base URL is correct: ${config.base_url}
3. Your network allows connections to the SAP BDC instance
4. Check the gateway logs for more details`;
      } else if (error.message.includes('401') || error.message.includes('Unauthorized')) {
        message = 'Authentication failed. Please check your API token is valid and has not expired.';
      } else if (error.message.includes('404') || error.message.includes('Not Found')) {
        message = 'Formation not found. Please verify the Formation ID is correct.';
      } else {
        message = error.message;
      }
    }
    return {
      success: false,
      message,
    };
  }
}

/**
 * Convert SAP schema to graph format (for visualization)
 */
export function convertSAPSchemaToGraph(schema: SAPSchema): {
  nodes: Array<{ id: string; type: string; label: string; properties: Record<string, any> }>;
  edges: Array<{ source: string; target: string; type: string; properties: Record<string, any> }>;
} {
  const nodes: Array<{ id: string; type: string; label: string; properties: Record<string, any> }> = [];
  const edges: Array<{ source: string; target: string; type: string; properties: Record<string, any> }> = [];

  // Database node
  const dbId = `db:${schema.database}`;
  nodes.push({
    id: dbId,
    type: 'database',
    label: schema.database,
    properties: { schema: schema.schema, source: 'sap_bdc' },
  });

  // Table nodes
  schema.tables.forEach((table) => {
    const tableId = `table:${schema.database}.${schema.schema}.${table.name}`;
    nodes.push({
      id: tableId,
      type: 'table',
      label: table.name,
      properties: {
        database: schema.database,
        schema: schema.schema,
        column_count: table.columns.length,
        ...table.metadata,
      },
    });

    // Database contains table
    edges.push({
      source: dbId,
      target: tableId,
      type: 'CONTAINS',
      properties: {},
    });

    // Column nodes
    table.columns.forEach((column) => {
      const columnId = `column:${tableId}.${column.name}`;
      nodes.push({
        id: columnId,
        type: 'column',
        label: column.name,
        properties: {
          data_type: column.type,
          nullable: column.nullable,
          default: column.default,
          comment: column.comment,
          ...column.metadata,
        },
      });

      // Table has column
      edges.push({
        source: tableId,
        target: columnId,
        type: 'HAS_COLUMN',
        properties: {},
      });
    });

    // Foreign key relationships
    table.foreign_keys?.forEach((fk) => {
      const targetTableId = `table:${schema.database}.${schema.schema}.${fk.referenced_table}`;
      edges.push({
        source: tableId,
        target: targetTableId,
        type: 'REFERENCES',
        properties: {
          column: fk.column,
          referenced_column: fk.referenced_column,
        },
      });
    });
  });

  // View nodes
  schema.views?.forEach((view) => {
    const viewId = `view:${schema.database}.${schema.schema}.${view.name}`;
    nodes.push({
      id: viewId,
      type: 'view',
      label: view.name,
      properties: {
        database: schema.database,
        schema: schema.schema,
        definition: view.definition,
        ...view.metadata,
      },
    });

    // Database contains view
    edges.push({
      source: dbId,
      target: viewId,
      type: 'CONTAINS',
      properties: {},
    });

    // View columns
    view.columns.forEach((column) => {
      const columnId = `column:${viewId}.${column.name}`;
      nodes.push({
        id: columnId,
        type: 'column',
        label: column.name,
        properties: {
          data_type: column.type,
          nullable: column.nullable,
          ...column.metadata,
        },
      });

      edges.push({
        source: viewId,
        target: columnId,
        type: 'HAS_COLUMN',
        properties: {},
      });
    });
  });

  return { nodes, edges };
}

