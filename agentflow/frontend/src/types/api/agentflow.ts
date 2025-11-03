export interface AgentFlowCatalogItem {
  id: string;
  name: string;
  category?: string | null;
  description?: string | null;
  relative_path: string;
}

export interface AgentFlowCatalogResponse {
  total: number;
  page: number;
  page_size: number;
  items: AgentFlowCatalogItem[];
}

export interface AgentFlowImportResponse {
  imported: boolean;
  flow: {
    id?: string;
    name?: string;
    [key: string]: unknown;
  };
}
