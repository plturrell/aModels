import { useApiData } from "./client";

export interface LocalAIModelEntry {
  id: string;
  readme: boolean;
}

export interface LocalAIInventory {
  generatedAt: string;
  models: LocalAIModelEntry[];
}

export function useLocalAIInventory() {
  return useApiData<LocalAIInventory>('/api/localai/models');
}

export interface DocumentRecord {
  id: string;
  name: string;
  description?: string | null;
  storage_path: string;
  catalog_identifier?: string | null;
  extraction_summary?: string | null;
  created_at: string; // ISO 8601 datetime string from FastAPI
  updated_at: string; // ISO 8601 datetime string from FastAPI
}

export function useDocuments() {
  return useApiData<DocumentRecord[]>("/dms/documents");
}
