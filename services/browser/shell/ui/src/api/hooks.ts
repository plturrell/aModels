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
  created_at: string;
  updated_at: string;
}

export function useDocuments() {
  return useApiData<DocumentRecord[]>("/dms/documents");
}
