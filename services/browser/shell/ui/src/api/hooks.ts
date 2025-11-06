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
