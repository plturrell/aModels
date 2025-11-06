import { create } from "zustand";

export interface InteractionMetric {
  id: string;
  model: string;
  durationMs: number;
  timestamp: number;
  promptChars: number;
  completionChars: number;
  promptTokens?: number;
  completionTokens?: number;
  citations: number;
}

export interface TelemetryState {
  metrics: InteractionMetric[];
  recordInteraction: (metric: InteractionMetric) => void;
  reset: () => void;
}

const MAX_METRICS = 25;

export const useTelemetryStore = create<TelemetryState>((set) => ({
  metrics: [],
  recordInteraction: (metric) =>
    set((state) => ({
      metrics: [metric, ...state.metrics].slice(0, MAX_METRICS)
    })),
  reset: () => set({ metrics: [] })
}));
