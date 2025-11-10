/**
 * useServiceContext Hook
 * Provides context sharing between services (analytics, search, LocalAI)
 * for seamless cross-service workflows
 */

import { create } from "zustand";

export interface ServiceContext {
  analytics?: {
    data?: any;
    query?: string;
    selectedMetrics?: string[];
  };
  search?: {
    query?: string;
    results?: any[];
    selectedResults?: string[];
  };
  localai?: {
    conversationId?: string;
    lastMessage?: string;
    context?: string;
  };
}

interface ServiceContextState {
  context: ServiceContext;
  setAnalyticsContext: (analytics: ServiceContext["analytics"]) => void;
  setSearchContext: (search: ServiceContext["search"]) => void;
  setLocalAIContext: (localai: ServiceContext["localai"]) => void;
  clearContext: () => void;
  getContextForLocalAI: () => string;
}

export const useServiceContext = create<ServiceContextState>((set, get) => ({
  context: {},
  
  setAnalyticsContext: (analytics) => {
    set((state) => ({
      context: {
        ...state.context,
        analytics
      }
    }));
  },
  
  setSearchContext: (search) => {
    set((state) => ({
      context: {
        ...state.context,
        search
      }
    }));
  },
  
  setLocalAIContext: (localai) => {
    set((state) => ({
      context: {
        ...state.context,
        localai
      }
    }));
  },
  
  clearContext: () => {
    set({ context: {} });
  },
  
  getContextForLocalAI: () => {
    const { context } = get();
    const parts: string[] = [];
    
    if (context.analytics) {
      parts.push("Analytics Context:");
      if (context.analytics.query) {
        parts.push(`- Current analytics query: "${context.analytics.query}"`);
      }
      if (context.analytics.selectedMetrics?.length) {
        parts.push(`- Selected metrics: ${context.analytics.selectedMetrics.join(", ")}`);
      }
      if (context.analytics.data) {
        const summary = JSON.stringify(context.analytics.data, null, 2).substring(0, 500);
        parts.push(`- Analytics data summary: ${summary}...`);
      }
    }
    
    if (context.search) {
      parts.push("\nSearch Context:");
      if (context.search.query) {
        parts.push(`- Current search query: "${context.search.query}"`);
      }
      if (context.search.results?.length) {
        parts.push(`- Found ${context.search.results.length} results`);
        const topResults = context.search.results.slice(0, 3).map((r, idx) => 
          `${idx + 1}. ${r.title || r.content?.substring(0, 100) || "Result"}`
        ).join("\n");
        parts.push(`- Top results:\n${topResults}`);
      }
      if (context.search.selectedResults?.length) {
        parts.push(`- Selected results: ${context.search.selectedResults.join(", ")}`);
      }
    }
    
    if (context.localai) {
      parts.push("\nLocalAI Context:");
      if (context.localai.lastMessage) {
        parts.push(`- Last message: "${context.localai.lastMessage.substring(0, 200)}"`);
      }
    }
    
    return parts.join("\n") || "No context available.";
  }
}));

