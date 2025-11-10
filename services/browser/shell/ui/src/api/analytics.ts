import { useMemo } from "react";
import { ApiState, fetchJSON, useApiData } from "./client";

const ANALYTICS_BASE = "/api/runtime/analytics";

export interface PopularElement {
  element_id: string;
  element_name: string;
  access_count: number;
  last_accessed: string;
  trend: string;
}

export interface ActivityEvent {
  type: string;
  element_id: string;
  user_id: string;
  timestamp: string;
  details?: string;
}

export interface QualityTrend {
  element_id: string;
  element_name: string;
  current_score: number;
  trend: string;
  risk_level: string;
  last_updated: string;
}

export interface UserStat {
  user_id: string;
  access_count: number;
  last_access: string;
}

export interface UsageStatistics {
  total_accesses: number;
  unique_users: number;
  average_access_time: number;
  top_users: UserStat[];
  access_by_hour: Record<string, number> | Record<number, number>;
  access_by_day: Record<string, number>;
}

export interface Prediction {
  type: string;
  element_id?: string;
  title: string;
  description: string;
  confidence: number;
  predicted_at: string;
  value?: unknown;
}

export interface DashboardStats {
  total_data_elements: number;
  total_data_products: number;
  popular_elements: PopularElement[];
  recent_activity: ActivityEvent[];
  quality_trends: QualityTrend[];
  usage_statistics: UsageStatistics;
  predictions: Prediction[];
}

export interface ElementAnalytics {
  element_id: string;
  element_name: string;
  access_count: number;
  unique_users: number;
  average_quality: number;
  trend: string;
  recommendations: string[];
}

export interface TopElementsResponse {
  elements: PopularElement[];
  count: number;
}

export async function fetchDashboardStats(): Promise<DashboardStats> {
  return fetchJSON<DashboardStats>(`${ANALYTICS_BASE}/dashboard`);
}

export async function fetchElementAnalytics(elementId: string): Promise<ElementAnalytics> {
  return fetchJSON<ElementAnalytics>(`${ANALYTICS_BASE}/elements/${encodeURIComponent(elementId)}`);
}

export async function fetchTopElements(metric?: string, limit?: number): Promise<TopElementsResponse> {
  const params = new URLSearchParams();
  if (metric) {
    params.set("metric", metric);
  }
  if (typeof limit === "number" && limit > 0) {
    params.set("limit", String(limit));
  }
  const query = params.toString();
  const suffix = query ? `?${query}` : "";
  return fetchJSON<TopElementsResponse>(`${ANALYTICS_BASE}/top${suffix}`);
}

export function useDashboardStats(): ApiState<DashboardStats> {
  return useApiData<DashboardStats>(`${ANALYTICS_BASE}/dashboard`);
}

export function useTopElements(metric?: string, limit?: number): ApiState<TopElementsResponse> {
  const endpoint = useMemo(() => {
    const params = new URLSearchParams();
    if (metric) {
      params.set("metric", metric);
    }
    if (typeof limit === "number" && limit > 0) {
      params.set("limit", String(limit));
    }
    const query = params.toString();
    return `${ANALYTICS_BASE}/top${query ? `?${query}` : ""}`;
  }, [metric, limit]);

  return useApiData<TopElementsResponse>(endpoint);
}

export function useElementAnalytics(elementId: string | null | undefined): ApiState<ElementAnalytics> {
  if (!elementId) {
    return {
      data: null,
      loading: false,
      error: null,
      refresh: () => {
        /* no-op while idle */
      }
    };
  }

  return useApiData<ElementAnalytics>(
    `${ANALYTICS_BASE}/elements/${encodeURIComponent(elementId)}`,
    undefined,
    [elementId]
  );
}
