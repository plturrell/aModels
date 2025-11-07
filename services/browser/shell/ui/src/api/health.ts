import { fetchJSON } from "./client";

export interface ServiceHealth {
  [serviceName: string]: string | { status: string; error?: string };
}

export interface HealthCheckResponse {
  gateway: string;
  [key: string]: string | { status: string; error?: string };
}

/**
 * Fetch health status of all services from the gateway
 */
export async function getServiceHealth(): Promise<HealthCheckResponse> {
  return fetchJSON<HealthCheckResponse>("/healthz");
}

/**
 * Check if a specific service is healthy
 */
export function isServiceHealthy(health: ServiceHealth, serviceName: string): boolean {
  const status = health[serviceName];
  if (typeof status === "string") {
    return status === "ok";
  }
  if (status && typeof status === "object") {
    return status.status === "ok";
  }
  return false;
}

/**
 * Get service status message
 */
export function getServiceStatus(health: ServiceHealth, serviceName: string): string {
  const status = health[serviceName];
  if (typeof status === "string") {
    return status;
  }
  if (status && typeof status === "object") {
    return status.status || "unknown";
  }
  return "unknown";
}

/**
 * Get service error message if available
 */
export function getServiceError(health: ServiceHealth, serviceName: string): string | null {
  const status = health[serviceName];
  if (status && typeof status === "object" && status.error) {
    return status.error;
  }
  return null;
}

