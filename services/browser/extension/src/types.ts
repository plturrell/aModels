/**
 * TypeScript type definitions for aModels Browser Extension
 * Following Apple standards for type safety
 */

export interface Command {
  id: string;
  name: string;
  description: string;
  icon: string;
  keywords: string[];
  action: () => void | Promise<void>;
  category: 'Quick Actions' | 'Advanced' | 'Navigation';
}

export interface ConnectionStatus {
  connected: boolean;
  gatewayUrl: string;
  lastChecked: number;
  error?: string;
}

export interface StorageConfig {
  gatewayBaseUrl?: string;
  browserUrl?: string;
  setupCompleted?: boolean;
}

export interface ValidationResult {
  isValid: boolean;
  error?: string;
}

export interface ErrorInfo {
  code?: string;
  message: string;
  suggestion?: string;
  recoverable: boolean;
}

// Extend Window interface for global functions
declare global {
  interface Window {
    runOcr?: () => Promise<void>;
    runSql?: () => Promise<void>;
    runTelemetry?: () => Promise<void>;
    openBrowser?: () => Promise<void>;
    chatSend?: () => Promise<void>;
    runAgentFlow?: () => Promise<void>;
    runSearch?: () => Promise<void>;
    redisSet?: () => Promise<void>;
    redisGet?: () => Promise<void>;
    checkConnection?: () => void;
    toggleAdvanced?: () => void;
    commandPalette?: any;  // CommandPalette instance
    connectionManager?: any;  // ConnectionManager instance
  }
}

// Note: Class implementations are in separate files
// command-palette.ts exports CommandPalette class
// connection-manager.ts exports ConnectionManager class
