/**
 * Core Services Integration
 * Week 2: UX Excellence & Polish
 * Integrates framework, runtime, plot, and stdlib services
 */

// Service endpoints configuration
export const SERVICE_ENDPOINTS = {
  framework: {
    base: '/api/framework',
    analytics: '/api/framework/analytics',
    health: '/api/framework/health'
  },
  runtime: {
    base: '/api/runtime',
    orchestrator: '/api/runtime/orchestrator',
    rest: '/api/runtime/rest',
    health: '/api/runtime/health'
  },
  plot: {
    base: '/api/plot',
    dashboard: '/api/plot/dashboard',
    render: '/api/plot/render'
  },
  stdlib: {
    base: '/api/stdlib',
    analytics: '/api/stdlib/analyticsutil',
    utils: '/api/stdlib/utils'
  }
};

// Service health status
type ServiceHealth = {
  framework: boolean;
  runtime: boolean;
  plot: boolean;
  stdlib: boolean;
};

// Service capabilities
type ServiceCapabilities = {
  framework: {
    analytics: boolean;
    health: boolean;
  };
  runtime: {
    orchestrator: boolean;
    rest: boolean;
    health: boolean;
  };
  plot: {
    dashboard: boolean;
    render: boolean;
  };
  stdlib: {
    analytics: boolean;
    utils: boolean;
  };
};

// Integration configuration
export const INTEGRATION_CONFIG = {
  // Framework service
  framework: {
    name: 'aModels Framework',
    description: 'Core analytics and framework services',
    endpoints: SERVICE_ENDPOINTS.framework,
    capabilities: ['analytics', 'health', 'metrics'],
    priority: 'high'
  },
  
  // Runtime service  
  runtime: {
    name: 'aModels Runtime',
    description: 'Orchestration and REST services',
    endpoints: SERVICE_ENDPOINTS.runtime,
    capabilities: ['orchestrator', 'rest', 'health'],
    priority: 'high'
  },
  
  // Plot service
  plot: {
    name: 'aModels Plot',
    description: 'Visualization and dashboard services',
    endpoints: SERVICE_ENDPOINTS.plot,
    capabilities: ['dashboard', 'render', 'charts'],
    priority: 'medium'
  },
  
  // Standard library
  stdlib: {
    name: 'aModels Standard Library',
    description: 'Utility and analytics services',
    endpoints: SERVICE_ENDPOINTS.stdlib,
    capabilities: ['analytics', 'utils', 'helpers'],
    priority: 'medium'
  }
};

// Service health check
export const checkServiceHealth = async (): Promise<ServiceHealth> => {
  const health: ServiceHealth = {
    framework: false,
    runtime: false,
    plot: false,
    stdlib: false
  };

  try {
    // Check framework health
    const frameworkHealth = await fetch(`${SERVICE_ENDPOINTS.framework.health}`);
    health.framework = frameworkHealth.ok;
  } catch {
    health.framework = false;
  }

  try {
    // Check runtime health
    const runtimeHealth = await fetch(`${SERVICE_ENDPOINTS.runtime.health}`);
    health.runtime = runtimeHealth.ok;
  } catch {
    health.runtime = false;
  }

  try {
    // Check plot health
    const plotHealth = await fetch(`${SERVICE_ENDPOINTS.plot.base}/health`);
    health.plot = plotHealth.ok;
  } catch {
    health.plot = false;
  }

  try {
    // Check stdlib health
    const stdlibHealth = await fetch(`${SERVICE_ENDPOINTS.stdlib.base}/health`);
    health.stdlib = stdlibHealth.ok;
  } catch {
    health.stdlib = false;
  }

  return health;
};

// Service integration hooks
import { useState, useEffect } from 'react';

export const useServiceIntegration = () => {
  const [health, setHealth] = useState<ServiceHealth>({
    framework: false,
    runtime: false,
    plot: false,
    stdlib: false
  });

  const [capabilities, setCapabilities] = useState<ServiceCapabilities>({
    framework: { analytics: false, health: false },
    runtime: { orchestrator: false, rest: false, health: false },
    plot: { dashboard: false, render: false },
    stdlib: { analytics: false, utils: false }
  });

  useEffect(() => {
    const initializeServices = async () => {
      const healthStatus = await checkServiceHealth();
      setHealth(healthStatus);
      
      // Set capabilities based on health
      setCapabilities({
        framework: {
          analytics: healthStatus.framework,
          health: healthStatus.framework
        },
        runtime: {
          orchestrator: healthStatus.runtime,
          rest: healthStatus.runtime,
          health: healthStatus.runtime
        },
        plot: {
          dashboard: healthStatus.plot,
          render: healthStatus.plot
        },
        stdlib: {
          analytics: healthStatus.stdlib,
          utils: healthStatus.stdlib
        }
      });
    };

    initializeServices();
    
    // Poll health every 30 seconds
    const interval = setInterval(initializeServices, 30000);
    return () => clearInterval(interval);
  }, []);

  return {
    health,
    capabilities,
    config: INTEGRATION_CONFIG,
    refresh: checkServiceHealth
  };
};

// Service data types
export interface ServiceData {
  framework?: {
    analytics: any;
    health: any;
  };
  runtime?: {
    orchestrator: any;
    rest: any;
    health: any;
  };
  plot?: {
    dashboard: any;
    render: any;
  };
  stdlib?: {
    analytics: any;
    utils: any;
  };
}

// Service integration component
export class ServiceIntegration {
  private endpoints = SERVICE_ENDPOINTS;
  
  async fetchFrameworkAnalytics() {
    try {
      const response = await fetch(this.endpoints.framework.analytics);
      return response.json();
    } catch (error) {
      console.error('Framework analytics error:', error);
      return null;
    }
  }
  
  async fetchRuntimeOrchestrator() {
    try {
      const response = await fetch(this.endpoints.runtime.orchestrator);
      return response.json();
    } catch (error) {
      console.error('Runtime orchestrator error:', error);
      return null;
    }
  }
  
  async fetchPlotDashboard() {
    try {
      const response = await fetch(this.endpoints.plot.dashboard);
      return response.json();
    } catch (error) {
      console.error('Plot dashboard error:', error);
      return null;
    }
  }
  
  async fetchStdlibUtils() {
    try {
      const response = await fetch(this.endpoints.stdlib.utils);
      return response.json();
    } catch (error) {
      console.error('Stdlib utils error:', error);
      return null;
    }
  }
}

// Export for use in components
export const serviceIntegration = new ServiceIntegration();
