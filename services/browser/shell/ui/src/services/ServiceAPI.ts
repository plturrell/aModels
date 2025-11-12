/**
 * Service API Integration Layer
 * Provides unified API access to core services
 */

import axios from 'axios';

// Base configuration
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8080';

// Service-specific clients
const frameworkClient = axios.create({
  baseURL: `${API_BASE_URL}/api/framework`,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

const runtimeClient = axios.create({
  baseURL: `${API_BASE_URL}/api/runtime`,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

const plotClient = axios.create({
  baseURL: `${API_BASE_URL}/api/plot`,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

const stdlibClient = axios.create({
  baseURL: `${API_BASE_URL}/api/stdlib`,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Framework Service API
export const FrameworkAPI = {
  async getAnalytics() {
    try {
      const response = await frameworkClient.get('/analytics');
      return response.data;
    } catch (error) {
      console.error('Framework analytics error:', error);
      throw error;
    }
  },

  async getHealth() {
    try {
      const response = await frameworkClient.get('/health');
      return response.data;
    } catch (error) {
      console.error('Framework health error:', error);
      throw error;
    }
  },

  async getMetrics() {
    try {
      const response = await frameworkClient.get('/metrics');
      return response.data;
    } catch (error) {
      console.error('Framework metrics error:', error);
      throw error;
    }
  }
};

// Runtime Service API
export const RuntimeAPI = {
  async getOrchestratorStatus() {
    try {
      const response = await runtimeClient.get('/orchestrator/status');
      return response.data;
    } catch (error) {
      console.error('Runtime orchestrator error:', error);
      throw error;
    }
  },

  async getRESTServices() {
    try {
      const response = await runtimeClient.get('/rest/services');
      return response.data;
    } catch (error) {
      console.error('Runtime REST services error:', error);
      throw error;
    }
  },

  async getHealth() {
    try {
      const response = await runtimeClient.get('/health');
      return response.data;
    } catch (error) {
      console.error('Runtime health error:', error);
      throw error;
    }
  }
};

// Plot Service API
export const PlotAPI = {
  async getDashboard() {
    try {
      const response = await plotClient.get('/dashboard');
      return response.data;
    } catch (error) {
      console.error('Plot dashboard error:', error);
      throw error;
    }
  },

  async renderChart(chartData: any) {
    try {
      const response = await plotClient.post('/render', chartData);
      return response.data;
    } catch (error) {
      console.error('Plot render error:', error);
      throw error;
    }
  },

  async getAvailableCharts() {
    try {
      const response = await plotClient.get('/charts');
      return response.data;
    } catch (error) {
      console.error('Plot charts error:', error);
      throw error;
    }
  }
};

// Standard Library API
export const StdlibAPI = {
  async getAnalyticsUtil() {
    try {
      const response = await stdlibClient.get('/analyticsutil');
      return response.data;
    } catch (error) {
      console.error('Stdlib analytics util error:', error);
      throw error;
    }
  },

  async getUtils() {
    try {
      const response = await stdlibClient.get('/utils');
      return response.data;
    } catch (error) {
      console.error('Stdlib utils error:', error);
      throw error;
    }
  },

  async getHelpers() {
    try {
      const response = await stdlibClient.get('/helpers');
      return response.data;
    } catch (error) {
      console.error('Stdlib helpers error:', error);
      throw error;
    }
  }
};

// Unified service client
export const ServiceClient = {
  framework: FrameworkAPI,
  runtime: RuntimeAPI,
  plot: PlotAPI,
  stdlib: StdlibAPI,
  
  // Health check for all services
  async checkAllHealth() {
    const results = await Promise.allSettled([
      FrameworkAPI.getHealth(),
      RuntimeAPI.getHealth(),
      PlotAPI.getDashboard(),
      StdlibAPI.getUtils()
    ]);
    
    return {
      framework: results[0].status === 'fulfilled',
      runtime: results[1].status === 'fulfilled',
      plot: results[2].status === 'fulfilled',
      stdlib: results[3].status === 'fulfilled'
    };
  }
};

// Error handling utilities
export class ServiceError extends Error {
  constructor(
    public service: string,
    public statusCode: number,
    public details: any
  ) {
    super(`Service ${service} error: ${statusCode}`);
    this.name = 'ServiceError';
  }
}

// Response transformers
export const transformServiceResponse = (data: any, service: string) => {
  switch (service) {
    case 'framework':
      return {
        ...data,
        service: 'framework',
        timestamp: new Date().toISOString()
      };
    case 'runtime':
      return {
        ...data,
        service: 'runtime',
        timestamp: new Date().toISOString()
      };
    case 'plot':
      return {
        ...data,
        service: 'plot',
        timestamp: new Date().toISOString()
      };
    case 'stdlib':
      return {
        ...data,
        service: 'stdlib',
        timestamp: new Date().toISOString()
      };
    default:
      return data;
  }
};
