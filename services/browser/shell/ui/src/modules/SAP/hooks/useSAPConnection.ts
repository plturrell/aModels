/**
 * Hook for managing SAP BDC connection state
 */

import { useState, useCallback } from 'react';
import type { SAPBDCConnectionConfig } from '../../../api/sap';
import { testSAPConnection } from '../../../api/sap';

const CONNECTION_STORAGE_KEY = 'sap_bdc_connection';

export function useSAPConnection() {
  const [connection, setConnectionState] = useState<SAPBDCConnectionConfig>(() => {
    try {
      const stored = localStorage.getItem(CONNECTION_STORAGE_KEY);
      if (stored) {
        return JSON.parse(stored);
      }
    } catch {
      // Ignore parse errors
    }
    return {
      base_url: '',
      api_token: '',
      formation_id: '',
      datasphere_url: '',
    };
  });

  const [isConnected, setIsConnected] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);

  const setConnection = useCallback((config: SAPBDCConnectionConfig) => {
    setConnectionState(config);
    try {
      localStorage.setItem(CONNECTION_STORAGE_KEY, JSON.stringify(config));
    } catch {
      // Ignore storage errors
    }
    setIsConnected(false);
    setConnectionError(null);
  }, []);

  const testConnection = useCallback(async () => {
    if (!connection.base_url || !connection.api_token || !connection.formation_id) {
      setConnectionError('Please fill in all required connection fields');
      setIsConnected(false);
      return;
    }

    setConnectionError(null);
    try {
      const result = await testSAPConnection(connection);
      setIsConnected(result.success);
      if (!result.success) {
        setConnectionError(result.message);
      }
    } catch (error) {
      setIsConnected(false);
      setConnectionError(error instanceof Error ? error.message : 'Connection test failed');
    }
  }, [connection]);

  return {
    connection,
    setConnection,
    isConnected,
    connectionError,
    testConnection,
  };
}




