/**
 * SAP Extract Wizard
 * 
 * SAP-specific extraction wizard with BDC connection, formation, and data product selection
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  Alert,
  CircularProgress,
  Stack,
  Card,
  CardContent,
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import {
  listSAPDataProducts,
  getSAPFormation,
  testSAPConnection,
  SAPBDCConnectionConfig,
  SAPDataProduct,
  SAPFormation,
} from '../../api/sap';

interface SAPExtractWizardProps {
  projectId?: string;
  systemId?: string;
  onConfigChange?: (config: any) => void;
}

export function SAPExtractWizard({
  projectId,
  systemId,
  onConfigChange,
}: SAPExtractWizardProps) {
  const [connection, setConnection] = useState<SAPBDCConnectionConfig>({
    base_url: '',
    api_token: '',
    formation_id: '',
  });
  const [formation, setFormation] = useState<SAPFormation | null>(null);
  const [dataProducts, setDataProducts] = useState<SAPDataProduct[]>([]);
  const [selectedDataProduct, setSelectedDataProduct] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [testing, setTesting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [connectionTested, setConnectionTested] = useState(false);

  useEffect(() => {
    if (connectionTested && connection.formation_id) {
      loadFormation();
      loadDataProducts();
    }
  }, [connection.formation_id, connectionTested]);

  const handleTestConnection = async () => {
    setTesting(true);
    setError(null);

    try {
      const result = await testSAPConnection(connection);
      if (result.success) {
        setConnectionTested(true);
        if (onConfigChange) {
          onConfigChange({ connection, tested: true });
        }
      } else {
        setError(result.message);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Connection test failed');
    } finally {
      setTesting(false);
    }
  };

  const loadFormation = async () => {
    try {
      const formationData = await getSAPFormation();
      setFormation(formationData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load formation');
    }
  };

  const loadDataProducts = async () => {
    setLoading(true);
    try {
      const products = await listSAPDataProducts();
      setDataProducts(products);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data products');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        SAP Business Data Cloud Configuration
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Stack spacing={3} sx={{ mt: 2 }}>
        {/* Connection Configuration */}
        <Card>
          <CardContent>
            <Typography variant="subtitle1" gutterBottom>
              Connection Settings
            </Typography>
            <TextField
              fullWidth
              label="SAP BDC Base URL"
              value={connection.base_url}
              onChange={(e) =>
                setConnection((prev) => ({
                  ...prev,
                  base_url: e.target.value,
                }))
              }
              sx={{ mt: 2 }}
              placeholder="https://your-instance.businessdata.cloud"
            />
            <TextField
              fullWidth
              label="API Token"
              type="password"
              value={connection.api_token}
              onChange={(e) =>
                setConnection((prev) => ({
                  ...prev,
                  api_token: e.target.value,
                }))
              }
              sx={{ mt: 2 }}
            />
            <TextField
              fullWidth
              label="Formation ID"
              value={connection.formation_id}
              onChange={(e) =>
                setConnection((prev) => ({
                  ...prev,
                  formation_id: e.target.value,
                }))
              }
              sx={{ mt: 2 }}
            />
            <Button
              variant="outlined"
              onClick={handleTestConnection}
              disabled={testing || !connection.base_url || !connection.api_token || !connection.formation_id}
              startIcon={testing ? <CircularProgress size={20} /> : <RefreshIcon />}
              sx={{ mt: 2 }}
            >
              Test Connection
            </Button>
            {connectionTested && (
              <Alert severity="success" sx={{ mt: 2 }}>
                Connection successful! Formation: {formation?.name || connection.formation_id}
              </Alert>
            )}
          </CardContent>
        </Card>

        {/* Formation Info */}
        {formation && (
          <Card>
            <CardContent>
              <Typography variant="subtitle1" gutterBottom>
                Formation Information
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Name: {formation.name}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                ID: {formation.id}
              </Typography>
            </CardContent>
          </Card>
        )}

        {/* Data Product Selection */}
        {connectionTested && (
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="subtitle1">Data Products</Typography>
                <Button
                  size="small"
                  startIcon={<RefreshIcon />}
                  onClick={loadDataProducts}
                  disabled={loading}
                >
                  Refresh
                </Button>
              </Box>
              {loading ? (
                <CircularProgress />
              ) : dataProducts.length === 0 ? (
                <Typography color="textSecondary">No data products available</Typography>
              ) : (
                <FormControl fullWidth>
                  <InputLabel>Select Data Product</InputLabel>
                  <Select
                    value={selectedDataProduct}
                    label="Select Data Product"
                    onChange={(e) => {
                      setSelectedDataProduct(e.target.value);
                      if (onConfigChange) {
                        onConfigChange({
                          connection,
                          dataProductId: e.target.value,
                        });
                      }
                    }}
                  >
                    {dataProducts.map((product) => (
                      <MenuItem key={product.id} value={product.id}>
                        {product.name} {product.version ? `(v${product.version})` : ''}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              )}
            </CardContent>
          </Card>
        )}
      </Stack>
    </Box>
  );
}

