/**
 * SAP Connection Configuration View
 * 
 * Wizard for configuring SAP BDC connection
 */

import React, { useState } from 'react';
import {
  Box,
  TextField,
  Button,
  Stack,
  Alert,
  Typography,
  Card,
  CardContent,
  IconButton,
  InputAdornment,
} from '@mui/material';
import VisibilityIcon from '@mui/icons-material/Visibility';
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import { Panel } from '../../../components/Panel';
import type { SAPBDCConnectionConfig } from '../../../api/sap';

interface ConnectionConfigViewProps {
  connection: SAPBDCConnectionConfig;
  onConnectionChange: (config: SAPBDCConnectionConfig) => void;
  onTestConnection: () => Promise<void>;
  isConnected: boolean;
}

export function ConnectionConfigView({
  connection,
  onConnectionChange,
  onTestConnection,
  isConnected,
}: ConnectionConfigViewProps) {
  const [showToken, setShowToken] = useState(false);
  const [testing, setTesting] = useState(false);

  const handleFieldChange = (field: keyof SAPBDCConnectionConfig) => (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    onConnectionChange({
      ...connection,
      [field]: event.target.value,
    });
  };

  const handleTest = async () => {
    setTesting(true);
    try {
      await onTestConnection();
    } finally {
      setTesting(false);
    }
  };

  return (
    <Stack spacing={3}>
      <Panel title="SAP BDC Connection Configuration" dense>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          Configure your connection to SAP Business Data Cloud. All fields are required for extraction.
        </Typography>

        <Stack spacing={2}>
          <TextField
            label="SAP BDC Base URL"
            value={connection.base_url}
            onChange={handleFieldChange('base_url')}
            placeholder="https://your-bdc-instance.com"
            fullWidth
            required
            helperText="Base URL of your SAP Business Data Cloud instance"
          />

          <TextField
            label="API Token"
            type={showToken ? 'text' : 'password'}
            value={connection.api_token}
            onChange={handleFieldChange('api_token')}
            placeholder="Enter your API token"
            fullWidth
            required
            helperText="API token for authenticating with SAP BDC"
            InputProps={{
              endAdornment: (
                <InputAdornment position="end">
                  <IconButton
                    onClick={() => setShowToken(!showToken)}
                    edge="end"
                  >
                    {showToken ? <VisibilityOffIcon /> : <VisibilityIcon />}
                  </IconButton>
                </InputAdornment>
              ),
            }}
          />

          <TextField
            label="Formation ID"
            value={connection.formation_id}
            onChange={handleFieldChange('formation_id')}
            placeholder="formation-123"
            fullWidth
            required
            helperText="ID of the formation you want to extract from"
          />

          <TextField
            label="SAP Datasphere URL (Optional)"
            value={connection.datasphere_url || ''}
            onChange={handleFieldChange('datasphere_url')}
            placeholder="https://your-datasphere-instance.com"
            fullWidth
            helperText="Optional: URL for SAP Datasphere if different from BDC base URL"
          />

          <Box sx={{ display: 'flex', gap: 2, justifyContent: 'flex-end' }}>
            <Button
              variant="outlined"
              onClick={handleTest}
              disabled={testing || !connection.base_url || !connection.api_token || !connection.formation_id}
            >
              {testing ? 'Testing...' : 'Test Connection'}
            </Button>
          </Box>
        </Stack>
      </Panel>

      {isConnected && (
        <Alert
          icon={<CheckCircleIcon />}
          severity="success"
          action={
            <Button color="inherit" size="small" onClick={() => window.location.reload()}>
              Continue
            </Button>
          }
        >
          Connection successful! You can now browse data products and extract schemas.
        </Alert>
      )}

      {!isConnected && connection.base_url && (
        <Alert severity="info">
          Fill in all required fields and click "Test Connection" to verify your configuration.
        </Alert>
      )}

      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Configuration Help
          </Typography>
          <Typography variant="body2" color="text.secondary" component="div">
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              <li>
                <strong>SAP BDC Base URL:</strong> The base URL of your SAP Business Data Cloud instance.
                Typically looks like: <code>https://your-tenant.businessdata.cloud</code>
              </li>
              <li>
                <strong>API Token:</strong> Generate this from your SAP BDC Cockpit under API Management.
                Keep this token secure and never share it.
              </li>
              <li>
                <strong>Formation ID:</strong> The ID of the formation containing the data products
                you want to extract. You can find this in the SAP BDC Cockpit.
              </li>
              <li>
                <strong>SAP Datasphere URL:</strong> Only required if your Datasphere instance
                is hosted separately from your BDC instance.
              </li>
            </ul>
          </Typography>
        </CardContent>
      </Card>
    </Stack>
  );
}


