import React, { useEffect, useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  CircularProgress,
  Chip,
  Alert,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  CheckCircle,
  Error,
  Refresh,
  Analytics,
  Storage,
  Dashboard,
  Functions
} from '@mui/icons-material';
import { useServiceIntegration } from '../../services/ServiceIntegration';
import { INTEGRATION_CONFIG } from '../../services/ServiceIntegration';

export const ServiceDashboard: React.FC = () => {
  const { health, capabilities, config, refresh } = useServiceIntegration();
  const [loading, setLoading] = useState(false);

  const serviceIcons = {
    framework: <Analytics />,
    runtime: <Storage />,
    plot: <Dashboard />,
    stdlib: <Functions />
  };

  const handleRefresh = async () => {
    setLoading(true);
    await refresh();
    setLoading(false);
  };

  const getStatusColor = (status: boolean) => status ? 'success' : 'error';
  const getStatusIcon = (status: boolean) => status ? <CheckCircle /> : <Error />;

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Core Services Integration
        </Typography>
        <Tooltip title="Refresh services">
          <IconButton onClick={handleRefresh} disabled={loading}>
            <Refresh />
          </IconButton>
        </Tooltip>
      </Box>

      <Grid container spacing={3}>
        {Object.entries(config).map(([key, service]) => (
          <Grid item xs={12} sm={6} md={3} key={key}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  {React.cloneElement(serviceIcons[key as keyof typeof serviceIcons], {
                    sx: { fontSize: 40, color: health[key as keyof typeof health] ? 'success.main' : 'error.main' }
                  })}
                  <Box sx={{ ml: 2 }}>
                    <Typography variant="h6" component="h2">
                      {service.name}
                    </Typography>
                    <Chip
                      icon={getStatusIcon(health[key as keyof typeof health])}
                      label={health[key as keyof typeof health] ? 'Online' : 'Offline'}
                      color={getStatusColor(health[key as keyof typeof health])}
                      size="small"
                    />
                  </Box>
                </Box>

                <Typography variant="body2" color="text.secondary" gutterBottom>
                  {service.description}
                </Typography>

                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Capabilities:
                  </Typography>
                  {service.capabilities.map((capability) => (
                    <Chip
                      key={capability}
                      label={capability}
                      size="small"
                      variant="outlined"
                      sx={{ mr: 0.5, mb: 0.5 }}
                    />
                  ))}
                </Box>

                <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
                  Priority: {service.priority}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Box sx={{ mt: 4 }}>
        <Typography variant="h5" gutterBottom>
          Service Health Overview
        </Typography>
        <Grid container spacing={2}>
          {Object.entries(health).map(([service, status]) => (
            <Grid item xs={12} sm={6} key={service}>
              <Alert
                severity={status ? 'success' : 'error'}
                icon={getStatusIcon(status)}
                sx={{ display: 'flex', alignItems: 'center' }}
              >
                <Typography variant="body2">
                  {config[service as keyof typeof config].name}: {status ? 'Operational' : 'Offline'}
                </Typography>
              </Alert>
            </Grid>
          ))}
        </Grid>
      </Box>

      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
          <CircularProgress />
        </Box>
      )}
    </Box>
  );
};
