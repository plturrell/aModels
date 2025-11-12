/**
 * Training Module
 * 
 * Module for viewing training metrics, model performance, and training dashboards
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Tabs,
  Tab,
  CircularProgress,
  Alert,
  Card,
  CardContent,
} from '@mui/material';
import { GridLegacy as Grid } from '@mui/material';
import DashboardIcon from '@mui/icons-material/Dashboard';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import AssessmentIcon from '@mui/icons-material/Assessment';
import { fetchJSON } from '../../api/client';

const TRAINING_SERVICE_URL = import.meta.env.VITE_TRAINING_SERVICE_URL || 'http://localhost:8080';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`training-tabpanel-${index}`}
      aria-labelledby={`training-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

interface HealthResponse {
  status: string;
  service: string;
  timestamp: string;
  components?: Record<string, boolean>;
}

export function TrainingModule() {
  const [activeTab, setActiveTab] = useState(0);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        setLoading(true);
        setError(null);
        // Use full URL for training service since it's not proxied through gateway
        const response = await fetch(`${TRAINING_SERVICE_URL}/healthz`);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        const data = await response.json() as HealthResponse;
        setHealth(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to connect to training service');
        setHealth(null);
      } finally {
        setLoading(false);
      }
    };

    fetchHealth();
    const interval = setInterval(fetchHealth, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  return (
    <Box sx={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Paper sx={{ mb: 2 }}>
        <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
          <Typography variant="h4" gutterBottom>
            Training Service
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Monitor training metrics, model performance, and training pipelines
          </Typography>
        </Box>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs
            value={activeTab}
            onChange={handleTabChange}
            aria-label="training module tabs"
            variant="scrollable"
            scrollButtons="auto"
          >
            <Tab
              icon={<DashboardIcon />}
              iconPosition="start"
              label="Dashboard"
              id="training-tab-0"
              aria-controls="training-tabpanel-0"
            />
            <Tab
              icon={<TrendingUpIcon />}
              iconPosition="start"
              label="Metrics"
              id="training-tab-1"
              aria-controls="training-tabpanel-1"
            />
            <Tab
              icon={<AssessmentIcon />}
              iconPosition="start"
              label="Health"
              id="training-tab-2"
              aria-controls="training-tabpanel-2"
            />
          </Tabs>
        </Box>
      </Paper>

      <Box sx={{ flex: 1, overflow: 'auto' }}>
        <TabPanel value={activeTab} index={0}>
          <Box>
            <Typography variant="h6" gutterBottom>
              Training Dashboard
            </Typography>
            <Alert severity="info" sx={{ mb: 2 }}>
              Training dashboard interface. Connect to training service at {TRAINING_SERVICE_URL}
            </Alert>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Service Status
                    </Typography>
                    {loading ? (
                      <CircularProgress size={24} />
                    ) : health ? (
                      <Typography variant="body2" color={health.status === 'healthy' ? 'success.main' : 'error.main'}>
                        {health.status}
                      </Typography>
                    ) : (
                      <Typography variant="body2" color="error.main">
                        Service unavailable
                      </Typography>
                    )}
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Service URL
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {TRAINING_SERVICE_URL}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Box>
        </TabPanel>

        <TabPanel value={activeTab} index={1}>
          <Box>
            <Typography variant="h6" gutterBottom>
              Training Metrics
            </Typography>
            <Alert severity="info">
              Training metrics and performance data will be displayed here.
            </Alert>
          </Box>
        </TabPanel>

        <TabPanel value={activeTab} index={2}>
          <Box>
            <Typography variant="h6" gutterBottom>
              Service Health
            </Typography>
            {loading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                <CircularProgress />
              </Box>
            ) : error ? (
              <Alert severity="error">{error}</Alert>
            ) : health ? (
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Overall Status
                      </Typography>
                      <Typography variant="body1" color={health.status === 'healthy' ? 'success.main' : 'error.main'}>
                        {health.status}
                      </Typography>
                      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
                        Service: {health.service}
                      </Typography>
                      <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                        Last checked: {health.timestamp ? new Date(health.timestamp).toLocaleString() : 'Unknown'}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                {health.components && (
                  <Grid item xs={12}>
                    <Card>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          Components
                        </Typography>
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                          {Object.entries(health.components).map(([key, value]) => (
                            <Box key={key} sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <Typography variant="body2">{key}</Typography>
                              <Typography
                                variant="body2"
                                color={value ? 'success.main' : 'error.main'}
                              >
                                {value ? 'Available' : 'Unavailable'}
                              </Typography>
                            </Box>
                          ))}
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>
                )}
              </Grid>
            ) : (
              <Alert severity="warning">No health data available</Alert>
            )}
          </Box>
        </TabPanel>
      </Box>
    </Box>
  );
}

