/**
 * Extract Dashboard
 * 
 * Overview dashboard showing recent extractions, statistics, and quick actions
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  CircularProgress,
  Alert,
  Chip,
  Stack,
  LinearProgress,
} from '@mui/material';
import { GridLegacy as Grid } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import TimelineIcon from '@mui/icons-material/Timeline';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import { listExtractJobs, ExtractJob } from '../../api/extract';

interface ExtractDashboardProps {
  projectId?: string;
  systemId?: string;
}

export function ExtractDashboard({ projectId, systemId }: ExtractDashboardProps) {
  const [jobs, setJobs] = useState<ExtractJob[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadRecentJobs();
  }, []);

  const loadRecentJobs = async () => {
    setLoading(true);
    setError(null);
    try {
      const recentJobs = await listExtractJobs({ limit: 10 });
      setJobs(recentJobs);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load recent jobs');
    } finally {
      setLoading(false);
    }
  };

  const stats = {
    total: jobs.length,
    completed: jobs.filter(j => j.status === 'completed').length,
    running: jobs.filter(j => j.status === 'running').length,
    failed: jobs.filter(j => j.status === 'failed').length,
    successRate: jobs.length > 0
      ? (jobs.filter(j => j.status === 'completed').length / jobs.length) * 100
      : 0,
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 400 }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Statistics Cards */}
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom variant="body2">
                Total Extractions
              </Typography>
              <Typography variant="h4">{stats.total}</Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom variant="body2">
                Completed
              </Typography>
              <Typography variant="h4" color="success.main">
                {stats.completed}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom variant="body2">
                Running
              </Typography>
              <Typography variant="h4" color="info.main">
                {stats.running}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom variant="body2">
                Success Rate
              </Typography>
              <Typography variant="h4" color="primary.main">
                {stats.successRate.toFixed(1)}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Quick Actions */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Quick Actions
              </Typography>
              <Stack direction="row" spacing={2} sx={{ mt: 2 }}>
                <Button
                  variant="contained"
                  startIcon={<PlayArrowIcon />}
                  onClick={() => {
                    // This would typically navigate to the wizard tab
                    window.location.hash = '#extract-wizard';
                  }}
                >
                  New Extraction
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<TimelineIcon />}
                  onClick={() => {
                    window.location.hash = '#extract-graph-workflow';
                  }}
                >
                  Extract & Visualize
                </Button>
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Extractions */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">Recent Extractions</Typography>
                <Button size="small" onClick={loadRecentJobs}>
                  Refresh
                </Button>
              </Box>
              {jobs.length === 0 ? (
                <Typography color="textSecondary" sx={{ textAlign: 'center', py: 4 }}>
                  No extractions yet. Start a new extraction to get started.
                </Typography>
              ) : (
                <Stack spacing={2}>
                  {jobs.map((job) => (
                    <Box
                      key={job.id}
                      sx={{
                        p: 2,
                        border: 1,
                        borderColor: 'divider',
                        borderRadius: 1,
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                      }}
                    >
                      <Box sx={{ flex: 1 }}>
                        <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
                          <Typography variant="subtitle1">{job.source_type}</Typography>
                          <Chip
                            label={job.status}
                            size="small"
                            color={
                              job.status === 'completed'
                                ? 'success'
                                : job.status === 'failed'
                                ? 'error'
                                : job.status === 'running'
                                ? 'info'
                                : 'default'
                            }
                            icon={
                              job.status === 'completed' ? (
                                <CheckCircleIcon />
                              ) : job.status === 'failed' ? (
                                <ErrorIcon />
                              ) : undefined
                            }
                          />
                        </Stack>
                        <Typography variant="body2" color="textSecondary">
                          {new Date(job.created_at).toLocaleString()}
                        </Typography>
                        {job.error && (
                          <Alert severity="error" sx={{ mt: 1 }}>
                            {job.error}
                          </Alert>
                        )}
                      </Box>
                    </Box>
                  ))}
                </Stack>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

