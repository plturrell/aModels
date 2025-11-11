/**
 * Job Monitor
 * 
 * Real-time job monitoring component with status updates
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  CircularProgress,
  LinearProgress,
  Typography,
  Tooltip,
  IconButton,
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import { getExtractJob, ExtractJob } from '../../api/extract';

interface JobMonitorProps {
  jobId: string;
  onStatusChange?: (status: string) => void;
}

export function JobMonitor({ jobId, onStatusChange }: JobMonitorProps) {
  const [job, setJob] = useState<ExtractJob | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadJobStatus();
    
    // Poll for status updates if job is still running
    const interval = setInterval(() => {
      if (job?.status === 'running' || job?.status === 'pending') {
        loadJobStatus();
      }
    }, 3000); // Poll every 3 seconds

    return () => clearInterval(interval);
  }, [jobId, job?.status]);

  const loadJobStatus = async () => {
    try {
      const jobData = await getExtractJob(jobId);
      setJob(jobData);
      setError(null);
      
      if (onStatusChange && jobData.status !== job?.status) {
        onStatusChange(jobData.status);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load job status');
    } finally {
      setLoading(false);
    }
  };

  if (loading && !job) {
    return <CircularProgress size={20} />;
  }

  if (error) {
    return (
      <Tooltip title={error}>
        <IconButton size="small" onClick={loadJobStatus}>
          <RefreshIcon />
        </IconButton>
      </Tooltip>
    );
  }

  if (!job) {
    return null;
  }

  if (job.status === 'running' || job.status === 'pending') {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <CircularProgress size={16} />
        <Typography variant="caption" color="textSecondary">
          {job.status === 'running' ? 'Running...' : 'Pending...'}
        </Typography>
      </Box>
    );
  }

  return null; // Don't show anything for completed/failed jobs
}

