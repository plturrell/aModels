import { useMemo } from "react";
import {
  Box,
  Typography,
  Stack,
  Card,
  CardContent,
  Alert,
  CircularProgress,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper as MuiPaper
} from '@mui/material';
import { Panel } from "../../../components/Panel";
import type { ProcessingRequest } from "../../../api/perplexity";

interface AnalyticsViewProps {
  historyData: {
    requests: ProcessingRequest[];
    total: number;
  } | null;
  loading: boolean;
  error: Error | null;
}

export function AnalyticsView({ historyData, loading, error }: AnalyticsViewProps) {
  const analytics = useMemo(() => {
    if (!historyData?.requests) return null;

    const requests = historyData.requests;
    const completed = requests.filter(r => r.status === 'completed').length;
    const failed = requests.filter(r => r.status === 'failed').length;
    const processing = requests.filter(r => r.status === 'processing').length;
    const pending = requests.filter(r => r.status === 'pending').length;

    const avgProcessingTime = requests
      .filter(r => r.processing_time_ms)
      .reduce((sum, r) => sum + (r.processing_time_ms || 0), 0) / 
      requests.filter(r => r.processing_time_ms).length || 0;

    return {
      total: historyData.total,
      completed,
      failed,
      processing,
      pending,
      successRate: requests.length > 0 ? (completed / requests.length) * 100 : 0,
      avgProcessingTime: avgProcessingTime / 1000 // Convert to seconds
    };
  }, [historyData]);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 200 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error">
        Failed to load analytics: {error.message}
      </Alert>
    );
  }

  if (!historyData) {
    return (
      <Alert severity="info">
        No analytics data available yet. Process some documents to see analytics.
      </Alert>
    );
  }

  return (
    <Box>
      {/* Summary Cards */}
      <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} sx={{ mb: 3 }}>
        <Card sx={{ flex: 1 }}>
          <CardContent>
            <Typography variant="h4" color="primary" gutterBottom>
              {analytics?.total || 0}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Total Requests
            </Typography>
          </CardContent>
        </Card>
        <Card sx={{ flex: 1 }}>
          <CardContent>
            <Typography variant="h4" color="success.main" gutterBottom>
              {analytics?.completed || 0}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Completed
            </Typography>
          </CardContent>
        </Card>
        <Card sx={{ flex: 1 }}>
          <CardContent>
            <Typography variant="h4" color="error.main" gutterBottom>
              {analytics?.failed || 0}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Failed
            </Typography>
          </CardContent>
        </Card>
        <Card sx={{ flex: 1 }}>
          <CardContent>
            <Typography variant="h4" color="info.main" gutterBottom>
              {analytics?.successRate.toFixed(1) || 0}%
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Success Rate
            </Typography>
          </CardContent>
        </Card>
      </Stack>

      {/* Performance Metrics */}
      {analytics && analytics.avgProcessingTime > 0 && (
        <Panel title="Performance Metrics" dense>
          <Box>
            <Typography variant="body2" color="text.secondary">
              Average Processing Time
            </Typography>
            <Typography variant="h6">
              {analytics.avgProcessingTime.toFixed(2)}s
            </Typography>
          </Box>
        </Panel>
      )}

      {/* Recent Requests */}
      <Panel title="Recent Requests" subtitle={`Showing ${historyData.requests.length} of ${historyData.total}`}>
        {historyData.requests.length === 0 ? (
          <Alert severity="info">
            No requests found.
          </Alert>
        ) : (
          <TableContainer component={MuiPaper} variant="outlined">
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Request ID</TableCell>
                  <TableCell>Query</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Documents</TableCell>
                  <TableCell>Created</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {historyData.requests.slice(0, 20).map((request) => (
                  <TableRow key={request.request_id} hover>
                    <TableCell sx={{ fontFamily: 'monospace', fontSize: '0.875rem' }}>
                      {request.request_id.substring(0, 8)}...
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" noWrap sx={{ maxWidth: 200 }}>
                        {request.query}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip 
                        label={request.status} 
                        color={
                          request.status === "completed" ? "success" :
                          request.status === "failed" ? "error" :
                          request.status === "processing" ? "info" :
                          "default"
                        }
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      {request.statistics?.documents_processed || 0}
                    </TableCell>
                    <TableCell>
                      {new Date(request.created_at).toLocaleString()}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}
      </Panel>
    </Box>
  );
}

