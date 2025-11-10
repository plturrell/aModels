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
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
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
  const [realTimeEnabled, setRealTimeEnabled] = useState(false);
  const [wsUrl] = useState(() => {
    // Construct WebSocket URL from current location
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    return `${protocol}//${host}/api/runtime/analytics/ws`;
  });

  // WebSocket connection for real-time updates
  const { connected: wsConnected } = useAnalyticsWebSocket(
    realTimeEnabled ? wsUrl : null,
    {
      onMessage: (message) => {
        // Handle real-time dashboard updates
        if (message.type === 'dashboard_update' && message.stats) {
          // Update analytics data if needed
          console.log('Received dashboard update:', message);
        }
      },
    }
  );

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

    // Prepare chart data
    const statusDistribution = [
      { name: 'Completed', value: completed, color: '#4caf50' },
      { name: 'Failed', value: failed, color: '#f44336' },
      { name: 'Processing', value: processing, color: '#2196f3' },
      { name: 'Pending', value: pending, color: '#ff9800' }
    ];

    // Timeline data (last 30 requests)
    const timelineData = requests
      .slice(-30)
      .map((r, idx) => ({
        index: idx,
        timestamp: new Date(r.created_at).toLocaleTimeString(),
        processingTime: (r.processing_time_ms || 0) / 1000,
        documents: r.statistics?.documents_processed || 0,
        status: r.status
      }));

    // Processing time distribution
    const processingTimeData = requests
      .filter(r => r.processing_time_ms)
      .reduce((acc, r) => {
        const timeBucket = Math.floor((r.processing_time_ms || 0) / 1000 / 5) * 5; // 5 second buckets
        acc[timeBucket] = (acc[timeBucket] || 0) + 1;
        return acc;
      }, {} as Record<number, number>);

    const processingTimeChart = Object.entries(processingTimeData)
      .map(([time, count]) => ({
        timeRange: `${time}-${time + 5}s`,
        count
      }))
      .sort((a, b) => parseInt(a.timeRange) - parseInt(b.timeRange));

    return {
      total: historyData.total,
      completed,
      failed,
      processing,
      pending,
      successRate: requests.length > 0 ? (completed / requests.length) * 100 : 0,
      avgProcessingTime: avgProcessingTime / 1000, // Convert to seconds
      statusDistribution,
      timelineData,
      processingTimeChart
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
      {/* Real-time Toggle */}
      <Box sx={{ mb: 2, display: 'flex', justifyContent: 'flex-end', alignItems: 'center', gap: 1 }}>
        <FormControlLabel
          control={
            <Switch
              checked={realTimeEnabled}
              onChange={(e) => setRealTimeEnabled(e.target.checked)}
              color="primary"
            />
          }
          label="Real-time Updates"
        />
        {wsConnected && (
          <Chip
            label="Connected"
            color="success"
            size="small"
            sx={{ ml: 1 }}
          />
        )}
      </Box>

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

      {/* Status Distribution Chart */}
      {analytics && analytics.statusDistribution && (
        <Panel title="Status Distribution" dense>
          <Box sx={{ width: '100%', height: 300, mt: 2 }}>
            <ResponsiveContainer>
              <PieChart>
                <Pie
                  data={analytics.statusDistribution}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  label
                >
                  {analytics.statusDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </Box>
        </Panel>
      )}

      {/* Timeline Chart */}
      {analytics && analytics.timelineData && analytics.timelineData.length > 0 && (
        <Panel title="Processing Timeline" dense>
          <Box sx={{ width: '100%', height: 300, mt: 2 }}>
            <ResponsiveContainer>
              <LineChart data={analytics.timelineData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timestamp" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="processingTime" stroke="#8884d8" name="Processing Time (s)" />
                <Line type="monotone" dataKey="documents" stroke="#82ca9d" name="Documents" />
              </LineChart>
            </ResponsiveContainer>
          </Box>
        </Panel>
      )}

      {/* Processing Time Distribution */}
      {analytics && analytics.processingTimeChart && analytics.processingTimeChart.length > 0 && (
        <Panel title="Processing Time Distribution" dense>
          <Box sx={{ width: '100%', height: 300, mt: 2 }}>
            <ResponsiveContainer>
              <BarChart data={analytics.processingTimeChart}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timeRange" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="count" fill="#8884d8" name="Request Count" />
              </BarChart>
            </ResponsiveContainer>
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

