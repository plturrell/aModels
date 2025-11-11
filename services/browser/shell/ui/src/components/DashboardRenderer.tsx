/**
 * Dashboard Renderer Component
 * Renders charts from dashboard specifications
 */

import React from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Stack,
  Card,
  CardContent,
  Alert
} from '@mui/material';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  AreaChart,
  Area
} from 'recharts';

interface DashboardChart {
  type: string;
  title: string;
  data_source: string;
  x_axis?: string;
  y_axis?: string;
  config?: Record<string, unknown>;
}

interface DashboardMetric {
  label: string;
  value: string | number;
  format?: string;
}

interface DashboardSpecification {
  title?: string;
  description?: string;
  charts?: DashboardChart[];
  metrics?: DashboardMetric[];
  insights?: string[];
}

interface DashboardRendererProps {
  specification: DashboardSpecification;
  data?: Record<string, unknown>;
}

// Standard dashboard templates
const DASHBOARD_TEMPLATES = {
  source_distribution: {
    type: 'pie',
    title: 'Source Distribution',
    data_source: 'source_distribution'
  },
  score_statistics: {
    type: 'bar',
    title: 'Score Statistics',
    data_source: 'score_statistics'
  },
  timeline: {
    type: 'line',
    title: 'Timeline',
    data_source: 'timeline'
  },
  results_overview: {
    type: 'bar',
    title: 'Results Overview',
    data_source: 'results'
  }
};

// Color palette for charts
const CHART_COLORS = [
  '#0088FE', '#00C49F', '#FFBB28', '#FF8042',
  '#8884d8', '#82ca9d', '#ffc658', '#ff7300',
  '#8dd1e1', '#d084d0', '#ffb347', '#87ceeb'
];

export function DashboardRenderer({ specification, data }: DashboardRendererProps) {
  const { title, description, charts = [], metrics = [], insights = [] } = specification;

  // Transform data for charts
  const transformChartData = (chart: DashboardChart): any[] => {
    if (!data) return [];
    
    const source = chart.data_source;
    const sourceData = data[source];
    
    if (!sourceData) return [];
    
    // Handle different data source types
    if (source === 'source_distribution' && typeof sourceData === 'object') {
      return Object.entries(sourceData as Record<string, number>).map(([name, value]) => ({
        name,
        value
      }));
    }
    
    if (source === 'score_statistics' && typeof sourceData === 'object') {
      const stats = sourceData as Record<string, number>;
      return [
        { name: 'Average', value: stats.average || 0 },
        { name: 'Min', value: stats.min || 0 },
        { name: 'Max', value: stats.max || 0 }
      ];
    }
    
    if (source === 'timeline' && Array.isArray(sourceData)) {
      return (sourceData as any[]).map((item, idx) => ({
        index: idx,
        score: item.score || 0,
        timestamp: item.timestamp || idx,
        source: item.source || 'unknown'
      }));
    }
    
    if (Array.isArray(sourceData)) {
      return sourceData;
    }
    
    return [];
  };

  const renderChart = (chart: DashboardChart, index: number) => {
    const chartData = transformChartData(chart);
    
    if (chartData.length === 0) {
      return (
        <Paper key={index} variant="outlined" sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            {chart.title}
          </Typography>
          <Alert severity="info">No data available for this chart</Alert>
        </Paper>
      );
    }

    const chartType = chart.type?.toLowerCase() || 'bar';
    const xAxisKey = chart.x_axis || (chartData[0] ? Object.keys(chartData[0])[0] : 'name');
    const yAxisKey = chart.y_axis || (chartData[0] ? Object.keys(chartData[0])[1] : 'value');

    return (
      <Paper key={index} variant="outlined" sx={{ p: 2, height: '100%' }}>
        <Typography variant="h6" gutterBottom>
          {chart.title}
        </Typography>
        <Box sx={{ width: '100%', height: 300, mt: 2 }}>
          <ResponsiveContainer>
            {chartType === 'pie' && (
              <PieChart>
                <Pie
                  data={chartData}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  label
                >
                  {chartData.map((entry, idx) => (
                    <Cell key={`cell-${idx}`} fill={CHART_COLORS[idx % CHART_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            )}
            
            {chartType === 'bar' && (
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey={xAxisKey} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey={yAxisKey} fill={CHART_COLORS[0]} />
              </BarChart>
            )}
            
            {chartType === 'line' && (
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey={xAxisKey} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey={yAxisKey} stroke={CHART_COLORS[0]} />
              </LineChart>
            )}
            
            {chartType === 'area' && (
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey={xAxisKey} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Area type="monotone" dataKey={yAxisKey} stroke={CHART_COLORS[0]} fill={CHART_COLORS[0]} fillOpacity={0.6} />
              </AreaChart>
            )}
            
            {chartType === 'scatter' && (
              <ScatterChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey={xAxisKey} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Scatter dataKey={yAxisKey} fill={CHART_COLORS[0]} />
              </ScatterChart>
            )}
          </ResponsiveContainer>
        </Box>
      </Paper>
    );
  };

  const formatMetric = (metric: DashboardMetric): string => {
    const { value, format } = metric;
    
    if (format === 'percentage' && typeof value === 'number') {
      return `${(value * 100).toFixed(1)}%`;
    }
    
    if (format === 'currency' && typeof value === 'number') {
      return `$${value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    }
    
    if (typeof value === 'number') {
      return value.toLocaleString('en-US');
    }
    
    return String(value);
  };

  return (
    <Box>
      {title && (
        <Typography variant="h4" gutterBottom>
          {title}
        </Typography>
      )}
      
      {description && (
        <Typography variant="body2" color="text.secondary" paragraph>
          {description}
        </Typography>
      )}
      
      {metrics.length > 0 && (
        <Grid container spacing={2} sx={{ mb: 3 }}>
          {metrics.map((metric, idx) => (
            <Grid item xs={12} sm={6} md={3} key={idx}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="caption" color="text.secondary">
                    {metric.label}
                  </Typography>
                  <Typography variant="h5">
                    {formatMetric(metric)}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}
      
      {charts.length > 0 && (
        <Grid container spacing={2}>
          {charts.map((chart, idx) => (
            <Grid item xs={12} md={charts.length === 1 ? 12 : 6} key={idx}>
              {renderChart(chart, idx)}
            </Grid>
          ))}
        </Grid>
      )}
      
      {insights.length > 0 && (
        <Box sx={{ mt: 3 }}>
          <Typography variant="h6" gutterBottom>
            Key Insights
          </Typography>
          <Stack spacing={1}>
            {insights.map((insight, idx) => (
              <Alert key={idx} severity="info">
                {insight}
              </Alert>
            ))}
          </Stack>
        </Box>
      )}
      
      {charts.length === 0 && metrics.length === 0 && insights.length === 0 && (
        <Alert severity="info">
          No dashboard content available. Generate a dashboard to see visualizations.
        </Alert>
      )}
    </Box>
  );
}

