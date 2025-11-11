/**
 * Dashboard Renderer Component
 * Renders charts from dashboard specifications with interactive features
 */

import React, { useState, useCallback, useMemo, lazy, Suspense } from 'react';
import {
  Box,
  Paper,
  Typography,
  Stack,
  Card,
  CardContent,
  Alert,
  Button,
  Menu,
  MenuItem,
  TextField,
  IconButton,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  SelectChangeEvent,
  CircularProgress,
} from '@mui/material';
import { GridLegacy as Grid } from '@mui/material';
import { useScreenReaderAnnouncement } from '../../hooks/useAccessibility';
import { useKeyboardShortcuts, DASHBOARD_SHORTCUTS } from '../../hooks/useKeyboardShortcuts';
import { useDebounce } from '../../hooks/useDebounce';
import {
  DatePicker,
  LocalizationProvider
} from '@mui/x-date-pickers';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import ZoomInIcon from '@mui/icons-material/ZoomIn';
import ZoomOutIcon from '@mui/icons-material/ZoomOut';
import FilterListIcon from '@mui/icons-material/FilterList';
import DownloadIcon from '@mui/icons-material/Download';
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
  Area,
  Brush,
  ReferenceLine,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Treemap
} from 'recharts';
import { ResponsiveSankey } from '@nivo/sankey';
import { ResponsiveNetwork } from '@nivo/network';
// @ts-ignore - Module resolution issue
import * as exportUtils from '../../utils/export';

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
  onDrillDown?: (chartId: string, data: any) => void;
  onChartClick?: (chartId: string, data: any) => void;
  enableInteractivity?: boolean;
  linkedCharts?: string[]; // Chart IDs that should be linked
}

interface ChartState {
  zoomDomain?: [number, number];
  selectedData?: any;
  filteredData?: any[];
  dateRange?: [Date | null, Date | null];
  selectedMetrics?: string[];
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

export function DashboardRenderer({ 
  specification, 
  data,
  onDrillDown,
  onChartClick,
  enableInteractivity = true,
  linkedCharts = []
}: DashboardRendererProps) {
  const { title, description, charts = [], metrics = [], insights = [] } = specification;
  
  // Accessibility
  const announce = useScreenReaderAnnouncement();
  
  // Keyboard shortcuts
  useKeyboardShortcuts(DASHBOARD_SHORTCUTS);
  
  // State for interactivity
  const [chartStates, setChartStates] = useState<Record<string, ChartState>>({});
  const [selectedChart, setSelectedChart] = useState<string | null>(null);
  const [metricMenuAnchor, setMetricMenuAnchor] = useState<{ chartId: string; anchor: HTMLElement } | null>(null);
  const [dateRangeDialog, setDateRangeDialog] = useState<{ chartId: string; open: boolean } | null>(null);
  const [detailDialog, setDetailDialog] = useState<{ open: boolean; data: any; title: string } | null>(null);
  
  // Debounced search/filter
  const [searchQuery, setSearchQuery] = useState('');
  const debouncedSearchQuery = useDebounce(searchQuery, 300);
  
  // Transform data for charts
  const transformChartData = useCallback((chart: DashboardChart): any[] => {
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
  }, [data]);

  // Get available metrics from data
  const availableMetrics = useMemo(() => {
    if (!data) return [];
    const metricsSet = new Set<string>();
    charts.forEach(chart => {
      const chartData = transformChartData(chart);
      chartData.forEach(item => {
        Object.keys(item).forEach(key => {
          if (typeof item[key] === 'number') {
            metricsSet.add(key);
          }
        });
      });
    });
    return Array.from(metricsSet);
  }, [data, charts, transformChartData]);

  // Handle chart click for drill-down
  const handleChartClick = useCallback((chartId: string, data: any) => {
    if (onChartClick) {
      onChartClick(chartId, data);
    }
    if (onDrillDown) {
      onDrillDown(chartId, data);
    }
    if (enableInteractivity) {
      setDetailDialog({ open: true, data, title: `Details: ${chartId}` });
    }
  }, [onChartClick, onDrillDown, enableInteractivity]);

  // Handle brush change for filtering
  const handleBrushChange = useCallback((chartId: string, domain: { startIndex: number; endIndex: number } | null) => {
    if (!domain) return;
    setChartStates(prev => ({
      ...prev,
      [chartId]: {
        ...prev[chartId],
        zoomDomain: [domain.startIndex, domain.endIndex]
      }
    }));
    
    // Link other charts if specified
    if (linkedCharts.includes(chartId)) {
      linkedCharts.forEach(id => {
        if (id !== chartId) {
          setChartStates(prev => ({
            ...prev,
            [id]: {
              ...prev[id],
              zoomDomain: [domain.startIndex, domain.endIndex]
            }
          }));
        }
      });
    }
  }, [linkedCharts]);

  // Handle date range filter
  const handleDateRangeChange = useCallback((chartId: string, range: [Date | null, Date | null]) => {
    setChartStates(prev => ({
      ...prev,
      [chartId]: {
        ...prev[chartId],
        dateRange: range
      }
    }));
  }, []);

  // Handle metric selection
  const handleMetricSelection = useCallback((chartId: string, metrics: string[]) => {
    setChartStates(prev => ({
      ...prev,
      [chartId]: {
        ...prev[chartId],
        selectedMetrics: metrics
      }
    }));
    setMetricMenuAnchor(null);
  }, []);

  // Filter data based on state
  const getFilteredChartData = useCallback((chart: DashboardChart, chartId: string): any[] => {
    let chartData = transformChartData(chart);
    const state = chartStates[chartId];
    
    if (!state) return chartData;
    
    // Apply date range filter
    if (state.dateRange && state.dateRange[0] && state.dateRange[1]) {
      chartData = chartData.filter((item: any) => {
        const itemDate = item.timestamp ? new Date(item.timestamp) : null;
        if (!itemDate) return true;
        return itemDate >= state.dateRange![0]! && itemDate <= state.dateRange![1]!;
      });
    }
    
    // Apply zoom domain
    if (state.zoomDomain) {
      const [start, end] = state.zoomDomain;
      chartData = chartData.slice(start, end + 1);
    }
    
    return chartData;
  }, [chartStates, transformChartData]);

  const renderChart = (chart: DashboardChart, index: number) => {
    const chartId = `chart-${index}`;
    const chartData = getFilteredChartData(chart, chartId);
    const state = chartStates[chartId] || {};
    
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
    const selectedMetrics = state.selectedMetrics || [yAxisKey];

    return (
      <Paper 
        key={index} 
        id={chartId}
        variant="outlined" 
        sx={{ 
          p: 2, 
          height: '100%',
          position: 'relative',
          '&:hover': enableInteractivity ? {
            boxShadow: 3,
            cursor: 'pointer'
          } : {}
        }}
        onClick={() => enableInteractivity && setSelectedChart(chartId)}
        role="region"
        aria-label={`Chart: ${chart.title}`}
      >
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
          <Typography variant="h6" gutterBottom>
            {chart.title}
          </Typography>
          {enableInteractivity && (
            <Box sx={{ display: 'flex', gap: 0.5 }}>
              <IconButton
                size="small"
                onClick={(e) => {
                  e.stopPropagation();
                  setDateRangeDialog({ chartId, open: true });
                }}
                aria-label="Filter by date range"
              >
                <FilterListIcon fontSize="small" />
              </IconButton>
              <IconButton
                size="small"
                onClick={(e) => {
                  e.stopPropagation();
                  setMetricMenuAnchor({ chartId, anchor: e.currentTarget });
                }}
                aria-label="Select metrics"
              >
                <ZoomInIcon fontSize="small" />
              </IconButton>
              <IconButton
                size="small"
                onClick={async (e) => {
                  e.stopPropagation();
                  const chartElement = document.getElementById(chartId);
                  if (chartElement) {
                    const filename = chart.title.replace(/\s+/g, '_');
                    await exportUtils.exportChartToPNG(chartId, `${filename}.png`);
                  }
                }}
                aria-label="Export chart"
                title="Export chart"
              >
                <DownloadIcon fontSize="small" />
              </IconButton>
            </Box>
          )}
        </Box>
        <Box sx={{ width: '100%', height: 300, mt: 2 }}>
          <ResponsiveContainer>
            <>
            {chartType === 'pie' && (
              <PieChart
                onClick={(data) => enableInteractivity && handleChartClick(chartId, data)}
              >
                <Pie
                  data={chartData}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  label
                  onClick={(data) => enableInteractivity && handleChartClick(chartId, data)}
                >
                  {chartData.map((entry, idx) => (
                    <Cell 
                      key={`cell-${idx}`} 
                      fill={CHART_COLORS[idx % CHART_COLORS.length]}
                      style={{ cursor: enableInteractivity ? 'pointer' : 'default' }}
                    />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            )}
            {chartType === 'bar' && (
              <BarChart 
                data={chartData}
                onClick={(data) => enableInteractivity && handleChartClick(chartId, data)}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey={xAxisKey} />
                <YAxis />
                <Tooltip />
                <Legend />
                {selectedMetrics.map((metric, idx) => (
                  <Bar 
                    key={metric}
                    dataKey={metric} 
                    fill={CHART_COLORS[idx % CHART_COLORS.length]}
                    onClick={(data) => enableInteractivity && handleChartClick(chartId, data)}
                    style={{ cursor: enableInteractivity ? 'pointer' : 'default' }}
                  />
                ))}
                {enableInteractivity && chartData.length > 10 && (
                  <Brush 
                    dataKey={xAxisKey}
                    height={30}
                    onChange={(domain: { startIndex?: number; endIndex?: number } | null) => {
                      if (domain && domain.startIndex !== undefined && domain.endIndex !== undefined) {
                        handleBrushChange(chartId, { startIndex: domain.startIndex, endIndex: domain.endIndex });
                      }
                    }}
                  />
                )}
              </BarChart>
            )}
            {chartType === 'line' && (
              <LineChart 
                data={chartData}
                onClick={(data) => enableInteractivity && handleChartClick(chartId, data)}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey={xAxisKey} />
                <YAxis />
                <Tooltip />
                <Legend />
                {selectedMetrics.map((metric, idx) => (
                  <Line 
                    key={metric}
                    type="monotone" 
                    dataKey={metric} 
                    stroke={CHART_COLORS[idx % CHART_COLORS.length]}
                    dot={{ r: 4, onClick: (data) => enableInteractivity && handleChartClick(chartId, data) }}
                    style={{ cursor: enableInteractivity ? 'pointer' : 'default' }}
                  />
                ))}
                {enableInteractivity && chartData.length > 10 && (
                  <Brush 
                    dataKey={xAxisKey}
                    height={30}
                    onChange={(domain: { startIndex?: number; endIndex?: number } | null) => {
                      if (domain && domain.startIndex !== undefined && domain.endIndex !== undefined) {
                        handleBrushChange(chartId, { startIndex: domain.startIndex, endIndex: domain.endIndex });
                      }
                    }}
                  />
                )}
              </LineChart>
            )}
            {chartType === 'area' && (
              <AreaChart 
                data={chartData}
                onClick={(data) => enableInteractivity && handleChartClick(chartId, data)}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey={xAxisKey} />
                <YAxis />
                <Tooltip />
                <Legend />
                {selectedMetrics.map((metric, idx) => (
                  <Area 
                    key={metric}
                    type="monotone" 
                    dataKey={metric} 
                    stroke={CHART_COLORS[idx % CHART_COLORS.length]} 
                    fill={CHART_COLORS[idx % CHART_COLORS.length]} 
                    fillOpacity={0.6}
                    onClick={(data) => enableInteractivity && handleChartClick(chartId, data)}
                    style={{ cursor: enableInteractivity ? 'pointer' : 'default' }}
                  />
                ))}
                {enableInteractivity && chartData.length > 10 && (
                  <Brush 
                    dataKey={xAxisKey}
                    height={30}
                    onChange={(domain: { startIndex?: number; endIndex?: number } | null) => {
                      if (domain && domain.startIndex !== undefined && domain.endIndex !== undefined) {
                        handleBrushChange(chartId, { startIndex: domain.startIndex, endIndex: domain.endIndex });
                      }
                    }}
                  />
                )}
              </AreaChart>
            )}
            {chartType === 'scatter' && (
              <ScatterChart 
                data={chartData}
                onClick={(data) => enableInteractivity && handleChartClick(chartId, data)}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey={xAxisKey} />
                <YAxis />
                <Tooltip />
                <Legend />
                {selectedMetrics.map((metric, idx) => (
                  <Scatter 
                    key={metric}
                    dataKey={metric} 
                    fill={CHART_COLORS[idx % CHART_COLORS.length]}
                    onClick={(data) => enableInteractivity && handleChartClick(chartId, data)}
                    style={{ cursor: enableInteractivity ? 'pointer' : 'default' }}
                  />
                ))}
              </ScatterChart>
            )}
            {chartType === 'radar' && (
              <RadarChart data={chartData}>
                <PolarGrid />
                <PolarAngleAxis dataKey={xAxisKey} />
                <PolarRadiusAxis />
                {selectedMetrics.map((metric, idx) => (
                  <Radar
                    key={metric}
                    name={metric}
                    dataKey={metric}
                    stroke={CHART_COLORS[idx % CHART_COLORS.length]}
                    fill={CHART_COLORS[idx % CHART_COLORS.length]}
                    fillOpacity={0.6}
                  />
                ))}
                <Tooltip />
                <Legend />
              </RadarChart>
            )}
            {chartType === 'treemap' && (
              <Treemap
                data={chartData}
                dataKey={yAxisKey}
                nameKey={xAxisKey}
                aspectRatio={4/3}
                fill={CHART_COLORS[0]}
                onClick={(data) => enableInteractivity && handleChartClick(chartId, data)}
              />
            )}
            </>
          </ResponsiveContainer>
        </Box>
      </Paper>
    );
  };

  // Render advanced chart types that require special handling
  const renderAdvancedChart = (chart: DashboardChart, index: number) => {
    const chartId = `chart-${index}`;
    const chartData = getFilteredChartData(chart, chartId);
    const chartType = chart.type?.toLowerCase() || 'bar';

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

    // Sankey diagram
    if (chartType === 'sankey') {
      // Transform data for Sankey (expects nodes and links)
      const sankeyData = chart.config?.sankey || (chartData.length > 0 ? {
        nodes: chartData.map((item: any, idx: number) => ({
          id: item.source || item.name || `node-${idx}`,
          label: item.source || item.name || `Node ${idx}`
        })),
        links: chartData.map((item: any) => ({
          source: item.source || item.name,
          target: item.target || item.name,
          value: item.value || 1
        }))
      } : { nodes: [] as readonly any[], links: [] as readonly any[] });

      return (
        <Paper key={index} variant="outlined" sx={{ p: 2, height: '100%' }}>
          <Typography variant="h6" gutterBottom>
            {chart.title}
          </Typography>
          <Box sx={{ width: '100%', height: 400, mt: 2 }}>
            <ResponsiveSankey
              data={sankeyData}
              margin={{ top: 40, right: 160, bottom: 40, left: 50 }}
              align="justify"
              colors={{ scheme: 'category10' }}
              nodeOpacity={1}
              nodeHoverOthersOpacity={0.35}
              nodeThickness={18}
              nodeSpacing={24}
              nodeBorderWidth={0}
              nodeBorderColor={{ from: 'color', modifiers: [['darker', 0.8]] }}
              linkOpacity={0.5}
              linkHoverOthersOpacity={0.1}
              linkContract={3}
              enableLinkGradient={true}
              labelPosition="outside"
              labelOrientation="vertical"
              labelPadding={16}
              labelTextColor={{ from: 'color', modifiers: [['darker', 1]] }}
              animate={true}
              motionConfig="gentle"
            />
          </Box>
        </Paper>
      );
    }

    // Network graph
    if (chartType === 'network') {
      const networkData: { nodes: Array<{ id: string; label: string; size: number }>; links: Array<{ source: string; target: string; distance: number }> } = (chart.config?.network as any) || (chartData.length > 0 ? {
        nodes: chartData.map((item: any, idx: number) => ({
          id: item.id || item.name || `node-${idx}`,
          label: item.label || item.name || `Node ${idx}`,
          size: item.size || item.value || 10
        })),
        links: chartData.map((item: any) => ({
          source: item.source || item.from,
          target: item.target || item.to,
          distance: item.distance || item.value || 50
        }))
      } : { nodes: [] as readonly any[], links: [] as readonly any[] });

      return (
        <Paper key={index} variant="outlined" sx={{ p: 2, height: '100%' }}>
          <Typography variant="h6" gutterBottom>
            {chart.title}
          </Typography>
          <Box sx={{ width: '100%', height: 400, mt: 2 }}>
            <ResponsiveNetwork
              data={networkData}
              margin={{ top: 0, right: 0, bottom: 0, left: 0 }}
              linkDistance={50}
              centeringStrength={0.3}
              repulsivity={6}
              nodeSize={10}
              activeNodeSize={20}
              inactiveNodeSize={5}
              nodeColor={(node: any) => node.color || CHART_COLORS[0]}
              nodeBorderWidth={2}
              nodeBorderColor={{ from: 'color', modifiers: [['darker', 0.5]] }}
              linkThickness={2}
              linkColor={{ from: 'source.color', modifiers: [] }}
              motionConfig="gentle"
            />
          </Box>
        </Paper>
      );
    }

    // Heatmap (using Recharts with custom rendering)
    if (chartType === 'heatmap') {
      return (
        <Paper key={index} variant="outlined" sx={{ p: 2, height: '100%' }}>
          <Typography variant="h6" gutterBottom>
            {chart.title}
          </Typography>
          <Box sx={{ width: '100%', height: 300, mt: 2 }}>
            <ResponsiveContainer>
              <ScatterChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey={chart.x_axis || 'x'} 
                  type="category"
                  allowDuplicatedCategory={false}
                />
                <YAxis 
                  dataKey={chart.y_axis || 'y'} 
                  type="category"
                  allowDuplicatedCategory={false}
                />
                <Tooltip 
                  cursor={{ strokeDasharray: '3 3' }}
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      return (
                        <Box sx={{ bgcolor: 'background.paper', p: 1, border: 1, borderColor: 'divider' }}>
                          <Typography variant="body2">
                            {`${chart.x_axis || 'x'}: ${payload[0].payload[chart.x_axis || 'x']}`}
                          </Typography>
                          <Typography variant="body2">
                            {`${chart.y_axis || 'y'}: ${payload[0].payload[chart.y_axis || 'y']}`}
                          </Typography>
                          <Typography variant="body2" fontWeight="bold">
                            Value: {payload[0].payload.value || payload[0].value}
                          </Typography>
                        </Box>
                      );
                    }
                    return null;
                  }}
                />
                <Scatter
                  dataKey="value"
                  fill={CHART_COLORS[0]}
                  onClick={(data) => enableInteractivity && handleChartClick(chartId, data)}
                />
              </ScatterChart>
            </ResponsiveContainer>
          </Box>
        </Paper>
      );
    }

    return null;
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

  // Announce dashboard load to screen readers
  React.useEffect(() => {
    if (title) {
      announce(`Dashboard ${title} loaded with ${charts.length} charts`, 'polite');
    }
  }, [title, charts.length, announce]);

  return (
    <Box 
      role="main"
      aria-label={title || 'Dashboard'}
      id="main-content"
      className="keyboard-navigation"
    >
      {title && (
        <Typography 
          variant="h4" 
          gutterBottom
          component="h1"
          id="dashboard-title"
        >
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
          {charts.map((chart, idx) => {
            const chartType = chart.type?.toLowerCase() || 'bar';
            const isAdvancedChart = ['sankey', 'network', 'heatmap'].includes(chartType);
            
            return (
              <Grid item xs={12} md={charts.length === 1 ? 12 : 6} key={idx}>
                {isAdvancedChart ? renderAdvancedChart(chart, idx) : renderChart(chart, idx)}
                </Grid>
            );
          })}
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

      {/* Date Range Filter Dialog */}
      {dateRangeDialog && (
        <LocalizationProvider dateAdapter={AdapterDateFns}>
          <Dialog 
            open={dateRangeDialog.open} 
            onClose={() => setDateRangeDialog(null)}
            aria-labelledby="date-range-dialog-title"
          >
            <DialogTitle id="date-range-dialog-title">Filter by Date Range</DialogTitle>
            <DialogContent>
              <Stack spacing={2} sx={{ mt: 1 }}>
                <DatePicker
                  label="Start Date"
                  value={chartStates[dateRangeDialog.chartId]?.dateRange?.[0] || null}
                  onChange={(date: Date | null) => {
                    const currentRange = chartStates[dateRangeDialog.chartId]?.dateRange || [null, null];
                    handleDateRangeChange(dateRangeDialog.chartId, [date, currentRange[1]]);
                  }}
                  slotProps={{ textField: { fullWidth: true } }}
                />
                <DatePicker
                  label="End Date"
                  value={chartStates[dateRangeDialog.chartId]?.dateRange?.[1] || null}
                  onChange={(date: Date | null) => {
                    const currentRange = chartStates[dateRangeDialog.chartId]?.dateRange || [null, null];
                    handleDateRangeChange(dateRangeDialog.chartId, [currentRange[0], date]);
                  }}
                  slotProps={{ textField: { fullWidth: true } }}
                />
              </Stack>
            </DialogContent>
            <DialogActions>
              <Button onClick={() => {
                handleDateRangeChange(dateRangeDialog.chartId, [null, null]);
                setDateRangeDialog(null);
              }}>
                Clear
              </Button>
              <Button onClick={() => setDateRangeDialog(null)}>Close</Button>
            </DialogActions>
          </Dialog>
        </LocalizationProvider>
      )}

      {/* Metric Selection Menu */}
      {metricMenuAnchor && (
        <Menu
          anchorEl={metricMenuAnchor.anchor}
          open={true}
          onClose={() => setMetricMenuAnchor(null)}
        >
          {availableMetrics.map((metric) => (
            <MenuItem
              key={metric}
              selected={chartStates[metricMenuAnchor.chartId]?.selectedMetrics?.includes(metric)}
              onClick={() => {
                const current = chartStates[metricMenuAnchor.chartId]?.selectedMetrics || [];
                const newSelection = current.includes(metric)
                  ? current.filter(m => m !== metric)
                  : [...current, metric];
                handleMetricSelection(metricMenuAnchor.chartId, newSelection);
              }}
            >
              {metric}
            </MenuItem>
          ))}
        </Menu>
      )}

      {/* Detail Dialog */}
      {detailDialog && (
        <Dialog
          open={detailDialog.open}
          onClose={() => setDetailDialog(null)}
          maxWidth="md"
          fullWidth
          aria-labelledby="detail-dialog-title"
        >
          <DialogTitle id="detail-dialog-title">{detailDialog.title}</DialogTitle>
          <DialogContent>
            <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '0.875rem' }}>
              {JSON.stringify(detailDialog.data, null, 2)}
            </pre>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setDetailDialog(null)}>Close</Button>
          </DialogActions>
        </Dialog>
      )}
    </Box>
  );
}

