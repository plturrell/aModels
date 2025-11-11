/**
 * Phase 3.2: Graph Analytics Dashboard View
 * 
 * Displays graph metrics, community detection, centrality, and growth trends
 */

import React, { useState, useCallback, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Tabs,
  Tab,
  Card,
  CardContent,
  Alert,
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  List,
  ListItem,
  ListItemText,
  Chip,
  LinearProgress,
} from '@mui/material';
import { GridLegacy as Grid } from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import PeopleIcon from '@mui/icons-material/People';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { getGraphStats } from '../../../api/graph';
import { detectCommunities, getCentrality, getGrowthTrends } from '../../../api/graphAnalytics';

export interface AnalyticsProps {
  projectId: string;
  systemId?: string;
}

export function Analytics({ projectId, systemId }: AnalyticsProps) {
  const [activeTab, setActiveTab] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Stats state
  const [stats, setStats] = useState<any>(null);

  // Community detection state
  const [communities, setCommunities] = useState<any>(null);
  const [communityAlgorithm, setCommunityAlgorithm] = useState<'louvain' | 'leiden' | 'label_propagation'>('louvain');

  // Centrality state
  const [centrality, setCentrality] = useState<any>(null);
  const [centralityType, setCentralityType] = useState<'degree' | 'betweenness' | 'closeness' | 'pagerank'>('degree');
  const [topK, setTopK] = useState(20);

  // Growth trends state
  const [growthTrends, setGrowthTrends] = useState<any>(null);
  const [days, setDays] = useState(30);

  // Load stats on mount
  useEffect(() => {
    if (projectId) {
      loadStats();
    }
  }, [projectId, systemId]);

  const loadStats = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const statsData = await getGraphStats(projectId, systemId);
      setStats(statsData);
    } catch (err: any) {
      setError(err.message || 'Failed to load stats');
      console.error('Stats error:', err);
    } finally {
      setLoading(false);
    }
  }, [projectId, systemId]);

  const loadCommunities = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const result = await detectCommunities(projectId, systemId, communityAlgorithm);
      setCommunities(result);
    } catch (err: any) {
      setError(err.message || 'Failed to detect communities');
      console.error('Community detection error:', err);
    } finally {
      setLoading(false);
    }
  }, [projectId, systemId, communityAlgorithm]);

  const loadCentrality = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const result = await getCentrality(projectId, systemId, centralityType, topK);
      setCentrality(result);
    } catch (err: any) {
      setError(err.message || 'Failed to calculate centrality');
      console.error('Centrality error:', err);
    } finally {
      setLoading(false);
    }
  }, [projectId, systemId, centralityType, topK]);

  const loadGrowthTrends = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const result = await getGrowthTrends(projectId, systemId, days);
      setGrowthTrends(result);
    } catch (err: any) {
      setError(err.message || 'Failed to load growth trends');
      console.error('Growth trends error:', err);
    } finally {
      setLoading(false);
    }
  }, [projectId, systemId, days]);

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">Graph Analytics</Typography>
        <Button
          variant="outlined"
          startIcon={<RefreshIcon />}
          onClick={loadStats}
          disabled={loading}
        >
          Refresh
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Tabs value={activeTab} onChange={(_, v) => setActiveTab(v)} sx={{ mb: 3 }}>
        <Tab label="Overview" />
        <Tab label="Communities" />
        <Tab label="Centrality" />
        <Tab label="Growth Trends" />
      </Tabs>

      {/* Overview Tab */}
      {activeTab === 0 && (
        <Grid container spacing={2}>
          {stats ? (
            <>
              <Grid item xs={12} md={3}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Total Nodes
                    </Typography>
                    <Typography variant="h4">
                      {stats.total_nodes?.toLocaleString() || 0}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={3}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Total Edges
                    </Typography>
                    <Typography variant="h4">
                      {stats.total_edges?.toLocaleString() || 0}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={3}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Density
                    </Typography>
                    <Typography variant="h4">
                      {stats.density ? stats.density.toFixed(4) : '0.0000'}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={3}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Avg Degree
                    </Typography>
                    <Typography variant="h4">
                      {stats.average_degree ? stats.average_degree.toFixed(2) : '0.00'}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Node Types
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 2 }}>
                      {Object.entries(stats.node_types || {}).map(([type, count]) => (
                        <Chip
                          key={type}
                          label={`${type}: ${count}`}
                          color="primary"
                          variant="outlined"
                        />
                      ))}
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Edge Types
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 2 }}>
                      {Object.entries(stats.edge_types || {}).map(([type, count]) => (
                        <Chip
                          key={type}
                          label={`${type}: ${count}`}
                          color="secondary"
                          variant="outlined"
                        />
                      ))}
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            </>
          ) : (
            <Grid item xs={12}>
              <Paper sx={{ p: 4, textAlign: 'center' }}>
                {loading ? (
                  <CircularProgress />
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    Click "Refresh" to load graph statistics
                  </Typography>
                )}
              </Paper>
            </Grid>
          )}
        </Grid>
      )}

      {/* Communities Tab */}
      {activeTab === 1 && (
        <Grid container spacing={2}>
          <Grid item xs={12} md={3}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Community Detection
              </Typography>
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Algorithm</InputLabel>
                <Select
                  value={communityAlgorithm}
                  label="Algorithm"
                  onChange={(e) => setCommunityAlgorithm(e.target.value as any)}
                  size="small"
                >
                  <MenuItem value="louvain">Louvain</MenuItem>
                  <MenuItem value="leiden">Leiden</MenuItem>
                  <MenuItem value="label_propagation">Label Propagation</MenuItem>
                </Select>
              </FormControl>
              <Button
                fullWidth
                variant="contained"
                onClick={loadCommunities}
                disabled={loading}
                startIcon={loading ? <CircularProgress size={16} /> : <PeopleIcon />}
              >
                Detect Communities
              </Button>
            </Paper>
          </Grid>
          <Grid item xs={12} md={9}>
            {communities ? (
              <Paper sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Communities ({communities.num_communities})
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  Algorithm: {communities.algorithm} • Total Nodes: {communities.total_nodes}
                </Typography>
                <List>
                  {communities.communities.slice(0, 20).map((community: any, idx: number) => (
                    <ListItem key={idx}>
                      <ListItemText
                        primary={
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Typography variant="body2">Community {idx + 1}</Typography>
                            <Chip label={`${community.size} nodes`} size="small" />
                          </Box>
                        }
                      />
                    </ListItem>
                  ))}
                </List>
              </Paper>
            ) : (
              <Paper sx={{ p: 4, textAlign: 'center' }}>
                <Typography variant="body2" color="text.secondary">
                  Click "Detect Communities" to analyze graph communities
                </Typography>
              </Paper>
            )}
          </Grid>
        </Grid>
      )}

      {/* Centrality Tab */}
      {activeTab === 2 && (
        <Grid container spacing={2}>
          <Grid item xs={12} md={3}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Centrality Metrics
              </Typography>
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Metric Type</InputLabel>
                <Select
                  value={centralityType}
                  label="Metric Type"
                  onChange={(e) => setCentralityType(e.target.value as any)}
                  size="small"
                >
                  <MenuItem value="degree">Degree</MenuItem>
                  <MenuItem value="betweenness">Betweenness</MenuItem>
                  <MenuItem value="closeness">Closeness</MenuItem>
                  <MenuItem value="pagerank">PageRank</MenuItem>
                </Select>
              </FormControl>
              <TextField
                fullWidth
                label="Top K"
                type="number"
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value) || 20)}
                sx={{ mb: 2 }}
                size="small"
              />
              <Button
                fullWidth
                variant="contained"
                onClick={loadCentrality}
                disabled={loading}
                startIcon={loading ? <CircularProgress size={16} /> : <ShowChartIcon />}
              >
                Calculate Centrality
              </Button>
            </Paper>
          </Grid>
          <Grid item xs={12} md={9}>
            {centrality ? (
              <Paper sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Top {centrality.top_k} Nodes by {centrality.metric_type}
                </Typography>
                <Box sx={{ mt: 2, maxHeight: 500, overflow: 'auto' }}>
                  {centrality.nodes.map((node: any, idx: number) => (
                    <Box key={idx} sx={{ mb: 2 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                        <Typography variant="body2">
                          {idx + 1}. {node.label || node.node_id}
                        </Typography>
                        <Chip
                          label={node.centrality.toFixed(2)}
                          size="small"
                          color="primary"
                        />
                      </Box>
                      <LinearProgress
                        variant="determinate"
                        value={(node.centrality / (centrality.nodes[0]?.centrality || 1)) * 100}
                        sx={{ height: 6, borderRadius: 1 }}
                      />
                    </Box>
                  ))}
                </Box>
              </Paper>
            ) : (
              <Paper sx={{ p: 4, textAlign: 'center' }}>
                <Typography variant="body2" color="text.secondary">
                  Click "Calculate Centrality" to analyze node importance
                </Typography>
              </Paper>
            )}
          </Grid>
        </Grid>
      )}

      {/* Growth Trends Tab */}
      {activeTab === 3 && (
        <Grid container spacing={2}>
          <Grid item xs={12} md={3}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Growth Trends
              </Typography>
              <TextField
                fullWidth
                label="Days"
                type="number"
                value={days}
                onChange={(e) => setDays(parseInt(e.target.value) || 30)}
                sx={{ mb: 2 }}
                size="small"
              />
              <Button
                fullWidth
                variant="contained"
                onClick={loadGrowthTrends}
                disabled={loading}
                startIcon={loading ? <CircularProgress size={16} /> : <TrendingUpIcon />}
              >
                Load Trends
              </Button>
            </Paper>
          </Grid>
          <Grid item xs={12} md={9}>
            {growthTrends ? (
              <Paper sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Growth Trends ({growthTrends.days} days)
                </Typography>
                {growthTrends.trends && growthTrends.trends.length > 0 ? (
                  <Box sx={{ mt: 2, height: 400 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={growthTrends.trends}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Line type="monotone" dataKey="count" stroke="#8884d8" name="Node Count" />
                      </LineChart>
                    </ResponsiveContainer>
                  </Box>
                ) : (
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                    No temporal data available. Total nodes: {growthTrends.total}
                  </Typography>
                )}
              </Paper>
            ) : (
              <Paper sx={{ p: 4, textAlign: 'center' }}>
                <Typography variant="body2" color="text.secondary">
                  Click "Load Trends" to view graph growth over time
                </Typography>
              </Paper>
            )}
          </Grid>
        </Grid>
      )}
    </Box>
  );
}

