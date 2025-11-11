/**
 * Phase 1.3: Graph Module
 * 
 * Interactive graph exploration module for visualizing and exploring knowledge graphs
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Card,
  CardContent,
  Alert,
  CircularProgress,
  Chip,
  Tabs,
  Tab,
  Stack,
} from '@mui/material';
import { GridLegacy as Grid } from '@mui/material';
import { GraphVisualization, LayoutType } from '../../components/GraphVisualization';
import { GraphExplorer } from '../../components/GraphExplorer';
import { GraphFilters, GraphFilterState } from '../../components/GraphFilters';
import { NaturalLanguageGraphQuery } from '../../components/NaturalLanguageGraphQuery';
import { GNNInsights } from './views/GNNInsights';
import { Analytics } from './views/Analytics';
import { PatternVisualization } from './views/PatternVisualization';
import { AIGraphAssistant } from '../../components/AIGraphAssistant';
import { GraphRecommendations } from '../../components/GraphRecommendations';
import { VisualQueryBuilder } from '../../components/VisualQueryBuilder';
import { NarrativeInsights } from './views/NarrativeInsights';
import {
  visualizeGraph,
  exploreGraph,
  getGraphStats,
  queryGraph,
  findPaths,
  GraphNode,
  GraphEdge,
} from '../../api/graph';
import { GraphData } from '../../types/graph';

interface GraphModuleProps {
  projectId?: string;
  systemId?: string;
}

export function GraphModule({ projectId, systemId }: GraphModuleProps) {
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], edges: [] });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedNodes, setSelectedNodes] = useState<string[]>([]);
  const [stats, setStats] = useState<any>(null);
  const [activeTab, setActiveTab] = useState(0);
  const [layout, setLayout] = useState<LayoutType>('force-directed');
  const [focusedNodeId, setFocusedNodeId] = useState<string>('');
  const [filters, setFilters] = useState<GraphFilterState>({
    nodeTypes: [],
    edgeTypes: [],
    propertyFilters: {},
  });

  // Form state
  const [formProjectId, setFormProjectId] = useState(projectId || '');
  const [formSystemId, setFormSystemId] = useState(systemId || '');
  const [nodeTypes, setNodeTypes] = useState<string[]>([]);
  const [edgeTypes, setEdgeTypes] = useState<string[]>([]);
  const [limit, setLimit] = useState(10000);

  // Explore state
  const [exploreNodeId, setExploreNodeId] = useState('');
  const [exploreDepth, setExploreDepth] = useState(2);
  const [exploreDirection, setExploreDirection] = useState<'outgoing' | 'incoming' | 'both'>('both');

  // Query state
  const [cypherQuery, setCypherQuery] = useState('MATCH (n:Node) RETURN n LIMIT 10');

  // Path finding state
  const [sourceNodeId, setSourceNodeId] = useState('');
  const [targetNodeId, setTargetNodeId] = useState('');

  // Load graph data
  const loadGraph = useCallback(async () => {
    if (!formProjectId) {
      setError('Project ID is required');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await visualizeGraph({
        project_id: formProjectId,
        system_id: formSystemId || undefined,
        node_types: nodeTypes.length > 0 ? nodeTypes : undefined,
        edge_types: edgeTypes.length > 0 ? edgeTypes : undefined,
        limit,
      });

      setGraphData({
        nodes: response.nodes,
        edges: response.edges,
      });
    } catch (err: any) {
      setError(err.message || 'Failed to load graph');
      console.error('Graph load error:', err);
    } finally {
      setLoading(false);
    }
  }, [formProjectId, formSystemId, nodeTypes, edgeTypes, limit]);

  // Load stats
  const loadStats = useCallback(async () => {
    if (!formProjectId) {
      return;
    }

    try {
      const statsData = await getGraphStats(formProjectId, formSystemId || undefined);
      setStats(statsData);
    } catch (err: any) {
      console.error('Stats load error:', err);
    }
  }, [formProjectId, formSystemId]);

  // Load graph on mount if project ID is provided
  useEffect(() => {
    if (projectId) {
      loadGraph();
      loadStats();
    }
  }, [projectId]);

  // Handle node click
  const handleNodeClick = useCallback((nodeId: string, node: GraphNode) => {
    setSelectedNodes([nodeId]);
    setExploreNodeId(nodeId);
  }, []);

  // Handle explore
  const handleExplore = useCallback(async () => {
    if (!exploreNodeId) {
      setError('Node ID is required for exploration');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await exploreGraph({
        node_id: exploreNodeId,
        depth: exploreDepth,
        direction: exploreDirection,
        limit: 1000,
      });

      setGraphData({
        nodes: response.nodes,
        edges: response.edges,
      });
    } catch (err: any) {
      setError(err.message || 'Failed to explore graph');
      console.error('Graph explore error:', err);
    } finally {
      setLoading(false);
    }
  }, [exploreNodeId, exploreDepth, exploreDirection]);

  // Handle query
  const handleQuery = useCallback(async () => {
    if (!cypherQuery.trim()) {
      setError('Cypher query is required');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await queryGraph({
        query: cypherQuery,
      });

      // Display query results (could be enhanced with a results table)
      console.log('Query results:', response);
      alert(`Query executed in ${response.execution_time_ms}ms. Found ${response.data.length} results.`);
    } catch (err: any) {
      setError(err.message || 'Failed to execute query');
      console.error('Graph query error:', err);
    } finally {
      setLoading(false);
    }
  }, [cypherQuery]);

  // Handle path finding
  const handleFindPaths = useCallback(async () => {
    if (!sourceNodeId || !targetNodeId) {
      setError('Both source and target node IDs are required');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await findPaths({
        source_id: sourceNodeId,
        target_id: targetNodeId,
        max_depth: 5,
      });

      if (response.shortest_path) {
        setSelectedNodes(response.shortest_path.nodes);
        alert(`Found path with length ${response.shortest_path.length}`);
      } else {
        setError('No path found between the specified nodes');
      }
    } catch (err: any) {
      setError(err.message || 'Failed to find paths');
      console.error('Path finding error:', err);
    } finally {
      setLoading(false);
    }
  }, [sourceNodeId, targetNodeId]);

  return (
    <Box sx={{ p: 3, height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Typography variant="h4" gutterBottom>
        Graph Explorer
      </Typography>

      <Tabs value={activeTab} onChange={(_, v) => setActiveTab(v)} sx={{ mb: 2 }}>
        <Tab label="Visualize" />
        <Tab label="Explore" />
        <Tab label="Natural Language" />
        <Tab label="GNN Insights" />
        <Tab label="Analytics" />
        <Tab label="Patterns" />
        <Tab label="Narrative" />
        <Tab label="AI Assistant" />
        <Tab label="Recommendations" />
        <Tab label="Query Builder" />
        <Tab label="Query" />
        <Tab label="Paths" />
        <Tab label="Stats" />
        <Tab label="From Extraction" />
      </Tabs>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {activeTab === 0 && (
        <Grid container spacing={2}>
          <Grid item xs={12} md={2}>
            <Stack spacing={2}>
              <Paper sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Load Graph
                </Typography>
                <TextField
                  fullWidth
                  label="Project ID"
                  value={formProjectId}
                  onChange={(e) => setFormProjectId(e.target.value)}
                  sx={{ mb: 2 }}
                  required
                  size="small"
                />
                <TextField
                  fullWidth
                  label="System ID"
                  value={formSystemId}
                  onChange={(e) => setFormSystemId(e.target.value)}
                  sx={{ mb: 2 }}
                  size="small"
                />
                <TextField
                  fullWidth
                  label="Limit"
                  type="number"
                  value={limit}
                  onChange={(e) => setLimit(parseInt(e.target.value) || 10000)}
                  sx={{ mb: 2 }}
                  size="small"
                />
                <FormControl fullWidth sx={{ mb: 2 }}>
                  <InputLabel>Layout</InputLabel>
                  <Select
                    value={layout}
                    label="Layout"
                    onChange={(e) => setLayout(e.target.value as LayoutType)}
                    size="small"
                  >
                    <MenuItem value="force-directed">Force-Directed</MenuItem>
                    <MenuItem value="hierarchical">Hierarchical</MenuItem>
                    <MenuItem value="circular">Circular</MenuItem>
                    <MenuItem value="breadthfirst">Breadth-First</MenuItem>
                    <MenuItem value="cose-bilkent">COSE-Bilkent</MenuItem>
                    <MenuItem value="dagre">Dagre</MenuItem>
                    <MenuItem value="cola">Cola</MenuItem>
                  </Select>
                </FormControl>
                <Button
                  fullWidth
                  variant="contained"
                  onClick={loadGraph}
                  disabled={loading || !formProjectId}
                  size="small"
                  sx={{ mb: 1 }}
                >
                  {loading ? <CircularProgress size={20} /> : 'Load Graph'}
                </Button>
                <Button
                  fullWidth
                  variant="outlined"
                  size="small"
                  onClick={() => {
                    // Navigate to Extract module's Extract & Visualize workflow
                    window.location.hash = '#extract-graph-workflow';
                  }}
                >
                  Extract & Visualize
                </Button>
              </Paper>
              {stats && (
                <GraphFilters
                  availableNodeTypes={Object.keys(stats.node_types || {})}
                  availableEdgeTypes={Object.keys(stats.edge_types || {})}
                  onFilterChange={(newFilters) => {
                    setFilters(newFilters);
                    // Apply filters to graph data
                    if (newFilters.nodeTypes.length > 0 || newFilters.edgeTypes.length > 0) {
                      setNodeTypes(newFilters.nodeTypes);
                      setEdgeTypes(newFilters.edgeTypes);
                    }
                  }}
                  initialFilters={filters}
                />
              )}
            </Stack>
          </Grid>
          <Grid item xs={12} md={7}>
            <Paper sx={{ p: 2, height: '600px' }}>
              {loading && graphData.nodes.length === 0 ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                  <CircularProgress />
                </Box>
              ) : (
                <GraphVisualization
                  graphData={graphData}
                  layout={layout}
                  onNodeClick={handleNodeClick}
                  onNodeSelect={setSelectedNodes}
                  selectedNodes={selectedNodes}
                  height={600}
                />
              )}
            </Paper>
          </Grid>
          <Grid item xs={12} md={3}>
            <GraphExplorer
              nodes={graphData.nodes}
              edges={graphData.edges}
              onNodeSelect={(nodeId) => {
                setSelectedNodes([nodeId]);
                handleNodeClick(nodeId, graphData.nodes.find(n => n.id === nodeId)!);
              }}
              onNodeFocus={setFocusedNodeId}
              selectedNodeId={selectedNodes[0]}
              focusedNodeId={focusedNodeId}
            />
          </Grid>
        </Grid>
      )}

      {activeTab === 1 && (
        <Grid container spacing={2}>
          <Grid item xs={12} md={3}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Explore from Node
              </Typography>
              <TextField
                fullWidth
                label="Node ID"
                value={exploreNodeId}
                onChange={(e) => setExploreNodeId(e.target.value)}
                sx={{ mb: 2 }}
                required
              />
              <TextField
                fullWidth
                label="Depth"
                type="number"
                value={exploreDepth}
                onChange={(e) => setExploreDepth(parseInt(e.target.value) || 2)}
                sx={{ mb: 2 }}
              />
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Direction</InputLabel>
                <Select
                  value={exploreDirection}
                  label="Direction"
                  onChange={(e) => setExploreDirection(e.target.value as any)}
                >
                  <MenuItem value="outgoing">Outgoing</MenuItem>
                  <MenuItem value="incoming">Incoming</MenuItem>
                  <MenuItem value="both">Both</MenuItem>
                </Select>
              </FormControl>
              <Button
                fullWidth
                variant="contained"
                onClick={handleExplore}
                disabled={loading || !exploreNodeId}
              >
                {loading ? <CircularProgress size={24} /> : 'Explore'}
              </Button>
            </Paper>
          </Grid>
          <Grid item xs={12} md={9}>
            <Paper sx={{ p: 2, height: '600px' }}>
              {loading && graphData.nodes.length === 0 ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                  <CircularProgress />
                </Box>
              ) : (
                <GraphVisualization
                  graphData={graphData}
                  layout={layout}
                  onNodeClick={handleNodeClick}
                  onNodeSelect={setSelectedNodes}
                  selectedNodes={selectedNodes}
                  height={600}
                />
              )}
            </Paper>
          </Grid>
        </Grid>
      )}

      {activeTab === 2 && (
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <NaturalLanguageGraphQuery
              onQueryGenerated={(cypher, results) => {
                setCypherQuery(cypher);
                // Could update graph visualization with results
                console.log('Query generated:', cypher, results);
              }}
              onError={(err) => setError(err)}
            />
          </Grid>
        </Grid>
      )}

      {activeTab === 3 && (
        <GNNInsights
          nodes={graphData.nodes}
          edges={graphData.edges}
          projectId={formProjectId}
          onNodeClick={(nodeId) => {
            setSelectedNodes([nodeId]);
            handleNodeClick(nodeId, graphData.nodes.find(n => n.id === nodeId)!);
          }}
        />
      )}

      {activeTab === 4 && (
        <Analytics
          projectId={formProjectId}
          systemId={formSystemId || undefined}
        />
      )}

      {activeTab === 5 && (
        <PatternVisualization
          nodes={graphData.nodes}
          edges={graphData.edges}
          projectId={formProjectId}
          systemId={formSystemId || undefined}
          onNodeClick={(nodeId) => {
            setSelectedNodes([nodeId]);
            handleNodeClick(nodeId, graphData.nodes.find(n => n.id === nodeId)!);
          }}
        />
      )}

      {activeTab === 6 && (
        <NarrativeInsights
          nodes={graphData.nodes}
          edges={graphData.edges}
          projectId={formProjectId}
          systemId={formSystemId || undefined}
          onNodeClick={(nodeId) => {
            setSelectedNodes([nodeId]);
            handleNodeClick(nodeId, graphData.nodes.find(n => n.id === nodeId)!);
          }}
        />
      )}

      {activeTab === 8 && (
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <AIGraphAssistant
              nodes={graphData.nodes}
              edges={graphData.edges}
              projectId={formProjectId}
              onNodeClick={(nodeId) => {
                setSelectedNodes([nodeId]);
                handleNodeClick(nodeId, graphData.nodes.find(n => n.id === nodeId)!);
              }}
              onQueryGenerated={(query) => {
                setCypherQuery(query);
                setActiveTab(11); // Switch to Query tab
              }}
            />
          </Grid>
        </Grid>
      )}

      {activeTab === 9 && (
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <GraphRecommendations
              nodes={graphData.nodes}
              edges={graphData.edges}
              selectedNodeId={selectedNodes[0]}
              onNodeClick={(nodeId) => {
                setSelectedNodes([nodeId]);
                handleNodeClick(nodeId, graphData.nodes.find(n => n.id === nodeId)!);
              }}
              onPathClick={(path) => {
                setSelectedNodes(path);
              }}
            />
          </Grid>
        </Grid>
      )}

      {activeTab === 10 && (
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <VisualQueryBuilder
              nodes={graphData.nodes}
              edges={graphData.edges}
              onQueryGenerated={(query) => {
                setCypherQuery(query);
                setActiveTab(11); // Switch to Query tab
              }}
              onQueryExecute={async (query) => {
                return await queryGraph({ query });
              }}
            />
          </Grid>
        </Grid>
      )}

      {activeTab === 11 && (
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Cypher Query
              </Typography>
              <TextField
                fullWidth
                multiline
                rows={6}
                value={cypherQuery}
                onChange={(e) => setCypherQuery(e.target.value)}
                sx={{ mb: 2 }}
                placeholder="MATCH (n:Node) RETURN n LIMIT 10"
              />
              <Button
                variant="contained"
                onClick={handleQuery}
                disabled={loading || !cypherQuery.trim()}
              >
                {loading ? <CircularProgress size={24} /> : 'Execute Query'}
              </Button>
            </Paper>
          </Grid>
        </Grid>
      )}

      {activeTab === 12 && (
        <Grid container spacing={2}>
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Find Paths
              </Typography>
              <TextField
                fullWidth
                label="Source Node ID"
                value={sourceNodeId}
                onChange={(e) => setSourceNodeId(e.target.value)}
                sx={{ mb: 2 }}
                required
              />
              <TextField
                fullWidth
                label="Target Node ID"
                value={targetNodeId}
                onChange={(e) => setTargetNodeId(e.target.value)}
                sx={{ mb: 2 }}
                required
              />
              <Button
                fullWidth
                variant="contained"
                onClick={handleFindPaths}
                disabled={loading || !sourceNodeId || !targetNodeId}
              >
                {loading ? <CircularProgress size={24} /> : 'Find Paths'}
              </Button>
            </Paper>
          </Grid>
        </Grid>
      )}

      {activeTab === 13 && (
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Graph Statistics
              </Typography>
              {stats ? (
                <Box>
                  <Card sx={{ mb: 2 }}>
                    <CardContent>
                      <Typography variant="h6">Overview</Typography>
                      <Box sx={{ mt: 2 }}>
                        <Chip label={`Total Nodes: ${stats.total_nodes}`} sx={{ mr: 1, mb: 1 }} />
                        <Chip label={`Total Edges: ${stats.total_edges}`} sx={{ mr: 1, mb: 1 }} />
                        <Chip label={`Density: ${stats.density.toFixed(4)}`} sx={{ mr: 1, mb: 1 }} />
                        <Chip label={`Avg Degree: ${stats.average_degree.toFixed(2)}`} sx={{ mr: 1, mb: 1 }} />
                      </Box>
                    </CardContent>
                  </Card>
                  <Card sx={{ mb: 2 }}>
                    <CardContent>
                      <Typography variant="h6">Node Types</Typography>
                      <Box sx={{ mt: 2 }}>
                        {Object.entries(stats.node_types || {}).map(([type, count]) => (
                          <Chip
                            key={type}
                            label={`${type}: ${count}`}
                            sx={{ mr: 1, mb: 1 }}
                          />
                        ))}
                      </Box>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent>
                      <Typography variant="h6">Edge Types</Typography>
                      <Box sx={{ mt: 2 }}>
                        {Object.entries(stats.edge_types || {}).map(([type, count]) => (
                          <Chip
                            key={type}
                            label={`${type}: ${count}`}
                            sx={{ mr: 1, mb: 1 }}
                          />
                        ))}
                      </Box>
                    </CardContent>
                  </Card>
                </Box>
              ) : (
                <Box>
                  <Button variant="contained" onClick={loadStats} disabled={!formProjectId}>
                    Load Statistics
                  </Button>
                </Box>
              )}
            </Paper>
          </Grid>
        </Grid>
      )}

      {activeTab === 13 && (
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Graphs from Extractions
              </Typography>
              <Typography variant="body2" color="textSecondary" sx={{ mb: 3 }}>
                View and explore graphs generated from extraction jobs
              </Typography>
              <Alert severity="info" sx={{ mb: 2 }}>
                Use the Extract module to create extractions and generate graphs. 
                Graphs generated from extractions will appear here.
              </Alert>
              <Button
                variant="contained"
                onClick={() => {
                  window.location.hash = '#extract';
                }}
              >
                Go to Extract Module
              </Button>
            </Paper>
          </Grid>
        </Grid>
      )}
    </Box>
  );
}

