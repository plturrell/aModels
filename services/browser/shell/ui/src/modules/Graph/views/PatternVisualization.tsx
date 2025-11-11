/**
 * Phase 3.3: Pattern Visualization View
 * 
 * Visualizes graph clusters, communities, and patterns
 */

import React, { useState, useCallback, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Grid,
  Alert,
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Stack,
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import { GraphVisualization, LayoutType } from '../../../components/GraphVisualization';
import { GraphNode, GraphEdge, GraphData } from '../../../types/graph';
import { detectCommunities } from '../../../api/graphAnalytics';
import { getStructuralInsights } from '../../../api/gnn';

export interface PatternVisualizationProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  projectId: string;
  systemId?: string;
  onNodeClick?: (nodeId: string) => void;
}

export function PatternVisualization({
  nodes,
  edges,
  projectId,
  systemId,
  onNodeClick,
}: PatternVisualizationProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [layout, setLayout] = useState<LayoutType>('force-directed');
  const [communities, setCommunities] = useState<any>(null);
  const [coloredGraphData, setColoredGraphData] = useState<GraphData>({ nodes: [], edges: [] });
  const [selectedPattern, setSelectedPattern] = useState<string | null>(null);

  // Load communities and color-code graph
  const loadPatterns = useCallback(async () => {
    if (nodes.length === 0) {
      setError('No nodes available for pattern visualization');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Get communities
      const communityResult = await detectCommunities(projectId, systemId, 'louvain');
      setCommunities(communityResult);

      // Get GNN patterns
      const insightsResult = await getStructuralInsights({
        nodes: nodes.map(n => ({
          id: n.id,
          type: n.type,
          label: n.label,
          properties: n.properties,
        })),
        edges: edges.map(e => ({
          source_id: e.source_id,
          target_id: e.target_id,
          label: e.label,
          type: e.type,
          properties: e.properties,
        })),
        insight_type: 'patterns',
      });

      // Color-code nodes by community/pattern
      const coloredNodes = nodes.map((node, idx) => {
        const communityIdx = idx % (communityResult.num_communities || 1);
        const colors = [
          '#619BD6', '#50C878', '#F5A623', '#FF6B6B', '#4ECDC4',
          '#95E1D3', '#F38181', '#AA96DA', '#FCBAD3', '#A8E6CF',
        ];
        
        return {
          ...node,
          properties: {
            ...node.properties,
            community: communityIdx,
            color: colors[communityIdx % colors.length],
          },
        };
      });

      setColoredGraphData({
        nodes: coloredNodes,
        edges: edges,
      });
    } catch (err: any) {
      setError(err.message || 'Failed to load patterns');
      console.error('Pattern visualization error:', err);
    } finally {
      setLoading(false);
    }
  }, [nodes, edges, projectId, systemId]);

  useEffect(() => {
    if (nodes.length > 0) {
      loadPatterns();
    }
  }, [nodes.length]);

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">Pattern Visualization</Typography>
        <Button
          variant="outlined"
          startIcon={loading ? <CircularProgress size={16} /> : <RefreshIcon />}
          onClick={loadPatterns}
          disabled={loading}
        >
          Refresh Patterns
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Grid container spacing={2}>
        <Grid item xs={12} md={3}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Pattern Controls
            </Typography>
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
                <MenuItem value="cose-bilkent">COSE-Bilkent</MenuItem>
              </Select>
            </FormControl>

            {communities && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Communities ({communities.num_communities})
                </Typography>
                <Stack spacing={1}>
                  {communities.communities.slice(0, 10).map((community: any, idx: number) => {
                    const colors = [
                      '#619BD6', '#50C878', '#F5A623', '#FF6B6B', '#4ECDC4',
                      '#95E1D3', '#F38181', '#AA96DA', '#FCBAD3', '#A8E6CF',
                    ];
                    return (
                      <Chip
                        key={idx}
                        label={`Community ${idx + 1} (${community.size} nodes)`}
                        sx={{
                          bgcolor: colors[idx % colors.length],
                          color: 'white',
                          cursor: 'pointer',
                        }}
                        onClick={() => setSelectedPattern(`community-${idx}`)}
                      />
                    );
                  })}
                </Stack>
              </Box>
            )}
          </Paper>
        </Grid>
        <Grid item xs={12} md={9}>
          <Paper sx={{ p: 2, height: '600px' }}>
            {loading && coloredGraphData.nodes.length === 0 ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                <CircularProgress />
              </Box>
            ) : (
              <GraphVisualization
                graphData={coloredGraphData.nodes.length > 0 ? coloredGraphData : { nodes, edges }}
                layout={layout}
                onNodeClick={onNodeClick}
                height={600}
              />
            )}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}

