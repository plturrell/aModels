/**
 * Phase 4.2: Graph-Based Recommendations Component
 * 
 * Suggests related entities, exploration paths, and important connections
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemButton,
  Chip,
  Button,
  CircularProgress,
  Alert,
  Stack,
  Divider,
} from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import ExploreIcon from '@mui/icons-material/Explore';
import LinkIcon from '@mui/icons-material/Link';
import RefreshIcon from '@mui/icons-material/Refresh';
import { GraphNode, GraphEdge } from '../types/graph';
import { useGNNAnalysis } from '../../hooks/useAI';

export interface Recommendation {
  type: 'entity' | 'path' | 'connection' | 'pattern';
  title: string;
  description: string;
  nodes?: string[];
  edges?: Array<{ source: string; target: string }>;
  score: number;
  reason?: string;
}

export interface GraphRecommendationsProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  selectedNodeId?: string;
  onNodeClick?: (nodeId: string) => void;
  onPathClick?: (path: string[]) => void;
  maxRecommendations?: number;
}

export function GraphRecommendations({
  nodes,
  edges,
  selectedNodeId,
  onNodeClick,
  onPathClick,
  maxRecommendations = 10,
}: GraphRecommendationsProps) {
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { analyzeGraph } = useGNNAnalysis();

  const generateRecommendations = useCallback(async () => {
    if (nodes.length === 0) {
      setError('No nodes available for recommendations');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const recs: Recommendation[] = [];

      // 1. Most connected nodes (importance-based)
      const nodeConnections = new Map<string, number>();
      edges.forEach(edge => {
        nodeConnections.set(edge.source_id, (nodeConnections.get(edge.source_id) || 0) + 1);
        nodeConnections.set(edge.target_id, (nodeConnections.get(edge.target_id) || 0) + 1);
      });

      const topNodes = Array.from(nodeConnections.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5)
        .map(([nodeId, count]) => {
          const node = nodes.find(n => n.id === nodeId);
          return {
            type: 'entity' as const,
            title: node?.label || nodeId,
            description: `Highly connected node with ${count} connections`,
            nodes: [nodeId],
            score: count / (edges.length || 1),
            reason: 'High connectivity indicates importance in the graph',
          };
        });

      recs.push(...topNodes);

      // 2. Exploration paths from selected node
      if (selectedNodeId) {
        const connectedNodes = edges
          .filter(e => e.source_id === selectedNodeId || e.target_id === selectedNodeId)
          .map(e => e.source_id === selectedNodeId ? e.target_id : e.source_id)
          .slice(0, 3);

        connectedNodes.forEach(targetId => {
          const path = [selectedNodeId, targetId];
          recs.push({
            type: 'path',
            title: `Explore from ${nodes.find(n => n.id === selectedNodeId)?.label || selectedNodeId}`,
            description: `Path to ${nodes.find(n => n.id === targetId)?.label || targetId}`,
            nodes: path,
            score: 0.7,
            reason: 'Directly connected to your selected node',
          });
        });
      }

      // 3. Interesting connections (nodes with multiple types of relationships)
      const nodeEdgeTypes = new Map<string, Set<string>>();
      edges.forEach(edge => {
        const type = edge.label || edge.type || 'unknown';
        [edge.source_id, edge.target_id].forEach(nodeId => {
          if (!nodeEdgeTypes.has(nodeId)) {
            nodeEdgeTypes.set(nodeId, new Set());
          }
          nodeEdgeTypes.get(nodeId)!.add(type);
        });
      });

      const diverseNodes = Array.from(nodeEdgeTypes.entries())
        .filter(([_, types]) => types.size > 2)
        .sort((a, b) => b[1].size - a[1].size)
        .slice(0, 3)
        .map(([nodeId, types]) => {
          const node = nodes.find(n => n.id === nodeId);
          return {
            type: 'connection' as const,
            title: node?.label || nodeId,
            description: `Has ${types.size} different types of relationships`,
            nodes: [nodeId],
            score: types.size / 10,
            reason: 'Diverse connections indicate complex relationships',
          };
        });

      recs.push(...diverseNodes);

      // 4. Pattern-based recommendations (using GNN embeddings if available)
      try {
        const result = await analyzeGraph({
          graph: {
            nodes: nodes.slice(0, 50).map(n => ({
              id: n.id,
              label: n.label,
              type: n.type,
              properties: n.properties,
            })),
            edges: edges.slice(0, 100).map(e => ({
              source_id: e.source_id,
              target_id: e.target_id,
              label: e.label,
              type: e.type,
              properties: e.properties,
            })),
          },
          graph_level: true,
          task: 'embeddings',
        });

        if (result && (result as any).graph_embedding) {
          recs.push({
            type: 'pattern',
            title: 'Graph Pattern Analysis',
            description: 'GNN embeddings available for pattern discovery',
            score: 0.8,
            reason: 'Graph structure analyzed with GNN embeddings',
          });
        }
      } catch (err) {
        console.debug('GNN embeddings not available for recommendations');
      }

      // Sort by score and limit
      recs.sort((a, b) => b.score - a.score);
      setRecommendations(recs.slice(0, maxRecommendations));
    } catch (err: any) {
      setError(err.message || 'Failed to generate recommendations');
      console.error('Recommendations error:', err);
    } finally {
      setLoading(false);
    }
  }, [nodes, edges, selectedNodeId, maxRecommendations]);

  useEffect(() => {
    if (nodes.length > 0) {
      generateRecommendations();
    }
  }, [nodes.length, selectedNodeId]);

  const getRecommendationIcon = (type: Recommendation['type']) => {
    switch (type) {
      case 'entity':
        return <TrendingUpIcon />;
      case 'path':
        return <ExploreIcon />;
      case 'connection':
        return <LinkIcon />;
      case 'pattern':
        return <TrendingUpIcon />;
    }
  };

  const handleRecommendationClick = (rec: Recommendation) => {
    if (rec.nodes && rec.nodes.length === 1 && onNodeClick) {
      onNodeClick(rec.nodes[0]);
    } else if (rec.nodes && rec.nodes.length > 1 && onPathClick) {
      onPathClick(rec.nodes);
    }
  };

  return (
    <Paper sx={{ p: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">You might be interested in...</Typography>
        <Button
          size="small"
          startIcon={<RefreshIcon />}
          onClick={generateRecommendations}
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

      {loading && recommendations.length === 0 ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
          <CircularProgress />
        </Box>
      ) : recommendations.length === 0 ? (
        <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 2 }}>
          No recommendations available. Load a graph to get suggestions.
        </Typography>
      ) : (
        <List>
          {recommendations.map((rec, idx) => (
            <React.Fragment key={idx}>
              <ListItem
                disablePadding
                secondaryAction={
                  <Chip
                    label={`${(rec.score * 100).toFixed(0)}%`}
                    size="small"
                    color={rec.score > 0.7 ? 'primary' : 'default'}
                  />
                }
              >
                <ListItemButton onClick={() => handleRecommendationClick(rec)}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mr: 2 }}>
                    {getRecommendationIcon(rec.type)}
                  </Box>
                  <ListItemText
                    primary={rec.title}
                    secondary={
                      <Box>
                        <Typography variant="caption" display="block">
                          {rec.description}
                        </Typography>
                        {rec.reason && (
                          <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 0.5 }}>
                            {rec.reason}
                          </Typography>
                        )}
                        {rec.nodes && rec.nodes.length > 0 && (
                          <Stack direction="row" spacing={0.5} sx={{ mt: 0.5, flexWrap: 'wrap' }}>
                            {rec.nodes.slice(0, 3).map((nodeId) => {
                              const node = nodes.find(n => n.id === nodeId);
                              return (
                                <Chip
                                  key={nodeId}
                                  label={node?.label || nodeId}
                                  size="small"
                                  variant="outlined"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    onNodeClick?.(nodeId);
                                  }}
                                />
                              );
                            })}
                            {rec.nodes.length > 3 && (
                              <Chip label={`+${rec.nodes.length - 3}`} size="small" variant="outlined" />
                            )}
                          </Stack>
                        )}
                      </Box>
                    }
                  />
                </ListItemButton>
              </ListItem>
              {idx < recommendations.length - 1 && <Divider />}
            </React.Fragment>
          ))}
        </List>
      )}
    </Paper>
  );
}

