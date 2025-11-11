/**
 * Phase 2.3: Graph Context Panel
 * 
 * Shows graph connections and relationships for search results
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  Chip,
  Button,
  List,
  ListItem,
  ListItemText,
  ListItemButton,
  CircularProgress,
  Alert,
  Stack,
  Divider,
} from '@mui/material';
import AccountTreeIcon from '@mui/icons-material/AccountTree';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import { exploreGraph, GraphNode, GraphEdge } from '../api/graph';

export interface GraphContextPanelProps {
  entityId?: string;
  entityLabel?: string;
  projectId?: string;
  onNodeClick?: (nodeId: string) => void;
  onExploreInGraph?: (nodeId: string) => void;
}

export function GraphContextPanel({
  entityId,
  entityLabel,
  projectId,
  onNodeClick,
  onExploreInGraph,
}: GraphContextPanelProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [relatedNodes, setRelatedNodes] = useState<GraphNode[]>([]);
  const [relatedEdges, setRelatedEdges] = useState<GraphEdge[]>([]);

  const loadRelationships = useCallback(async () => {
    if (!entityId || !projectId) {
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await exploreGraph({
        node_id: entityId,
        depth: 2,
        direction: 'both',
        limit: 50,
      });

      setRelatedNodes(response.nodes.filter(n => n.id !== entityId));
      setRelatedEdges(response.edges);
    } catch (err: any) {
      setError(err.message || 'Failed to load relationships');
      console.error('Graph context error:', err);
    } finally {
      setLoading(false);
    }
  }, [entityId, projectId]);

  useEffect(() => {
    if (entityId && projectId) {
      loadRelationships();
    }
  }, [entityId, projectId, loadRelationships]);

  if (!entityId) {
    return (
      <Paper sx={{ p: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
          <AccountTreeIcon color="primary" />
          <Typography variant="h6">Graph Context</Typography>
        </Box>
        <Typography variant="body2" color="text.secondary">
          Select an entity to see its graph relationships
        </Typography>
      </Paper>
    );
  }

  // Group related nodes by type
  const nodesByType: Record<string, GraphNode[]> = {};
  relatedNodes.forEach(node => {
    const type = node.type || 'unknown';
    if (!nodesByType[type]) {
      nodesByType[type] = [];
    }
    nodesByType[type].push(node);
  });

  return (
    <Paper sx={{ p: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <AccountTreeIcon color="primary" />
          <Typography variant="h6">Graph Context</Typography>
        </Box>
        {onExploreInGraph && (
          <Button
            size="small"
            variant="outlined"
            startIcon={<OpenInNewIcon />}
            onClick={() => onExploreInGraph(entityId)}
          >
            Explore in Graph
          </Button>
        )}
      </Box>

      <Box sx={{ mb: 2 }}>
        <Typography variant="subtitle2" gutterBottom>
          Entity: {entityLabel || entityId}
        </Typography>
        <Chip label={entityId} size="small" variant="outlined" />
      </Box>

      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
          <CircularProgress size={24} />
        </Box>
      )}

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {!loading && !error && (
        <>
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Related Entities ({relatedNodes.length})
            </Typography>
            {relatedEdges.length > 0 && (
              <Typography variant="caption" color="text.secondary">
                {relatedEdges.length} relationship{relatedEdges.length !== 1 ? 's' : ''} found
              </Typography>
            )}
          </Box>

          {Object.keys(nodesByType).length > 0 ? (
            <Stack spacing={2}>
              {Object.entries(nodesByType).map(([type, nodes]) => (
                <Box key={type}>
                  <Typography variant="caption" fontWeight="bold" display="block" sx={{ mb: 1 }}>
                    {type} ({nodes.length})
                  </Typography>
                  <List dense sx={{ maxHeight: 200, overflow: 'auto' }}>
                    {nodes.slice(0, 10).map((node) => (
                      <ListItem
                        key={node.id}
                        disablePadding
                        secondaryAction={
                          onNodeClick && (
                            <Button
                              size="small"
                              onClick={() => onNodeClick(node.id)}
                            >
                              View
                            </Button>
                          )
                        }
                      >
                        <ListItemButton onClick={() => onNodeClick?.(node.id)}>
                          <ListItemText
                            primary={node.label || node.id}
                            secondary={node.id}
                          />
                        </ListItemButton>
                      </ListItem>
                    ))}
                    {nodes.length > 10 && (
                      <ListItem>
                        <Typography variant="caption" color="text.secondary">
                          ... and {nodes.length - 10} more
                        </Typography>
                      </ListItem>
                    )}
                  </List>
                </Box>
              ))}
            </Stack>
          ) : (
            <Typography variant="body2" color="text.secondary">
              No related entities found in the graph
            </Typography>
          )}
        </>
      )}
    </Paper>
  );
}

