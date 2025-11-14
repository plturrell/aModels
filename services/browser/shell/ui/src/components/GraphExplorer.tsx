/**
 * GraphExplorer Component
 * 
 * Side panel for exploring graph nodes and edges
 */

import React from 'react';
import { Box, Typography, List, ListItem, ListItemButton, ListItemText, Chip } from '@mui/material';
import { GraphNode, GraphEdge } from '../api/graph';

interface GraphExplorerProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  onNodeSelect: (nodeId: string) => void;
  onNodeFocus: (nodeId: string) => void;
  selectedNodeId?: string;
  focusedNodeId?: string;
}

export function GraphExplorer({
  nodes,
  edges,
  onNodeSelect,
  onNodeFocus,
  selectedNodeId,
  focusedNodeId
}: GraphExplorerProps) {
  return (
    <Box sx={{ height: '100%', overflow: 'auto', p: 2 }}>
      <Typography variant="h6" gutterBottom>
        Graph Explorer
      </Typography>
      
      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle2" gutterBottom>
          Nodes ({nodes.length})
        </Typography>
        <List dense>
          {nodes.map((node) => (
            <ListItem key={node.id} disablePadding>
              <ListItemButton
                selected={selectedNodeId === node.id}
                onClick={() => onNodeSelect(node.id)}
                onMouseEnter={() => onNodeFocus(node.id)}
              >
                <ListItemText
                  primary={node.label || node.id}
                  secondary={
                    <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                      {Object.entries(node.properties || {}).slice(0, 2).map(([key, value]) => (
                        <Chip
                          key={key}
                          label={`${key}: ${value}`}
                          size="small"
                          variant="outlined"
                        />
                      ))}
                    </Box>
                  }
                />
              </ListItemButton>
            </ListItem>
          ))}
        </List>
      </Box>

      <Box>
        <Typography variant="subtitle2" gutterBottom>
          Edges ({edges.length})
        </Typography>
        <List dense>
          {edges.slice(0, 10).map((edge) => (
            <ListItem key={edge.id} disablePadding>
              <ListItemButton>
                <ListItemText
                  primary={edge.label || `${edge.source} → ${edge.target}`}
                  secondary={`${edge.source} → ${edge.target}`}
                />
              </ListItemButton>
            </ListItem>
          ))}
        </List>
      </Box>
    </Box>
  );
}
