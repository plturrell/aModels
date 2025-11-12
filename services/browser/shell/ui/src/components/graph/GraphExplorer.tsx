/**
 * Phase 2.1: Interactive Graph Explorer
 * 
 * Enhanced graph exploration with search, filters, and relationship browsing
 */

import React, { useState, useCallback, useMemo } from 'react';
import {
  Box,
  Paper,
  TextField,
  Button,
  Autocomplete,
  Chip,
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemButton,
  IconButton,
  Divider,
  Tooltip,
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import ClearIcon from '@mui/icons-material/Clear';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import { GraphNode, GraphEdge } from '../types/graph';
import { queryGraph } from '../api/graph';

export interface GraphExplorerProps {
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
  focusedNodeId,
}: GraphExplorerProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<GraphNode[]>([]);
  const [relationshipView, setRelationshipView] = useState<'outgoing' | 'incoming' | 'both'>('both');
  const [maxDepth, setMaxDepth] = useState(2);

  // Node search
  const handleSearch = useCallback(() => {
    if (!searchQuery.trim()) {
      setSearchResults([]);
      return;
    }

    const query = searchQuery.toLowerCase();
    const results = nodes.filter(node => {
      const label = (node.label || '').toLowerCase();
      const id = (node.id || '').toLowerCase();
      const type = (node.type || '').toLowerCase();
      
      return label.includes(query) || 
             id.includes(query) || 
             type.includes(query) ||
             JSON.stringify(node.properties || {}).toLowerCase().includes(query);
    });

    setSearchResults(results.slice(0, 50)); // Limit to 50 results
  }, [searchQuery, nodes]);

  // Get relationships for a node
  const getNodeRelationships = useCallback((nodeId: string) => {
    const outgoing = edges.filter(e => e.source_id === nodeId);
    const incoming = edges.filter(e => e.target_id === nodeId);
    
    if (relationshipView === 'outgoing') {
      return outgoing;
    } else if (relationshipView === 'incoming') {
      return incoming;
    } else {
      return [...outgoing, ...incoming];
    }
  }, [edges, relationshipView]);

  // Get connected nodes
  const getConnectedNodes = useCallback((nodeId: string, depth: number = 1): GraphNode[] => {
    if (depth <= 0) return [];
    
    const connectedIds = new Set<string>();
    const relationships = getNodeRelationships(nodeId);
    
    relationships.forEach(rel => {
      if (rel.source_id === nodeId) {
        connectedIds.add(rel.target_id);
      }
      if (rel.target_id === nodeId) {
        connectedIds.add(rel.source_id);
      }
    });

    const connectedNodes = nodes.filter(n => connectedIds.has(n.id));
    
    if (depth > 1) {
      const deeperNodes: GraphNode[] = [];
      connectedNodes.forEach(node => {
        const deeper = getConnectedNodes(node.id, depth - 1);
        deeper.forEach(dn => {
          if (!connectedIds.has(dn.id) && dn.id !== nodeId) {
            connectedIds.add(dn.id);
            deeperNodes.push(dn);
          }
        });
      });
      return [...connectedNodes, ...deeperNodes];
    }
    
    return connectedNodes;
  }, [nodes, getNodeRelationships]);

  // Node type distribution
  const nodeTypeCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    nodes.forEach(node => {
      const type = node.type || 'unknown';
      counts[type] = (counts[type] || 0) + 1;
    });
    return counts;
  }, [nodes]);

  // Edge type distribution
  const edgeTypeCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    edges.forEach(edge => {
      const type = edge.label || edge.type || 'unknown';
      counts[type] = (counts[type] || 0) + 1;
    });
    return counts;
  }, [edges]);

  const focusedNode = focusedNodeId ? nodes.find(n => n.id === focusedNodeId) : null;
  const focusedRelationships = focusedNode ? getNodeRelationships(focusedNode.id) : [];
  const connectedNodes = focusedNode ? getConnectedNodes(focusedNode.id, maxDepth) : [];

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, height: '100%' }}>
      {/* Search Section */}
      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          Node Search
        </Typography>
        <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
          <TextField
            fullWidth
            size="small"
            placeholder="Search by label, ID, type, or properties..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                handleSearch();
              }
            }}
            InputProps={{
              startAdornment: <SearchIcon sx={{ mr: 1, color: 'text.secondary' }} />,
            }}
          />
          <Button
            variant="contained"
            onClick={handleSearch}
            disabled={!searchQuery.trim()}
          >
            Search
          </Button>
          {searchQuery && (
            <IconButton onClick={() => {
              setSearchQuery('');
              setSearchResults([]);
            }}>
              <ClearIcon />
            </IconButton>
          )}
        </Box>

        {searchResults.length > 0 && (
          <Box>
            <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
              Found {searchResults.length} result{searchResults.length !== 1 ? 's' : ''}
            </Typography>
            <List dense sx={{ maxHeight: 200, overflow: 'auto' }}>
              {searchResults.map((node) => (
                <ListItem
                  key={node.id}
                  disablePadding
                  secondaryAction={
                    <IconButton
                      size="small"
                      onClick={() => onNodeFocus(node.id)}
                    >
                      <ArrowForwardIcon fontSize="small" />
                    </IconButton>
                  }
                >
                  <ListItemButton
                    selected={selectedNodeId === node.id}
                    onClick={() => onNodeSelect(node.id)}
                  >
                    <ListItemText
                      primary={node.label || node.id}
                      secondary={`${node.type || 'unknown'} • ${node.id}`}
                    />
                  </ListItemButton>
                </ListItem>
              ))}
            </List>
          </Box>
        )}
      </Paper>

      {/* Node Type Filters */}
      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          Node Types
        </Typography>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
          {Object.entries(nodeTypeCounts).map(([type, count]) => (
            <Chip
              key={type}
              label={`${type} (${count})`}
              size="small"
              onClick={() => {
                // Filter by node type - could be enhanced with actual filtering
                const nodesOfType = nodes.filter(n => n.type === type);
                if (nodesOfType.length > 0) {
                  onNodeFocus(nodesOfType[0].id);
                }
              }}
            />
          ))}
        </Box>
      </Paper>

      {/* Focused Node Details */}
      {focusedNode && (
        <Paper sx={{ p: 2 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">
              Focused Node
            </Typography>
            <IconButton
              size="small"
              onClick={() => onNodeFocus('')}
            >
              <ClearIcon />
            </IconButton>
          </Box>

          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle1" fontWeight="bold">
              {focusedNode.label || focusedNode.id}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Type: {focusedNode.type || 'unknown'} • ID: {focusedNode.id}
            </Typography>
          </Box>

          {focusedNode.properties && Object.keys(focusedNode.properties).length > 0 && (
            <Box sx={{ mb: 2 }}>
              <Typography variant="caption" fontWeight="bold" display="block" sx={{ mb: 1 }}>
                Properties:
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                {Object.entries(focusedNode.properties).slice(0, 10).map(([key, value]) => (
                  <Chip
                    key={key}
                    label={`${key}: ${String(value).slice(0, 20)}`}
                    size="small"
                    variant="outlined"
                  />
                ))}
              </Box>
            </Box>
          )}

          <Divider sx={{ my: 2 }} />

          {/* Relationship Controls */}
          <Box sx={{ display: 'flex', gap: 1, mb: 2, alignItems: 'center' }}>
            <Typography variant="caption">View:</Typography>
            <Button
              size="small"
              variant={relationshipView === 'outgoing' ? 'contained' : 'outlined'}
              onClick={() => setRelationshipView('outgoing')}
              startIcon={<ArrowForwardIcon />}
            >
              Outgoing
            </Button>
            <Button
              size="small"
              variant={relationshipView === 'incoming' ? 'contained' : 'outlined'}
              onClick={() => setRelationshipView('incoming')}
              startIcon={<ArrowBackIcon />}
            >
              Incoming
            </Button>
            <Button
              size="small"
              variant={relationshipView === 'both' ? 'contained' : 'outlined'}
              onClick={() => setRelationshipView('both')}
            >
              Both
            </Button>
          </Box>

          {/* Relationships */}
          <Box sx={{ mb: 2 }}>
            <Typography variant="caption" fontWeight="bold" display="block" sx={{ mb: 1 }}>
              Relationships ({focusedRelationships.length}):
            </Typography>
            <List dense sx={{ maxHeight: 150, overflow: 'auto' }}>
              {focusedRelationships.slice(0, 20).map((rel, idx) => {
                const otherNodeId = rel.source_id === focusedNode.id ? rel.target_id : rel.source_id;
                const otherNode = nodes.find(n => n.id === otherNodeId);
                return (
                  <ListItem
                    key={`${rel.source_id}-${rel.target_id}-${idx}`}
                    disablePadding
                    secondaryAction={
                      <IconButton
                        size="small"
                        onClick={() => onNodeFocus(otherNodeId)}
                      >
                        <ArrowForwardIcon fontSize="small" />
                      </IconButton>
                    }
                  >
                    <ListItemButton
                      onClick={() => onNodeSelect(otherNodeId)}
                    >
                      <ListItemText
                        primary={rel.label || 'relationship'}
                        secondary={
                          <Box component="span">
                            {rel.source_id === focusedNode.id ? (
                              <>
                                <ArrowForwardIcon fontSize="small" sx={{ verticalAlign: 'middle', fontSize: 14 }} />
                                {' '}
                                {otherNode?.label || otherNodeId}
                              </>
                            ) : (
                              <>
                                {otherNode?.label || otherNodeId}
                                {' '}
                                <ArrowBackIcon fontSize="small" sx={{ verticalAlign: 'middle', fontSize: 14 }} />
                              </>
                            )}
                          </Box>
                        }
                      />
                    </ListItemButton>
                  </ListItem>
                );
              })}
            </List>
          </Box>

          {/* Connected Nodes */}
          <Box>
            <Typography variant="caption" fontWeight="bold" display="block" sx={{ mb: 1 }}>
              Connected Nodes (depth {maxDepth}, {connectedNodes.length}):
            </Typography>
            <Box sx={{ display: 'flex', gap: 0.5, mb: 1 }}>
              {[1, 2, 3].map(depth => (
                <Button
                  key={depth}
                  size="small"
                  variant={maxDepth === depth ? 'contained' : 'outlined'}
                  onClick={() => setMaxDepth(depth)}
                >
                  Depth {depth}
                </Button>
              ))}
            </Box>
            <List dense sx={{ maxHeight: 150, overflow: 'auto' }}>
              {connectedNodes.slice(0, 20).map((node) => (
                <ListItem
                  key={node.id}
                  disablePadding
                  secondaryAction={
                    <IconButton
                      size="small"
                      onClick={() => onNodeFocus(node.id)}
                    >
                      <ArrowForwardIcon fontSize="small" />
                    </IconButton>
                  }
                >
                  <ListItemButton
                    onClick={() => onNodeSelect(node.id)}
                  >
                    <ListItemText
                      primary={node.label || node.id}
                      secondary={node.type || 'unknown'}
                    />
                  </ListItemButton>
                </ListItem>
              ))}
            </List>
          </Box>
        </Paper>
      )}

      {/* Edge Type Summary */}
      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          Relationship Types
        </Typography>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
          {Object.entries(edgeTypeCounts).slice(0, 10).map(([type, count]) => (
            <Chip
              key={type}
              label={`${type} (${count})`}
              size="small"
              variant="outlined"
            />
          ))}
        </Box>
      </Paper>
    </Box>
  );
}

