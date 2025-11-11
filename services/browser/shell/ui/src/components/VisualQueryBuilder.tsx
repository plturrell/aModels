/**
 * Phase 4.3: Visual Query Builder Component
 * 
 * Drag-and-drop Cypher query construction with templates
 */

import React, { useState, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  Stack,
  Grid,
  Alert,
  Divider,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  CircularProgress,
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import SaveIcon from '@mui/icons-material/Save';
import DeleteIcon from '@mui/icons-material/Delete';
import { GraphNode, GraphEdge } from '../types/graph';

export interface QueryTemplate {
  id: string;
  name: string;
  description: string;
  query: string;
  category: 'basic' | 'exploration' | 'analysis' | 'pattern';
}

const QUERY_TEMPLATES: QueryTemplate[] = [
  {
    id: 'find-node',
    name: 'Find Node',
    description: 'Find a specific node by ID or label',
    query: 'MATCH (n:Node {id: $nodeId}) RETURN n',
    category: 'basic',
  },
  {
    id: 'find-connections',
    name: 'Find Connections',
    description: 'Find all nodes connected to a specific node',
    query: 'MATCH (n:Node {id: $nodeId})-[r:RELATIONSHIP]-(connected) RETURN n, r, connected',
    category: 'exploration',
  },
  {
    id: 'shortest-path',
    name: 'Shortest Path',
    description: 'Find shortest path between two nodes',
    query: 'MATCH path = shortestPath((start:Node {id: $startId})-[*]-(end:Node {id: $endId})) RETURN path',
    category: 'exploration',
  },
  {
    id: 'most-connected',
    name: 'Most Connected Nodes',
    description: 'Find nodes with the most connections',
    query: 'MATCH (n:Node) WITH n, size((n)-[:RELATIONSHIP]-()) as degree ORDER BY degree DESC LIMIT 10 RETURN n, degree',
    category: 'analysis',
  },
  {
    id: 'find-pattern',
    name: 'Find Pattern',
    description: 'Find nodes matching a pattern',
    query: 'MATCH (a:Node)-[r:RELATIONSHIP]->(b:Node) WHERE a.type = $typeA AND b.type = $typeB RETURN a, r, b LIMIT 50',
    category: 'pattern',
  },
  {
    id: 'community-detection',
    name: 'Community Detection',
    description: 'Find communities in the graph',
    query: 'MATCH (n:Node)-[r:RELATIONSHIP]-(m:Node) WITH n, collect(m) as neighbors RETURN n, size(neighbors) as community_size',
    category: 'analysis',
  },
];

export interface VisualQueryBuilderProps {
  onQueryGenerated?: (query: string) => void;
  onQueryExecute?: (query: string) => Promise<any>;
  nodes?: GraphNode[];
  edges?: GraphEdge[];
}

export function VisualQueryBuilder({
  onQueryGenerated,
  onQueryExecute,
  nodes = [],
  edges = [],
}: VisualQueryBuilderProps) {
  const [selectedTemplate, setSelectedTemplate] = useState<string>('');
  const [query, setQuery] = useState('');
  const [queryParams, setQueryParams] = useState<Record<string, string>>({});
  const [savedQueries, setSavedQueries] = useState<QueryTemplate[]>([]);
  const [executing, setExecuting] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [saveDialogOpen, setSaveDialogOpen] = useState(false);
  const [saveName, setSaveName] = useState('');

  const handleTemplateSelect = useCallback((templateId: string) => {
    const template = QUERY_TEMPLATES.find(t => t.id === templateId);
    if (template) {
      setSelectedTemplate(templateId);
      setQuery(template.query);
      setQueryParams({});
    }
  }, []);

  const handleExecute = useCallback(async () => {
    if (!query.trim()) {
      setError('Query cannot be empty');
      return;
    }

    setExecuting(true);
    setError(null);
    setResults(null);

    try {
      // Replace parameters in query
      let finalQuery = query;
      Object.entries(queryParams).forEach(([key, value]) => {
        finalQuery = finalQuery.replace(new RegExp(`\\$${key}`, 'g'), `'${value}'`);
      });

      if (onQueryExecute) {
        const result = await onQueryExecute(finalQuery);
        setResults(result);
      } else if (onQueryGenerated) {
        onQueryGenerated(finalQuery);
      }
    } catch (err: any) {
      setError(err.message || 'Failed to execute query');
      console.error('Query execution error:', err);
    } finally {
      setExecuting(false);
    }
  }, [query, queryParams, onQueryExecute, onQueryGenerated]);

  const handleSave = useCallback(() => {
    if (!saveName.trim() || !query.trim()) {
      setError('Name and query are required');
      return;
    }

    const newQuery: QueryTemplate = {
      id: Date.now().toString(),
      name: saveName,
      description: 'User-saved query',
      query: query,
      category: 'basic',
    };

    setSavedQueries(prev => [...prev, newQuery]);
    setSaveDialogOpen(false);
    setSaveName('');
  }, [saveName, query]);

  const handleDelete = useCallback((queryId: string) => {
    setSavedQueries(prev => prev.filter(q => q.id !== queryId));
  }, []);

  // Extract parameters from query
  const queryParamNames = query.match(/\$(\w+)/g)?.map(p => p.substring(1)) || [];
  const uniqueParams = [...new Set(queryParamNames)];

  return (
    <Box>
      <Paper sx={{ p: 2, mb: 2 }}>
        <Typography variant="h6" gutterBottom>
          Visual Query Builder
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Build Cypher queries using templates or create custom queries
        </Typography>

        {/* Template Selection */}
        <FormControl fullWidth sx={{ mb: 2 }}>
          <InputLabel>Query Template</InputLabel>
          <Select
            value={selectedTemplate}
            label="Query Template"
            onChange={(e) => handleTemplateSelect(e.target.value)}
          >
            <MenuItem value="">Custom Query</MenuItem>
            {QUERY_TEMPLATES.map(template => (
              <MenuItem key={template.id} value={template.id}>
                [{template.category}] {template.name} - {template.description}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        {/* Query Editor */}
        <TextField
          fullWidth
          multiline
          rows={8}
          label="Cypher Query"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="MATCH (n:Node) RETURN n LIMIT 10"
          sx={{ mb: 2 }}
          helperText={`${nodes.length} nodes, ${edges.length} edges available`}
        />

        {/* Query Parameters */}
        {uniqueParams.length > 0 && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Query Parameters
            </Typography>
            <Grid container spacing={2}>
              {uniqueParams.map(param => (
                <Grid item xs={12} sm={6} key={param}>
                  <TextField
                    fullWidth
                    label={param}
                    value={queryParams[param] || ''}
                    onChange={(e) => setQueryParams(prev => ({ ...prev, [param]: e.target.value }))}
                    size="small"
                    placeholder={`Enter value for ${param}`}
                  />
                </Grid>
              ))}
            </Grid>
          </Box>
        )}

        {/* Actions */}
        <Stack direction="row" spacing={1}>
          <Button
            variant="contained"
            startIcon={executing ? <CircularProgress size={16} /> : <PlayArrowIcon />}
            onClick={handleExecute}
            disabled={executing || !query.trim()}
          >
            {executing ? 'Executing...' : 'Execute Query'}
          </Button>
          <Button
            variant="outlined"
            startIcon={<SaveIcon />}
            onClick={() => setSaveDialogOpen(true)}
            disabled={!query.trim()}
          >
            Save Query
          </Button>
        </Stack>
      </Paper>

      {/* Saved Queries */}
      {savedQueries.length > 0 && (
        <Paper sx={{ p: 2, mb: 2 }}>
          <Typography variant="h6" gutterBottom>
            Saved Queries
          </Typography>
          <List>
            {savedQueries.map((savedQuery) => (
              <ListItem
                key={savedQuery.id}
                secondaryAction={
                  <IconButton edge="end" onClick={() => handleDelete(savedQuery.id)}>
                    <DeleteIcon />
                  </IconButton>
                }
              >
                <ListItemText
                  primary={savedQuery.name}
                  secondary={savedQuery.query}
                />
                <Button
                  size="small"
                  onClick={() => {
                    setQuery(savedQuery.query);
                    setSelectedTemplate('');
                  }}
                  sx={{ ml: 2 }}
                >
                  Load
                </Button>
              </ListItem>
            ))}
          </List>
        </Paper>
      )}

      {/* Results */}
      {results && (
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Query Results
          </Typography>
          <Box
            sx={{
              p: 2,
              bgcolor: 'grey.50',
              borderRadius: 1,
              maxHeight: 400,
              overflow: 'auto',
            }}
          >
            <pre style={{ margin: 0, fontSize: '0.875rem' }}>
              {JSON.stringify(results, null, 2)}
            </pre>
          </Box>
        </Paper>
      )}

      {error && (
        <Alert severity="error" sx={{ mt: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Save Dialog */}
      <Dialog open={saveDialogOpen} onClose={() => setSaveDialogOpen(false)}>
        <DialogTitle>Save Query</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Query Name"
            fullWidth
            value={saveName}
            onChange={(e) => setSaveName(e.target.value)}
            placeholder="My Custom Query"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSaveDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleSave} variant="contained">
            Save
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

