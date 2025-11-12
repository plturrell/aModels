/**
 * Phase 2.2: Natural Language Graph Query Component
 * 
 * Convert natural language queries to Cypher and execute them
 */

import React, { useState, useCallback } from 'react';
import {
  Box,
  Paper,
  TextField,
  Button,
  Typography,
  Alert,
  CircularProgress,
  Chip,
  Stack,
  Divider,
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import { queryGraph, GraphQueryResponse } from '../api/graph';

export interface NaturalLanguageGraphQueryProps {
  onQueryGenerated?: (cypher: string, results: GraphQueryResponse) => void;
  onError?: (error: string) => void;
}

const EXAMPLE_QUERIES = [
  "Show me all tables connected to customer",
  "Find paths from source to target",
  "What's related to this entity?",
  "Show me anomalies in the graph",
  "Find all nodes with more than 10 connections",
  "Show me the most connected nodes",
  "What tables are related to orders?",
  "Find all relationships involving customer",
];

// Simple natural language to Cypher converter
// In production, this would use LocalAI/LLM for better understanding
function convertNLToCypher(query: string): string {
  const lowerQuery = query.toLowerCase();
  
  // Pattern matching for common queries
  if (lowerQuery.includes('all tables') || lowerQuery.includes('show me all tables')) {
    return "MATCH (n:Node {type: 'table'}) RETURN n LIMIT 100";
  }
  
  if (lowerQuery.includes('connected to') || lowerQuery.includes('related to')) {
    const entityMatch = query.match(/connected to (\w+)|related to (\w+)/i);
    if (entityMatch) {
      const entity = entityMatch[1] || entityMatch[2];
      return `MATCH (n:Node)-[r:RELATIONSHIP]-(m:Node) WHERE n.label CONTAINS '${entity}' OR m.label CONTAINS '${entity}' RETURN n, r, m LIMIT 100`;
    }
  }
  
  if (lowerQuery.includes('path') || lowerQuery.includes('paths')) {
    if (lowerQuery.includes('from') && lowerQuery.includes('to')) {
      const fromMatch = query.match(/from (\w+)/i);
      const toMatch = query.match(/to (\w+)/i);
      if (fromMatch && toMatch) {
        return `MATCH path = shortestPath((source:Node)-[*]-(target:Node)) WHERE source.label CONTAINS '${fromMatch[1]}' AND target.label CONTAINS '${toMatch[1]}' RETURN path LIMIT 10`;
      }
    }
    return "MATCH path = (n:Node)-[*1..3]-(m:Node) RETURN path LIMIT 50";
  }
  
  if (lowerQuery.includes('anomal') || lowerQuery.includes('unusual')) {
    return "MATCH (n:Node) WHERE size((n)-[:RELATIONSHIP]-()) < 2 OR size((n)-[:RELATIONSHIP]-()) > 50 RETURN n LIMIT 50";
  }
  
  if (lowerQuery.includes('most connected') || lowerQuery.includes('highest degree')) {
    return "MATCH (n:Node) WITH n, size((n)-[:RELATIONSHIP]-()) as degree ORDER BY degree DESC RETURN n, degree LIMIT 20";
  }
  
  if (lowerQuery.includes('more than') && lowerQuery.includes('connection')) {
    const numMatch = query.match(/more than (\d+)/i);
    if (numMatch) {
      const num = numMatch[1];
      return `MATCH (n:Node) WHERE size((n)-[:RELATIONSHIP]-()) > ${num} RETURN n LIMIT 100`;
    }
  }
  
  // Default: search by label
  const words = query.split(/\s+/).filter(w => w.length > 3);
  if (words.length > 0) {
    const searchTerm = words[0];
    return `MATCH (n:Node) WHERE n.label CONTAINS '${searchTerm}' OR n.id CONTAINS '${searchTerm}' RETURN n LIMIT 50`;
  }
  
  return "MATCH (n:Node) RETURN n LIMIT 10";
}

export function NaturalLanguageGraphQuery({
  onQueryGenerated,
  onError,
}: NaturalLanguageGraphQueryProps) {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [generatedCypher, setGeneratedCypher] = useState<string | null>(null);
  const [results, setResults] = useState<GraphQueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleQuery = useCallback(async () => {
    if (!query.trim()) {
      setError('Please enter a query');
      return;
    }

    setLoading(true);
    setError(null);
    setGeneratedCypher(null);
    setResults(null);

    try {
      // Convert natural language to Cypher
      const cypher = convertNLToCypher(query);
      setGeneratedCypher(cypher);

      // Execute the query
      const queryResults = await queryGraph({ query: cypher });
      setResults(queryResults);

      if (onQueryGenerated) {
        onQueryGenerated(cypher, queryResults);
      }
    } catch (err: any) {
      const errorMsg = err.message || 'Failed to execute query';
      setError(errorMsg);
      if (onError) {
        onError(errorMsg);
      }
    } finally {
      setLoading(false);
    }
  }, [query, onQueryGenerated, onError]);

  const handleExampleClick = useCallback((example: string) => {
    setQuery(example);
  }, []);

  return (
    <Paper sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
        <AutoAwesomeIcon color="primary" />
        <Typography variant="h6">
          Natural Language Graph Query
        </Typography>
      </Box>

      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Ask questions about your graph in plain English. The system will convert your query to Cypher and execute it.
      </Typography>

      <Box sx={{ mb: 2 }}>
        <TextField
          fullWidth
          multiline
          rows={3}
          placeholder="e.g., Show me all tables connected to customer"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
              handleQuery();
            }
          }}
          disabled={loading}
        />
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 1 }}>
          <Typography variant="caption" color="text.secondary">
            Press Ctrl+Enter to execute
          </Typography>
          <Button
            variant="contained"
            startIcon={loading ? <CircularProgress size={16} /> : <SendIcon />}
            onClick={handleQuery}
            disabled={loading || !query.trim()}
          >
            {loading ? 'Querying...' : 'Execute Query'}
          </Button>
        </Box>
      </Box>

      {/* Example Queries */}
      <Box sx={{ mb: 2 }}>
        <Typography variant="subtitle2" gutterBottom>
          Example Queries:
        </Typography>
        <Stack direction="row" spacing={1} sx={{ flexWrap: 'wrap', gap: 1 }}>
          {EXAMPLE_QUERIES.map((example, idx) => (
            <Chip
              key={idx}
              label={example}
              onClick={() => handleExampleClick(example)}
              size="small"
              variant="outlined"
              sx={{ cursor: 'pointer' }}
            />
          ))}
        </Stack>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {generatedCypher && (
        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Generated Cypher Query:
          </Typography>
          <Paper
            variant="outlined"
            sx={{
              p: 2,
              bgcolor: 'grey.50',
              fontFamily: 'monospace',
              fontSize: '0.875rem',
              overflow: 'auto',
            }}
          >
            {generatedCypher}
          </Paper>
        </Box>
      )}

      {results && (
        <Box>
          <Divider sx={{ my: 2 }} />
          <Typography variant="subtitle2" gutterBottom>
            Results ({results.data.length} rows, {results.execution_time_ms}ms):
          </Typography>
          {results.columns && (
            <Box sx={{ mb: 1 }}>
              <Typography variant="caption" color="text.secondary">
                Columns: {results.columns.join(', ')}
              </Typography>
            </Box>
          )}
          <Paper
            variant="outlined"
            sx={{
              p: 2,
              maxHeight: 300,
              overflow: 'auto',
              bgcolor: 'grey.50',
            }}
          >
            <pre style={{ margin: 0, fontSize: '0.75rem' }}>
              {JSON.stringify(results.data.slice(0, 10), null, 2)}
              {results.data.length > 10 && `\n... and ${results.data.length - 10} more rows`}
            </pre>
          </Paper>
        </Box>
      )}
    </Paper>
  );
}

