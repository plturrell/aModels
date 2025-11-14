/**
 * NaturalLanguageGraphQuery Component
 * 
 * Natural language interface for graph queries
 */

import React, { useState } from 'react';
import { Box, TextField, Button, Typography, Paper, Alert } from '@mui/material';
import { useToast } from '../hooks/useToast';

interface NaturalLanguageGraphQueryProps {
  onQueryGenerated?: (cypher: string, results: any[]) => void;
  onError?: (error: string) => void;
}

export function NaturalLanguageGraphQuery({
  onQueryGenerated,
  onError
}: NaturalLanguageGraphQueryProps) {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const toast = useToast();

  const handleQuery = async () => {
    if (!query.trim()) return;

    setLoading(true);
    try {
      // Mock implementation - replace with actual API call
      const mockCypher = `MATCH (n) WHERE n.name CONTAINS "${query}" RETURN n LIMIT 10`;
      const mockResults = [
        { id: '1', name: 'Mock Node 1', type: 'Person' },
        { id: '2', name: 'Mock Node 2', type: 'Organization' }
      ];
      
      toast.success('Query generated successfully');
      onQueryGenerated?.(mockCypher, mockResults);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to generate query';
      toast.error(errorMessage);
      onError?.(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        Natural Language Graph Query
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Ask questions about your graph in plain English
      </Typography>
      
      <Box sx={{ display: 'flex', gap: 2, alignItems: 'flex-start' }}>
        <TextField
          fullWidth
          multiline
          rows={3}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g., Find all people connected to organizations in the tech industry"
          variant="outlined"
        />
        <Button
          variant="contained"
          onClick={handleQuery}
          disabled={loading || !query.trim()}
          sx={{ minWidth: 100 }}
        >
          {loading ? 'Processing...' : 'Query'}
        </Button>
      </Box>

      <Alert severity="info" sx={{ mt: 2 }}>
        This feature uses AI to convert natural language to Cypher queries
      </Alert>
    </Paper>
  );
}
