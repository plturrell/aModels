/**
 * AIGraphAssistant Component
 * 
 * AI-powered assistant for graph exploration and analysis
 */

import React, { useState } from 'react';
import { Box, TextField, Button, Typography, Paper, List, ListItem, ListItemText, Chip } from '@mui/material';
import { useToast } from '../hooks/useToast';

interface AIGraphAssistantProps {
  nodes: any[];
  edges: any[];
  projectId?: string;
  onNodeClick?: (nodeId: string) => void;
  onQueryGenerated?: (query: string) => void;
}

export function AIGraphAssistant({
  nodes,
  edges,
  projectId,
  onNodeClick,
  onQueryGenerated
}: AIGraphAssistantProps) {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const toast = useToast();

  const handleAskQuestion = async () => {
    if (!question.trim()) return;

    setLoading(true);
    try {
      // Mock AI responses based on graph data
      const mockSuggestions = [
        `Find all nodes connected to ${nodes[0]?.label || 'the first node'}`,
        'Show me the shortest path between any two nodes',
        'Find nodes with the highest degree centrality',
        'Identify clusters or communities in the graph'
      ];
      
      setSuggestions(mockSuggestions);
      toast.success('AI suggestions generated');
    } catch (error) {
      toast.error('Failed to generate suggestions');
    } finally {
      setLoading(false);
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    // Convert suggestion to Cypher query
    const query = `MATCH (n) RETURN n LIMIT 10`; // Placeholder
    onQueryGenerated?.(query);
  };

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        AI Graph Assistant
      </Typography>
      
      <Box sx={{ mb: 3 }}>
        <TextField
          fullWidth
          multiline
          rows={3}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask me anything about your graph..."
          variant="outlined"
        />
        <Button
          variant="contained"
          onClick={handleAskQuestion}
          disabled={loading || !question.trim()}
          sx={{ mt: 2 }}
        >
          {loading ? 'Thinking...' : 'Ask AI'}
        </Button>
      </Box>

      {suggestions.length > 0 && (
        <Box>
          <Typography variant="h6" gutterBottom>
            Suggested Queries
          </Typography>
          <List>
            {suggestions.map((suggestion, index) => (
              <ListItem key={index} disablePadding>
                <Button
                  onClick={() => handleSuggestionClick(suggestion)}
                  sx={{ textAlign: 'left', textTransform: 'none' }}
                >
                  <ListItemText primary={suggestion} />
                </Button>
              </ListItem>
            ))}
          </List>
        </Box>
      )}

      <Box sx={{ mt: 3 }}>
        <Typography variant="body2" color="text.secondary">
          Graph Stats: {nodes.length} nodes, {edges.length} edges
          {projectId && ` in project ${projectId}`}
        </Typography>
      </Box>
    </Paper>
  );
}
