/**
 * Phase 4.1: AI Graph Assistant Component
 * 
 * Chat interface for graph exploration with graph-aware LocalAI integration
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  List,
  ListItem,
  ListItemText,
  Avatar,
  IconButton,
  Chip,
  CircularProgress,
  Alert,
  Stack,
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import PersonIcon from '@mui/icons-material/Person';
import ClearIcon from '@mui/icons-material/Clear';
import { GraphNode, GraphEdge } from '../types/graph';

const API_BASE = import.meta.env.VITE_GATEWAY_URL || 'http://localhost:8000';
const LOCALAI_URL = import.meta.env.VITE_LOCALAI_URL || 'http://localhost:8080';

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  graphContext?: {
    nodes?: string[];
    edges?: string[];
    query?: string;
  };
}

export interface AIGraphAssistantProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  projectId?: string;
  onNodeClick?: (nodeId: string) => void;
  onQueryGenerated?: (query: string) => void;
}

const EXAMPLE_QUESTIONS = [
  "What should I explore next?",
  "Explain this relationship",
  "Find similar patterns",
  "What's connected to this entity?",
  "Show me anomalies in the graph",
  "What are the most important nodes?",
];

export function AIGraphAssistant({
  nodes,
  edges,
  projectId,
  onNodeClick,
  onQueryGenerated,
}: AIGraphAssistantProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = useCallback(async () => {
    if (!input.trim() || loading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);
    setError(null);

    try {
      // Build graph context for the AI
      const graphContext = {
        node_count: nodes.length,
        edge_count: edges.length,
        node_types: [...new Set(nodes.map(n => n.type).filter(Boolean))],
        sample_nodes: nodes.slice(0, 10).map(n => ({
          id: n.id,
          label: n.label,
          type: n.type,
        })),
      };

      // Call LocalAI with graph context
      const response = await fetch(`${LOCALAI_URL}/v1/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'gpt-4',
          messages: [
            {
              role: 'system',
              content: `You are an AI assistant helping users explore a knowledge graph. 
              
Current graph context:
- ${nodes.length} nodes, ${edges.length} edges
- Node types: ${graphContext.node_types.join(', ')}
- Project ID: ${projectId || 'N/A'}

You can help users:
1. Understand graph structure and relationships
2. Suggest exploration paths
3. Explain connections between entities
4. Find patterns and anomalies
5. Generate Cypher queries for graph exploration

Be conversational, helpful, and suggest specific actions when appropriate.`,
            },
            ...messages.map(m => ({
              role: m.role,
              content: m.content,
            })),
            {
              role: 'user',
              content: input.trim(),
            },
          ],
          temperature: 0.7,
          max_tokens: 500,
        }),
      });

      if (!response.ok) {
        throw new Error(`LocalAI request failed: ${response.statusText}`);
      }

      const data = await response.json();
      const assistantContent = data.choices?.[0]?.message?.content || 'I apologize, but I could not generate a response.';

      // Try to extract Cypher queries or node references from the response
      const cypherMatch = assistantContent.match(/MATCH[\s\S]*?(?=\n\n|$)/i);
      const nodeMatches = assistantContent.match(/node[_\s]+id[:\s]+([a-zA-Z0-9_-]+)/gi);

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: assistantContent,
        timestamp: new Date(),
        graphContext: {
          query: cypherMatch?.[0],
          nodes: nodeMatches?.map((m: string) => m.split(/[: ]+/).pop() || '').filter(Boolean),
        },
      };

      setMessages(prev => [...prev, assistantMessage]);

      // If a query was generated, notify parent
      if (cypherMatch && onQueryGenerated) {
        onQueryGenerated(cypherMatch[0]);
      }
    } catch (err: any) {
      setError(err.message || 'Failed to get AI response');
      console.error('AI Assistant error:', err);
      
      // Add error message
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `I encountered an error: ${err.message}. Please try again or check your LocalAI service connection.`,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  }, [input, loading, messages, nodes, edges, projectId, onQueryGenerated]);

  const handleExampleClick = useCallback((example: string) => {
    setInput(example);
  }, []);

  const handleClear = useCallback(() => {
    setMessages([]);
    setError(null);
  }, []);

  return (
    <Paper sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <SmartToyIcon color="primary" />
          <Typography variant="h6">AI Graph Assistant</Typography>
        </Box>
        {messages.length > 0 && (
          <IconButton size="small" onClick={handleClear}>
            <ClearIcon />
          </IconButton>
        )}
      </Box>

      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Ask questions about your graph. I can help you explore, understand relationships, and find patterns.
      </Typography>

      {/* Example Questions */}
      {messages.length === 0 && (
        <Box sx={{ mb: 2 }}>
          <Typography variant="caption" color="text.secondary" display="block" gutterBottom>
            Try asking:
          </Typography>
          <Stack direction="row" spacing={1} sx={{ flexWrap: 'wrap', gap: 1 }}>
            {EXAMPLE_QUESTIONS.map((question, idx) => (
              <Chip
                key={idx}
                label={question}
                onClick={() => handleExampleClick(question)}
                size="small"
                variant="outlined"
                sx={{ cursor: 'pointer' }}
              />
            ))}
          </Stack>
        </Box>
      )}

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Messages */}
      <Box sx={{ flex: 1, overflow: 'auto', mb: 2, minHeight: 300 }}>
        {messages.length === 0 ? (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <SmartToyIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
            <Typography variant="body2" color="text.secondary">
              Start a conversation to explore your graph
            </Typography>
          </Box>
        ) : (
          <List>
            {messages.map((message) => (
              <ListItem
                key={message.id}
                sx={{
                  flexDirection: message.role === 'user' ? 'row-reverse' : 'row',
                  alignItems: 'flex-start',
                }}
              >
                <Avatar
                  sx={{
                    bgcolor: message.role === 'user' ? 'primary.main' : 'secondary.main',
                    mr: message.role === 'user' ? 0 : 1,
                    ml: message.role === 'user' ? 1 : 0,
                  }}
                >
                  {message.role === 'user' ? <PersonIcon /> : <SmartToyIcon />}
                </Avatar>
                <Paper
                  sx={{
                    p: 1.5,
                    maxWidth: '70%',
                    bgcolor: message.role === 'user' ? 'primary.light' : 'grey.100',
                    color: message.role === 'user' ? 'primary.contrastText' : 'text.primary',
                  }}
                >
                  <ListItemText
                    primary={message.content}
                    secondary={
                      <Box sx={{ mt: 1 }}>
                        <Typography variant="caption" color="text.secondary">
                          {message.timestamp.toLocaleTimeString()}
                        </Typography>
                        {message.graphContext?.query && (
                          <Chip
                            label="Query Generated"
                            size="small"
                            sx={{ ml: 1 }}
                            onClick={() => onQueryGenerated?.(message.graphContext!.query!)}
                          />
                        )}
                        {message.graphContext?.nodes && message.graphContext.nodes.length > 0 && (
                          <Box sx={{ mt: 1 }}>
                            {message.graphContext.nodes.slice(0, 3).map((nodeId) => (
                              <Chip
                                key={nodeId}
                                label={nodeId}
                                size="small"
                                sx={{ mr: 0.5, mb: 0.5 }}
                                onClick={() => onNodeClick?.(nodeId)}
                              />
                            ))}
                          </Box>
                        )}
                      </Box>
                    }
                  />
                </Paper>
              </ListItem>
            ))}
            {loading && (
              <ListItem>
                <Avatar sx={{ bgcolor: 'secondary.main', mr: 1 }}>
                  <SmartToyIcon />
                </Avatar>
                <CircularProgress size={20} />
              </ListItem>
            )}
            <div ref={messagesEndRef} />
          </List>
        )}
      </Box>

      {/* Input */}
      <Box sx={{ display: 'flex', gap: 1 }}>
        <TextField
          fullWidth
          multiline
          maxRows={4}
          placeholder="Ask about your graph..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleSend();
            }
          }}
          disabled={loading}
          size="small"
        />
        <Button
          variant="contained"
          onClick={handleSend}
          disabled={loading || !input.trim()}
          startIcon={loading ? <CircularProgress size={16} /> : <SendIcon />}
        >
          Send
        </Button>
      </Box>
    </Paper>
  );
}

