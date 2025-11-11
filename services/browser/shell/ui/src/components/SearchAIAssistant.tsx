/**
 * SearchAIAssistant Component
 * AI-enhanced search with query understanding, suggestions, and result summarization
 */

import React, { useState, useEffect } from "react";
import {
  Box,
  TextField,
  Button,
  Paper,
  Typography,
  Collapse,
  IconButton,
  CircularProgress,
  Alert,
  Chip,
  List,
  ListItem,
  ListItemText,
  Divider
} from "@mui/material";
import SendIcon from "@mui/icons-material/Send";
import SmartToyIcon from "@mui/icons-material/SmartToy";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ExpandLessIcon from "@mui/icons-material/ExpandLess";
import LightbulbIcon from "@mui/icons-material/Lightbulb";
import { useLocalAIChatStore } from "../state/useLocalAIChatStore";
import { useServiceContext } from "../hooks/useServiceContext";
import type { UnifiedSearchResult, UnifiedSearchResponse } from "../api/search";

interface SearchAIAssistantProps {
  query: string;
  searchResults: UnifiedSearchResult[];
  searchResponse: UnifiedSearchResponse | null;
  onQueryRefinement?: (refinedQuery: string) => void;
  onSummarize?: (summary: string) => void;
}

export function SearchAIAssistant({
  query,
  searchResults,
  searchResponse,
  onQueryRefinement,
  onSummarize
}: SearchAIAssistantProps) {
  const [expanded, setExpanded] = useState(false);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [summary, setSummary] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [refinedQuery, setRefinedQuery] = useState<string | null>(null);

  const { sendMessage, pending } = useLocalAIChatStore();
  const { setSearchContext, getContextForLocalAI } = useServiceContext();

  // Generate search suggestions when query changes
  useEffect(() => {
    if (query.trim() && query.length > 3) {
      generateSuggestions(query);
    }
  }, [query]);

  // Summarize results when they change
  useEffect(() => {
    if (searchResults.length > 0 && expanded) {
      generateSummary();
    }
  }, [searchResults, expanded]);

  const generateSuggestions = async (currentQuery: string) => {
    try {
      // Get cross-service context for better suggestions
      const crossServiceContext = getContextForLocalAI();
      
      const prompt = `Based on the search query "${currentQuery}", suggest 3-5 improved or related search queries that might yield better results. ${crossServiceContext && crossServiceContext !== "No context available." ? `Consider this context: ${crossServiceContext}` : ''} Return only the queries, one per line, without numbering or bullets.`;
      
      await sendMessage(prompt, { stream: false });
      
      setTimeout(() => {
        const messages = useLocalAIChatStore.getState().messages;
        const lastMessage = messages[messages.length - 1];
        if (lastMessage && lastMessage.role === "assistant") {
          const lines = lastMessage.content
            .split('\n')
            .map(line => line.trim())
            .filter(line => line.length > 0 && !line.match(/^\d+[\.\)]/))
            .slice(0, 5);
          setSuggestions(lines);
        }
      }, 1000);
    } catch (err) {
      console.error("Failed to generate suggestions:", err);
    }
  };

  const generateSummary = async () => {
    if (loading) return;
    
    setLoading(true);
    setError(null);

    try {
      // Update service context
      setSearchContext({
        query: query,
        results: searchResults
      });

      // Get cross-service context
      const crossServiceContext = getContextForLocalAI();

      const resultsContext = searchResults
        .slice(0, 10)
        .map((r, idx) => `${idx + 1}. ${r.title || r.content?.substring(0, 100)}...`)
        .join('\n');

      const prompt = `Summarize the following search results for the query "${query}":\n\n${resultsContext}\n\n${crossServiceContext ? `Additional Context:\n${crossServiceContext}\n\n` : ''}Provide a concise summary highlighting key findings and patterns.`;

      await sendMessage(prompt, { stream: true });

      setTimeout(() => {
        const messages = useLocalAIChatStore.getState().messages;
        const lastMessage = messages[messages.length - 1];
        if (lastMessage && lastMessage.role === "assistant") {
          setSummary(lastMessage.content);
          if (onSummarize) {
            onSummarize(lastMessage.content);
          }
        }
        setLoading(false);
      }, 2000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate summary");
      setLoading(false);
    }
  };

  const refineQuery = async (originalQuery: string) => {
    setLoading(true);
    setError(null);

    try {
      // Get cross-service context for better refinement
      const crossServiceContext = getContextForLocalAI();
      
      const prompt = `Refine and improve the following search query to get better results: "${originalQuery}". ${crossServiceContext && crossServiceContext !== "No context available." ? `Consider this context when refining: ${crossServiceContext}` : ''} Return only the refined query, nothing else.`;

      await sendMessage(prompt, { stream: false });

      setTimeout(() => {
        const messages = useLocalAIChatStore.getState().messages;
        const lastMessage = messages[messages.length - 1];
        if (lastMessage && lastMessage.role === "assistant") {
          const refined = lastMessage.content.trim().replace(/^["']|["']$/g, '');
          setRefinedQuery(refined);
          if (onQueryRefinement) {
            onQueryRefinement(refined);
          }
        }
        setLoading(false);
      }, 1000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to refine query");
      setLoading(false);
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    if (onQueryRefinement) {
      onQueryRefinement(suggestion);
    }
  };

  return (
    <Paper
      sx={{
        p: 2,
        mb: 2,
        bgcolor: "background.paper",
        border: "1px solid",
        borderColor: "divider",
      }}
    >
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          mb: expanded ? 2 : 0,
          cursor: "pointer",
        }}
        onClick={() => setExpanded(!expanded)}
      >
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <SmartToyIcon color="primary" />
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            AI Search Assistant
          </Typography>
          {suggestions.length > 0 && (
            <Chip label={`${suggestions.length} suggestions`} size="small" />
          )}
        </Box>
        <IconButton size="small">
          {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
        </IconButton>
      </Box>

      <Collapse in={expanded}>
        <Box>
          {error && (
            <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
              {error}
            </Alert>
          )}

          {/* Query Refinement */}
          {query && (
            <Box sx={{ mb: 2 }}>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}>
                <LightbulbIcon fontSize="small" color="primary" />
                <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                  Query Refinement
                </Typography>
              </Box>
              <Box sx={{ display: "flex", gap: 1, mb: 1 }}>
                <Button
                  size="small"
                  variant="outlined"
                  onClick={() => refineQuery(query)}
                  disabled={loading || pending}
                >
                  {loading ? <CircularProgress size={16} /> : "Refine Query"}
                </Button>
              </Box>
              {refinedQuery && refinedQuery !== query && (
                <Alert severity="info" sx={{ mt: 1 }}>
                  <Typography variant="body2">
                    <strong>Refined query:</strong> {refinedQuery}
                  </Typography>
                  <Button
                    size="small"
                    onClick={() => onQueryRefinement?.(refinedQuery)}
                    sx={{ mt: 1 }}
                  >
                    Use This Query
                  </Button>
                </Alert>
              )}
            </Box>
          )}

          {/* Search Suggestions */}
          {suggestions.length > 0 && (
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                Suggested Queries:
              </Typography>
              <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1 }}>
                {suggestions.map((suggestion, idx) => (
                  <Chip
                    key={idx}
                    label={suggestion}
                    size="small"
                    onClick={() => handleSuggestionClick(suggestion)}
                    sx={{ cursor: "pointer" }}
                  />
                ))}
              </Box>
            </Box>
          )}

          <Divider sx={{ my: 2 }} />

          {/* Result Summary */}
          {searchResults.length > 0 && (
            <Box>
              <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 1 }}>
                <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                  Result Summary
                </Typography>
                <Button
                  size="small"
                  onClick={generateSummary}
                  disabled={loading || pending}
                  startIcon={loading ? <CircularProgress size={16} /> : <SmartToyIcon />}
                >
                  {summary ? "Regenerate" : "Generate Summary"}
                </Button>
              </Box>

              {summary && (
                <Paper
                  sx={{
                    p: 2,
                    bgcolor: "primary.50",
                    border: "1px solid",
                    borderColor: "primary.200",
                  }}
                >
                  <Typography variant="body2" sx={{ whiteSpace: "pre-wrap" }}>
                    {summary}
                  </Typography>
                </Paper>
              )}

              {loading && !summary && (
                <Box sx={{ display: "flex", alignItems: "center", gap: 1, p: 2 }}>
                  <CircularProgress size={16} />
                  <Typography variant="body2" color="text.secondary">
                    Generating summary...
                  </Typography>
                </Box>
              )}

              {!summary && !loading && (
                <Typography variant="body2" color="text.secondary">
                  Click "Generate Summary" to get AI-powered insights about your search results.
                </Typography>
              )}
            </Box>
          )}

          {/* Answer Questions */}
          {searchResults.length > 0 && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                Ask About Results
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                Ask questions about the search results to get AI-powered answers.
              </Typography>
              <Box sx={{ display: "flex", gap: 1 }}>
                <Chip
                  label="What are the key findings?"
                  size="small"
                  onClick={() => {
                    const prompt = `Based on these search results, what are the key findings?`;
                    sendMessage(prompt);
                  }}
                  sx={{ cursor: "pointer" }}
                />
                <Chip
                  label="What patterns do you see?"
                  size="small"
                  onClick={() => {
                    const prompt = `What patterns or trends do you see in these search results?`;
                    sendMessage(prompt);
                  }}
                  sx={{ cursor: "pointer" }}
                />
              </Box>
            </Box>
          )}
        </Box>
      </Collapse>
    </Paper>
  );
}

