/**
 * AnalyticsAIAssistant Component
 * Allows users to ask natural language questions about analytics data
 */

import React, { useState } from "react";
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
  Chip
} from "@mui/material";
import SendIcon from "@mui/icons-material/Send";
import SmartToyIcon from "@mui/icons-material/SmartToy";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ExpandLessIcon from "@mui/icons-material/ExpandLess";
import { useLocalAIChatStore } from "../state/useLocalAIChatStore";
import { useServiceContext } from "../hooks/useServiceContext";

interface AnalyticsAIAssistantProps {
  analyticsData: any; // Analytics data to provide context
  onInsight?: (insight: string) => void;
}

export function AnalyticsAIAssistant({ analyticsData, onInsight }: AnalyticsAIAssistantProps) {
  const [expanded, setExpanded] = useState(false);
  const [query, setQuery] = useState("");
  const [insight, setInsight] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { sendMessage, pending } = useLocalAIChatStore();
  const { setAnalyticsContext, getContextForLocalAI } = useServiceContext();

  const handleAsk = async () => {
    if (!query.trim() || loading) return;

    setLoading(true);
    setError(null);
    setInsight(null);

    try {
      // Format analytics data as context
      const analyticsContext = JSON.stringify({
        total_requests: analyticsData?.total || 0,
        completed: analyticsData?.completed || 0,
        failed: analyticsData?.failed || 0,
        success_rate: analyticsData?.successRate || 0,
        avg_processing_time: analyticsData?.avgProcessingTime || 0,
        status_distribution: analyticsData?.statusDistribution || [],
      }, null, 2);

      // Update service context
      setAnalyticsContext({
        data: analyticsData,
        query: query
      });

      // Get cross-service context
      const crossServiceContext = getContextForLocalAI();

      // Create a prompt with analytics context
      const prompt = `Based on the following analytics data, ${query}\n\nAnalytics Data:\n${analyticsContext}\n\n${crossServiceContext ? `Additional Context:\n${crossServiceContext}\n\n` : ''}Please provide insights, trends, and recommendations.`;

      // Use LocalAI to generate insights
      await sendMessage(prompt, { stream: true });

      // The store will handle the response, but we can also extract insights
      // For now, we'll wait for the response and show it
      setTimeout(() => {
        const messages = useLocalAIChatStore.getState().messages;
        const lastMessage = messages[messages.length - 1];
        if (lastMessage && lastMessage.role === "assistant") {
          setInsight(lastMessage.content);
          if (onInsight) {
            onInsight(lastMessage.content);
          }
        }
        setLoading(false);
      }, 2000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate insights");
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleAsk();
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
            Ask AI About Analytics
          </Typography>
        </Box>
        <IconButton size="small">
          {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
        </IconButton>
      </Box>

      <Collapse in={expanded}>
        <Box>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Ask questions about your analytics data in natural language. For example:
            "What's the trend in success rate?" or "Why are there so many failed requests?"
          </Typography>

          {error && (
            <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
              {error}
            </Alert>
          )}

          <Box sx={{ display: "flex", gap: 1, mb: 2 }}>
            <TextField
              fullWidth
              size="small"
              placeholder="Ask about your analytics data..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={loading || pending}
              multiline
              maxRows={3}
            />
            <Button
              variant="contained"
              onClick={handleAsk}
              disabled={!query.trim() || loading || pending}
              sx={{ minWidth: 100 }}
            >
              {loading || pending ? (
                <CircularProgress size={20} color="inherit" />
              ) : (
                <>
                  <SendIcon sx={{ mr: 0.5 }} />
                  Ask
                </>
              )}
            </Button>
          </Box>

          {insight && (
            <Paper
              sx={{
                p: 2,
                bgcolor: "primary.50",
                border: "1px solid",
                borderColor: "primary.200",
              }}
            >
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                AI Insights:
              </Typography>
              <Typography variant="body2" sx={{ whiteSpace: "pre-wrap" }}>
                {insight}
              </Typography>
            </Paper>
          )}

          {pending && !insight && (
            <Box sx={{ display: "flex", alignItems: "center", gap: 1, p: 2 }}>
              <CircularProgress size={16} />
              <Typography variant="body2" color="text.secondary">
                Generating insights...
              </Typography>
            </Box>
          )}

          <Box sx={{ mt: 2, display: "flex", flexWrap: "wrap", gap: 1 }}>
            <Chip
              label="What's the trend?"
              size="small"
              onClick={() => setQuery("What's the trend in the analytics data?")}
              sx={{ cursor: "pointer" }}
            />
            <Chip
              label="Why failures?"
              size="small"
              onClick={() => setQuery("Why are there failed requests?")}
              sx={{ cursor: "pointer" }}
            />
            <Chip
              label="Performance insights"
              size="small"
              onClick={() => setQuery("What are the performance insights?")}
              sx={{ cursor: "pointer" }}
            />
          </Box>
        </Box>
      </Collapse>
    </Paper>
  );
}

