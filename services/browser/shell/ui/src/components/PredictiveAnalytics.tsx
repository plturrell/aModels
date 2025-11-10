/**
 * PredictiveAnalytics Component
 * Uses LocalAI to generate predictions from analytics data
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
  Chip,
  List,
  ListItem,
  ListItemText,
  Divider
} from "@mui/material";
import TrendingUpIcon from "@mui/icons-material/TrendingUp";
import SmartToyIcon from "@mui/icons-material/SmartToy";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ExpandLessIcon from "@mui/icons-material/ExpandLess";
import { useLocalAIChatStore } from "../state/useLocalAIChatStore";
import { useServiceContext } from "../hooks/useServiceContext";

interface PredictiveAnalyticsProps {
  analyticsData: any; // Analytics data to make predictions from
  onPrediction?: (prediction: string) => void;
}

export function PredictiveAnalytics({ analyticsData, onPrediction }: PredictiveAnalyticsProps) {
  const [expanded, setExpanded] = useState(false);
  const [question, setQuestion] = useState("");
  const [predictions, setPredictions] = useState<Array<{ question: string; answer: string; timestamp: number }>>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { sendMessage, pending } = useLocalAIChatStore();
  const { getContextForLocalAI } = useServiceContext();

  const handlePredict = async (customQuestion?: string) => {
    const query = customQuestion || question;
    if (!query.trim() || loading) return;

    setLoading(true);
    setError(null);

    try {
      // Format analytics data as context
      const analyticsContext = JSON.stringify({
        total_requests: analyticsData?.total || 0,
        completed: analyticsData?.completed || 0,
        failed: analyticsData?.failed || 0,
        success_rate: analyticsData?.successRate || 0,
        avg_processing_time: analyticsData?.avgProcessingTime || 0,
        status_distribution: analyticsData?.statusDistribution || [],
        timeline_data: analyticsData?.timelineData || [],
      }, null, 2);

      // Get cross-service context
      const crossServiceContext = getContextForLocalAI();

      // Create a predictive prompt
      const prompt = `Based on the following analytics data and trends, ${query}\n\nAnalytics Data:\n${analyticsContext}\n\n${crossServiceContext ? `Additional Context:\n${crossServiceContext}\n\n` : ''}Please provide predictions, forecasts, and what-if scenarios. Include:\n1. What will happen if current trends continue?\n2. Potential risks or opportunities\n3. Recommended actions based on predictions\n4. Confidence level in the predictions`;

      await sendMessage(prompt, { stream: true });

      setTimeout(() => {
        const messages = useLocalAIChatStore.getState().messages;
        const lastMessage = messages[messages.length - 1];
        if (lastMessage && lastMessage.role === "assistant") {
          const prediction = {
            question: query,
            answer: lastMessage.content,
            timestamp: Date.now()
          };
          setPredictions(prev => [prediction, ...prev].slice(0, 5)); // Keep last 5
          if (onPrediction) {
            onPrediction(lastMessage.content);
          }
        }
        setLoading(false);
      }, 2000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate prediction");
      setLoading(false);
    }
  };

  const quickQuestions = [
    "What will happen if this trend continues?",
    "What are the potential risks?",
    "What opportunities do you see?",
    "What should I do next?",
    "What if success rate drops by 10%?",
    "What if we double the request volume?"
  ];

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
          <TrendingUpIcon color="primary" />
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Predictive Analytics
          </Typography>
          {predictions.length > 0 && (
            <Chip label={`${predictions.length} predictions`} size="small" />
          )}
        </Box>
        <IconButton size="small">
          {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
        </IconButton>
      </Box>

      <Collapse in={expanded}>
        <Box>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Ask predictive questions about your analytics data. Get forecasts, what-if scenarios, and actionable insights.
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
              placeholder="Ask a predictive question..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyPress={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handlePredict();
                }
              }}
              disabled={loading || pending}
              multiline
              maxRows={3}
            />
            <Button
              variant="contained"
              onClick={() => handlePredict()}
              disabled={!question.trim() || loading || pending}
              sx={{ minWidth: 100 }}
            >
              {loading || pending ? (
                <CircularProgress size={20} color="inherit" />
              ) : (
                <>
                  <SmartToyIcon sx={{ mr: 0.5, fontSize: 18 }} />
                  Predict
                </>
              )}
            </Button>
          </Box>

          {/* Quick Questions */}
          <Box sx={{ mb: 2 }}>
            <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: "block" }}>
              Quick Questions:
            </Typography>
            <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1 }}>
              {quickQuestions.map((q, idx) => (
                <Chip
                  key={idx}
                  label={q}
                  size="small"
                  onClick={() => handlePredict(q)}
                  disabled={loading || pending}
                  sx={{ cursor: "pointer" }}
                />
              ))}
            </Box>
          </Box>

          <Divider sx={{ my: 2 }} />

          {/* Predictions History */}
          {predictions.length > 0 && (
            <Box>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                Recent Predictions:
              </Typography>
              <List dense>
                {predictions.map((pred, idx) => (
                  <ListItem
                    key={idx}
                    sx={{
                      flexDirection: "column",
                      alignItems: "flex-start",
                      bgcolor: "background.default",
                      mb: 1,
                      borderRadius: 1,
                      p: 1.5
                    }}
                  >
                    <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.5 }}>
                      {pred.question}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ whiteSpace: "pre-wrap", fontSize: "0.875rem" }}>
                      {pred.answer.substring(0, 300)}{pred.answer.length > 300 ? "..." : ""}
                    </Typography>
                    <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5 }}>
                      {new Date(pred.timestamp).toLocaleString()}
                    </Typography>
                  </ListItem>
                ))}
              </List>
            </Box>
          )}

          {loading && predictions.length === 0 && (
            <Box sx={{ display: "flex", alignItems: "center", gap: 1, p: 2 }}>
              <CircularProgress size={16} />
              <Typography variant="body2" color="text.secondary">
                Generating prediction...
              </Typography>
            </Box>
          )}

          {!loading && predictions.length === 0 && (
            <Typography variant="body2" color="text.secondary" sx={{ textAlign: "center", py: 2 }}>
              Ask a predictive question to get AI-powered forecasts and insights.
            </Typography>
          )}
        </Box>
      </Collapse>
    </Paper>
  );
}

