/**
 * Perplexity Module - Ruthlessly Simplified
 * One action: Ask question
 * One view: Answer with sources
 */

import React, { useState } from "react";
import {
  Box,
  Typography,
  TextField,
  Button,
  Card,
  CardContent,
  CircularProgress,
  Chip
} from "@mui/material";

export function PerplexityModule() {
  const [question, setQuestion] = useState("");
  const [processing, setProcessing] = useState(false);
  const [answer, setAnswer] = useState<string | null>(null);
  const [sources, setSources] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  const handleAsk = async () => {
    if (!question.trim()) return;

    setProcessing(true);
    setError(null);
    setAnswer(null);
    setSources([]);

    // Simulate API call
    setTimeout(() => {
      setAnswer("This is a placeholder answer. The Perplexity API would return actual results here based on your question.");
      setSources(["Source 1", "Source 2", "Source 3"]);
      setProcessing(false);
    }, 2000);
  };

  return (
    <Box sx={{ p: 6, maxWidth: 1200, margin: "0 auto" }}>
      {/* Question Input */}
      <Box sx={{ mb: 6 }}>
        <Typography variant="h4" sx={{ mb: 2, fontWeight: 600 }}>
          Ask Anything
        </Typography>

        <Box sx={{ display: "flex", gap: 2 }}>
          <TextField
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="What would you like to know?"
            fullWidth
            variant="outlined"
            onKeyPress={(e) => e.key === "Enter" && handleAsk()}
            sx={{
              "& .MuiOutlinedInput-root": {
                borderRadius: 3,
                fontSize: 18,
                py: 1.5
              }
            }}
          />
          <Button
            variant="contained"
            onClick={handleAsk}
            disabled={processing || !question.trim()}
            sx={{
              minWidth: 140,
              borderRadius: 3,
              textTransform: "none",
              fontSize: 16,
              py: 1.5
            }}
          >
            {processing ? <CircularProgress size={24} /> : "Ask"}
          </Button>
        </Box>

        {error && (
          <Typography color="error" sx={{ mt: 2 }}>
            {error}
          </Typography>
        )}
      </Box>

      {/* Processing State */}
      {processing && (
        <Card sx={{ borderRadius: 3 }}>
          <CardContent sx={{ textAlign: "center", py: 6 }}>
            <CircularProgress size={48} sx={{ mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              Searching for answers...
            </Typography>
            <Typography color="text.secondary">
              Analyzing sources and generating response
            </Typography>
          </CardContent>
        </Card>
      )}

      {/* Answer Display */}
      {answer && (
        <Box>
          <Card sx={{ borderRadius: 3, mb: 4 }}>
            <CardContent sx={{ p: 4 }}>
              <Typography variant="h5" sx={{ mb: 3, fontWeight: 600 }}>
                Answer
              </Typography>
              <Typography variant="body1" sx={{ lineHeight: 1.8 }}>
                {answer}
              </Typography>
            </CardContent>
          </Card>

          {/* Sources */}
          {sources.length > 0 && (
            <Box>
              <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                Sources
              </Typography>
              <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap" }}>
                {sources.map((source, index) => (
                  <Chip
                    key={index}
                    label={source}
                    variant="outlined"
                    sx={{ borderRadius: 2 }}
                  />
                ))}
              </Box>
            </Box>
          )}
        </Box>
      )}

      {/* Empty State */}
      {!processing && !answer && !error && (
        <Box sx={{ textAlign: "center", py: 8, color: "text.secondary" }}>
          <Typography variant="h6" gutterBottom>
            Ask a question to get started
          </Typography>
          <Typography variant="body2">
            Get AI-powered answers with sources
          </Typography>
        </Box>
      )}
    </Box>
  );
}
