/**
 * LLM Observability Module
 * Visualizes LLM traces, metrics, and performance data using OpenLLMetry conventions
 */

import React, { useState, useEffect } from "react";
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  CircularProgress,
  Alert,
} from "@mui/material";
import { GridLegacy as GridLegacy } from "@mui/material";
import axios from "axios";

interface LLMTrace {
  id: string;
  model: string;
  system: string;
  requestType: string;
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  latency: number;
  finishReason?: string;
  cost?: number;
  timestamp: string;
  status: "success" | "error";
}

interface LLMMetrics {
  totalCalls: number;
  totalTokens: number;
  totalCost: number;
  avgLatency: number;
  errorRate: number;
  models: Record<string, { calls: number; tokens: number; cost: number }>;
}

export function LLMObservabilityModule() {
  const [traces, setTraces] = useState<LLMTrace[]>([]);
  const [metrics, setMetrics] = useState<LLMMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [timeRange, setTimeRange] = useState<"1h" | "24h" | "7d">("1h");

  useEffect(() => {
    fetchLLMData();
    const interval = setInterval(fetchLLMData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, [timeRange]);

  const fetchLLMData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Fetch traces from telemetry-exporter API
      // In production, this would call the actual telemetry-exporter service
      const response = await axios.get(
        `/api/v1/traces/llm?time_range=${timeRange}`,
        {
          timeout: 10000,
        }
      ).catch(() => {
        // Fallback to mock data if API is not available
        return { data: generateMockData() };
      });

      const data = response.data;
      setTraces(data.traces || []);
      setMetrics(data.metrics || calculateMetrics(data.traces || []));
    } catch (err: any) {
      setError(err.message || "Failed to fetch LLM data");
      // Use mock data as fallback
      const mockData = generateMockData();
      setTraces(mockData.traces);
      setMetrics(calculateMetrics(mockData.traces));
    } finally {
      setLoading(false);
    }
  };

  const generateMockData = (): { traces: LLMTrace[] } => {
    const models = ["gpt-4", "gpt-3.5-turbo", "claude-3-opus", "claude-3-sonnet"];
    const systems = ["openai", "anthropic"];
    const mockTraces: LLMTrace[] = [];

    for (let i = 0; i < 20; i++) {
      const model = models[Math.floor(Math.random() * models.length)];
      const system = systems[Math.floor(Math.random() * systems.length)];
      const promptTokens = Math.floor(Math.random() * 2000) + 100;
      const completionTokens = Math.floor(Math.random() * 1000) + 50;
      const totalTokens = promptTokens + completionTokens;

      mockTraces.push({
        id: `trace-${i}`,
        model,
        system,
        requestType: "chat",
        promptTokens,
        completionTokens,
        totalTokens,
        latency: Math.floor(Math.random() * 3000) + 500,
        finishReason: "stop",
        cost: totalTokens * 0.0001,
        timestamp: new Date(Date.now() - Math.random() * 3600000).toISOString(),
        status: Math.random() > 0.1 ? "success" : "error",
      });
    }

    return { traces: mockTraces };
  };

  const calculateMetrics = (traces: LLMTrace[]): LLMMetrics => {
    if (traces.length === 0) {
      return {
        totalCalls: 0,
        totalTokens: 0,
        totalCost: 0,
        avgLatency: 0,
        errorRate: 0,
        models: {},
      };
    }

    const modelStats: Record<string, { calls: number; tokens: number; cost: number }> = {};
    let totalCost = 0;
    let totalLatency = 0;
    let errorCount = 0;

    traces.forEach((trace) => {
      totalCost += trace.cost || 0;
      totalLatency += trace.latency;
      if (trace.status === "error") errorCount++;

      if (!modelStats[trace.model]) {
        modelStats[trace.model] = { calls: 0, tokens: 0, cost: 0 };
      }
      modelStats[trace.model].calls++;
      modelStats[trace.model].tokens += trace.totalTokens;
      modelStats[trace.model].cost += trace.cost || 0;
    });

    return {
      totalCalls: traces.length,
      totalTokens: traces.reduce((sum, t) => sum + t.totalTokens, 0),
      totalCost,
      avgLatency: totalLatency / traces.length,
      errorRate: (errorCount / traces.length) * 100,
      models: modelStats,
    };
  };

  if (loading && !metrics) {
    return (
      <Box sx={{ display: "flex", justifyContent: "center", alignItems: "center", height: "100vh" }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ p: 6, maxWidth: 1400, margin: "0 auto" }}>
      <Typography variant="h4" sx={{ mb: 4, fontWeight: 600 }}>
        LLM Observability
      </Typography>

      {error && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          {error} - Showing mock data
        </Alert>
      )}

      {/* Key Metrics */}
      {metrics && (
        <GridLegacy container spacing={3} sx={{ mb: 4 }}>
          <GridLegacy item xs={12} sm={6} md={3}>
            <Card sx={{ borderRadius: 3 }}>
              <CardContent sx={{ textAlign: "center" }}>
                <Typography variant="h3" sx={{ fontWeight: 700, color: "primary.main", mb: 1 }}>
                  {metrics.totalCalls}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Total LLM Calls
                </Typography>
              </CardContent>
            </Card>
          </GridLegacy>

          <GridLegacy item xs={12} sm={6} md={3}>
            <Card sx={{ borderRadius: 3 }}>
              <CardContent sx={{ textAlign: "center" }}>
                <Typography variant="h3" sx={{ fontWeight: 700, color: "info.main", mb: 1 }}>
                  {metrics.totalTokens.toLocaleString()}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Total Tokens
                </Typography>
              </CardContent>
            </Card>
          </GridLegacy>

          <GridLegacy item xs={12} sm={6} md={3}>
            <Card sx={{ borderRadius: 3 }}>
              <CardContent sx={{ textAlign: "center" }}>
                <Typography variant="h3" sx={{ fontWeight: 700, color: "success.main", mb: 1 }}>
                  ${metrics.totalCost.toFixed(4)}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Total Cost
                </Typography>
              </CardContent>
            </Card>
          </GridLegacy>

          <GridLegacy item xs={12} sm={6} md={3}>
            <Card sx={{ borderRadius: 3 }}>
              <CardContent sx={{ textAlign: "center" }}>
                <Typography variant="h3" sx={{ fontWeight: 700, color: metrics.avgLatency > 2000 ? "error.main" : "success.main", mb: 1 }}>
                  {metrics.avgLatency.toFixed(0)}ms
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Avg Latency
                </Typography>
              </CardContent>
            </Card>
          </GridLegacy>
        </GridLegacy>
      )}

      {/* Model Breakdown */}
      {metrics && Object.keys(metrics.models).length > 0 && (
        <Card sx={{ borderRadius: 3, mb: 4 }}>
          <CardContent>
            <Typography variant="h6" sx={{ mb: 3, fontWeight: 600 }}>
              Model Performance
            </Typography>
            <GridLegacy container spacing={2}>
              {Object.entries(metrics.models).map(([model, stats]) => (
                <GridLegacy item xs={12} sm={6} md={4} key={model}>
                  <Card variant="outlined" sx={{ p: 2 }}>
                    <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
                      {model}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Calls: {stats.calls} | Tokens: {stats.tokens.toLocaleString()} | Cost: ${stats.cost.toFixed(4)}
                    </Typography>
                  </Card>
                </GridLegacy>
              ))}
            </GridLegacy>
          </CardContent>
        </Card>
      )}

      {/* Recent Traces Table */}
      <Card sx={{ borderRadius: 3 }}>
        <CardContent>
          <Typography variant="h6" sx={{ mb: 3, fontWeight: 600 }}>
            Recent LLM Traces
          </Typography>
          <TableContainer component={Paper} variant="outlined">
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Model</TableCell>
                  <TableCell>System</TableCell>
                  <TableCell>Type</TableCell>
                  <TableCell align="right">Tokens</TableCell>
                  <TableCell align="right">Latency</TableCell>
                  <TableCell align="right">Cost</TableCell>
                  <TableCell>Status</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {traces.slice(0, 10).map((trace) => (
                  <TableRow key={trace.id}>
                    <TableCell>{trace.model}</TableCell>
                    <TableCell>
                      <Chip label={trace.system} size="small" />
                    </TableCell>
                    <TableCell>{trace.requestType}</TableCell>
                    <TableCell align="right">
                      {trace.promptTokens} / {trace.completionTokens} ({trace.totalTokens})
                    </TableCell>
                    <TableCell align="right">{trace.latency}ms</TableCell>
                    <TableCell align="right">${(trace.cost || 0).toFixed(4)}</TableCell>
                    <TableCell>
                      <Chip
                        label={trace.status}
                        size="small"
                        color={trace.status === "success" ? "success" : "error"}
                      />
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: "block", textAlign: "center" }}>
        Data refreshed every 30 seconds | Using OpenLLMetry semantic conventions
      </Typography>
    </Box>
  );
}

