/**
 * Telemetry Module - Ruthlessly Simplified
 * No input: Auto-updates
 * One view: Key metrics + trend
 */

import React, { useState, useEffect } from "react";
import {
  Box,
  Typography,
  Card,
  CardContent,
} from "@mui/material";
import { GridLegacy as Grid } from "@mui/material";

export function TelemetryModule() {
  const [metrics, setMetrics] = useState({
    requestsPerSec: 127,
    avgLatency: 45,
    errorRate: 0.3,
    activeConnections: 1840
  });

  // Auto-update every 5 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      setMetrics({
        requestsPerSec: Math.floor(100 + Math.random() * 50),
        avgLatency: Math.floor(30 + Math.random() * 30),
        errorRate: Number((Math.random() * 1).toFixed(1)),
        activeConnections: Math.floor(1500 + Math.random() * 500)
      });
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  return (
    <Box sx={{ p: 6, maxWidth: 1200, margin: "0 auto" }}>
      <Typography variant="h4" sx={{ mb: 4, fontWeight: 600 }}>
        System Metrics
      </Typography>

      {/* Key Metrics */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ borderRadius: 3 }}>
            <CardContent sx={{ textAlign: "center" }}>
              <Typography variant="h3" sx={{ fontWeight: 700, color: "primary.main", mb: 1 }}>
                {metrics.requestsPerSec}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Requests/sec
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ borderRadius: 3 }}>
            <CardContent sx={{ textAlign: "center" }}>
              <Typography variant="h3" sx={{ fontWeight: 700, color: "success.main", mb: 1 }}>
                {metrics.avgLatency}ms
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Avg Latency
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ borderRadius: 3 }}>
            <CardContent sx={{ textAlign: "center" }}>
              <Typography variant="h3" sx={{ fontWeight: 700, color: metrics.errorRate > 0.5 ? "error.main" : "success.main", mb: 1 }}>
                {metrics.errorRate}%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Error Rate
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ borderRadius: 3 }}>
            <CardContent sx={{ textAlign: "center" }}>
              <Typography variant="h3" sx={{ fontWeight: 700, color: "info.main", mb: 1 }}>
                {metrics.activeConnections}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Active Connections
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Trend Chart Placeholder */}
      <Card sx={{ borderRadius: 3 }}>
        <CardContent>
          <Typography variant="h6" sx={{ mb: 3, fontWeight: 600 }}>
            Request Volume (Last Hour)
          </Typography>
          <Box 
            sx={{ 
              height: 300, 
              display: "flex", 
              alignItems: "center", 
              justifyContent: "center",
              bgcolor: "grey.50",
              borderRadius: 2
            }}
          >
            <Typography color="text.secondary">
              Chart visualization would go here
            </Typography>
          </Box>
        </CardContent>
      </Card>

      <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: "block", textAlign: "center" }}>
        Auto-updates every 5 seconds
      </Typography>
    </Box>
  );
}
