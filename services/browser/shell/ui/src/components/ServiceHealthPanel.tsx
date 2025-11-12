import React, { useEffect, useState } from "react";
import {
  Box,
  Typography,
  Paper,
  Chip,
  Stack,
  CircularProgress,
  IconButton,
  Tooltip,
  Alert,
  Collapse,
} from "@mui/material";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import ErrorIcon from "@mui/icons-material/Error";
import WarningIcon from "@mui/icons-material/Warning";
import RefreshIcon from "@mui/icons-material/Refresh";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ExpandLessIcon from "@mui/icons-material/ExpandLess";
import {
  getServiceHealth,
  isServiceHealthy,
  getServiceStatus,
  getServiceError,
  type HealthCheckResponse,
} from "../api/health";

interface ServiceHealthPanelProps {
  autoRefresh?: boolean;
  refreshInterval?: number; // in milliseconds
  showDetails?: boolean;
}

const SERVICE_DISPLAY_NAMES: Record<string, string> = {
  gateway: "Gateway",
  search_inference: "Search Inference",
  extract: "Extract / Knowledge Graph",
  catalog: "Catalog",
  localai: "LocalAI",
  agentflow: "AgentFlow",
  deep_research: "Deep Research",
  opensearch: "OpenSearch",
  redis: "Redis",
  hana: "HANA",
  data: "Data Service",
  deepagents: "DeepAgents",
  sap_bdc: "SAP BDC",
  layer4_browser: "Layer4 Browser",
  training: "Training Service",
  training_service: "Training Service",
  postgres: "PostgreSQL",
  postgresql: "PostgreSQL",
};

const getServiceIcon = (status: string) => {
  if (status === "ok") {
    return <CheckCircleIcon color="success" fontSize="small" />;
  }
  if (status.includes("error") || status.includes("failed")) {
    return <ErrorIcon color="error" fontSize="small" />;
  }
  return <WarningIcon color="warning" fontSize="small" />;
};

const getServiceColor = (status: string): "success" | "error" | "warning" | "default" => {
  if (status === "ok") {
    return "success";
  }
  if (status.includes("error") || status.includes("failed")) {
    return "error";
  }
  if (status.includes("status:")) {
    return "warning";
  }
  return "default";
};

export function ServiceHealthPanel({
  autoRefresh = true,
  refreshInterval = 30000, // 30 seconds
  showDetails = true,
}: ServiceHealthPanelProps) {
  const [health, setHealth] = useState<HealthCheckResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [expandedServices, setExpandedServices] = useState<Set<string>>(new Set());
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);

  const fetchHealth = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getServiceHealth();
      setHealth(data);
      setLastRefresh(new Date());
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
      setHealth(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHealth();

    if (autoRefresh) {
      const interval = setInterval(fetchHealth, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [autoRefresh, refreshInterval]);

  const handleRefresh = () => {
    fetchHealth();
  };

  const toggleServiceDetails = (serviceName: string) => {
    setExpandedServices((prev) => {
      const next = new Set(prev);
      if (next.has(serviceName)) {
        next.delete(serviceName);
      } else {
        next.add(serviceName);
      }
      return next;
    });
  };

  if (loading && !health) {
    return (
      <Paper sx={{ p: 3, textAlign: "center" }}>
        <CircularProgress size={24} sx={{ mb: 2 }} />
        <Typography variant="body2" color="text.secondary">
          Checking service health...
        </Typography>
      </Paper>
    );
  }

  const services = health ? Object.keys(health) : [];
  const healthyCount = services.filter((s) => isServiceHealthy(health!, s)).length;
  const totalCount = services.length;

  return (
    <Paper sx={{ p: 2 }}>
      <Stack direction="row" spacing={2} alignItems="center" sx={{ mb: 2 }}>
        <Typography variant="h6" sx={{ flex: 1 }}>
          Service Health
        </Typography>
        <Chip
          label={`${healthyCount}/${totalCount} healthy`}
          color={healthyCount === totalCount ? "success" : "warning"}
          size="small"
        />
        <Tooltip title="Refresh health status">
          <IconButton size="small" onClick={handleRefresh} disabled={loading}>
            <RefreshIcon />
          </IconButton>
        </Tooltip>
      </Stack>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          Failed to fetch health status: {error.message}
        </Alert>
      )}

      {lastRefresh && (
        <Typography variant="caption" color="text.secondary" sx={{ mb: 2, display: "block" }}>
          Last updated: {lastRefresh.toLocaleTimeString()}
        </Typography>
      )}

      <Stack spacing={1}>
        {services.map((serviceName) => {
          const displayName = SERVICE_DISPLAY_NAMES[serviceName] || serviceName;
          const status = getServiceStatus(health!, serviceName);
          const isHealthy = isServiceHealthy(health!, serviceName);
          const errorMsg = getServiceError(health!, serviceName);
          const isExpanded = expandedServices.has(serviceName);

          return (
            <Box key={serviceName}>
              <Stack
                direction="row"
                spacing={1}
                alignItems="center"
                sx={{
                  p: 1,
                  borderRadius: 1,
                  bgcolor: isHealthy ? "action.hover" : "error.light",
                  cursor: showDetails ? "pointer" : "default",
                }}
                onClick={() => showDetails && toggleServiceDetails(serviceName)}
              >
                {getServiceIcon(status)}
                <Typography variant="body2" sx={{ flex: 1 }}>
                  {displayName}
                </Typography>
                <Chip
                  label={status}
                  size="small"
                  color={getServiceColor(status)}
                  variant="outlined"
                />
                {showDetails && (
                  <IconButton size="small">
                    {isExpanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                  </IconButton>
                )}
              </Stack>

              {showDetails && (
                <Collapse in={isExpanded}>
                  <Box sx={{ pl: 4, pr: 2, pb: 1, pt: 0.5 }}>
                    {errorMsg && (
                      <Alert severity="error" sx={{ mt: 1 }}>
                        {errorMsg}
                      </Alert>
                    )}
                    {!errorMsg && !isHealthy && (
                      <Typography variant="caption" color="text.secondary">
                        Service may not be running or accessible
                      </Typography>
                    )}
                  </Box>
                </Collapse>
              )}
            </Box>
          );
        })}
      </Stack>

      {services.length === 0 && !loading && (
        <Typography variant="body2" color="text.secondary" sx={{ textAlign: "center", py: 2 }}>
          No services found
        </Typography>
      )}
    </Paper>
  );
}

