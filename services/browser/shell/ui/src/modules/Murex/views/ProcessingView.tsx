/**
 * Murex Processing View
 * Displays processing status for Murex trade requests
 */

import React from "react";
import {
  Box,
  Typography,
  Stack,
  Card,
  CardContent,
  LinearProgress,
  Alert,
  CircularProgress,
  Chip,
  TextField,
  Button,
  List,
  ListItem,
  ListItemText
} from "@mui/material";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import ErrorIcon from "@mui/icons-material/Error";
import RefreshIcon from "@mui/icons-material/Refresh";
import { Panel } from "../../../components/Panel";
import type { MurexProcessingRequest } from "../../../api/murex";

interface ProcessingViewProps {
  requestId: string;
  statusData: MurexProcessingRequest | null;
  loading: boolean;
  error: string | null;
  onRequestIdChange: (id: string) => void;
  onRefresh: () => void;
}

export function ProcessingView({
  requestId,
  statusData,
  loading,
  error,
  onRequestIdChange,
  onRefresh
}: ProcessingViewProps) {
  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight={400}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ m: 2 }}>
        {error}
      </Alert>
    );
  }

  if (!statusData) {
    return (
      <Panel title="Processing Status" dense>
        <Stack spacing={2}>
          <TextField
            label="Request ID"
            value={requestId}
            onChange={(e) => onRequestIdChange(e.target.value)}
            placeholder="Enter request ID"
            fullWidth
          />
          <Typography variant="body2" color="text.secondary" sx={{ textAlign: "center", py: 2 }}>
            No processing data available. Enter a request ID to view status.
          </Typography>
        </Stack>
      </Panel>
    );
  }

  const stats = statusData.statistics || {
    documents_processed: 0,
    documents_succeeded: 0,
    documents_failed: 0,
    steps_completed: 0
  };

  return (
    <Stack spacing={2}>
      <Panel title="Processing Status" dense>
        <Stack spacing={2}>
          <Box display="flex" justifyContent="space-between" alignItems="center">
            <Box>
              <Typography variant="h6" gutterBottom>
                Request: {statusData.request_id}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {statusData.query}
              </Typography>
            </Box>
            <Button startIcon={<RefreshIcon />} onClick={onRefresh} size="small">
              Refresh
            </Button>
          </Box>

          <Stack direction="row" spacing={2}>
            <Card variant="outlined" sx={{ flex: 1 }}>
              <CardContent>
                <Typography variant="h4" color="primary">
                  {stats.documents_processed}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Trades Processed
                </Typography>
              </CardContent>
            </Card>

            <Card variant="outlined" sx={{ flex: 1 }}>
              <CardContent>
                <Typography variant="h4" color="success.main">
                  {stats.documents_succeeded}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Succeeded
                </Typography>
              </CardContent>
            </Card>

            <Card variant="outlined" sx={{ flex: 1 }}>
              <CardContent>
                <Typography variant="h4" color={stats.documents_failed > 0 ? "error" : "text.secondary"}>
                  {stats.documents_failed}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Failed
                </Typography>
              </CardContent>
            </Card>
          </Stack>

          {statusData.progress_percent !== undefined && (
            <Box>
              <Box display="flex" justifyContent="space-between" mb={1}>
                <Typography variant="body2" color="text.secondary">
                  Progress
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {statusData.progress_percent.toFixed(1)}%
                </Typography>
              </Box>
              <LinearProgress variant="determinate" value={statusData.progress_percent} />
              {statusData.estimated_time_remaining_ms && (
                <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: "block" }}>
                  Estimated time remaining: {Math.round(statusData.estimated_time_remaining_ms / 1000)}s
                </Typography>
              )}
            </Box>
          )}

          {statusData.current_step && (
            <Box sx={{ p: 2, bgcolor: "grey.100", borderRadius: 1 }}>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Current Step
              </Typography>
              <Typography variant="h6">{statusData.current_step}</Typography>
            </Box>
          )}

          <Chip
            label={statusData.status}
            color={
              statusData.status === "completed"
                ? "success"
                : statusData.status === "failed"
                ? "error"
                : statusData.status === "processing"
                ? "info"
                : "default"
            }
            sx={{ alignSelf: "flex-start" }}
          />
        </Stack>
      </Panel>

      {statusData.errors && statusData.errors.length > 0 && (
        <Panel title="Errors" dense>
          <Alert severity="error" sx={{ mb: 1 }}>
            {statusData.errors.length} error(s) occurred during processing
          </Alert>
          <List dense>
            {statusData.errors.map((error, index) => {
              const errorMessage = typeof error === 'string' ? error : error.message;
              const errorCode = typeof error === 'string' ? undefined : error.code;
              return (
                <ListItem key={index} sx={{ py: 0.5 }}>
                  <ErrorIcon color="error" sx={{ fontSize: 16, mr: 1 }} />
                  <ListItemText
                    primary={errorMessage}
                    secondary={errorCode}
                    primaryTypographyProps={{ variant: "body2" }}
                  />
                </ListItem>
              );
            })}
          </List>
        </Panel>
      )}
    </Stack>
  );
}

