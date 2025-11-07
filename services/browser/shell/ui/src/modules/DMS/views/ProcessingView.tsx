/**
 * DMS Processing View
 * Displays processing status for DMS document requests
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
  List,
  ListItem,
  ListItemText,
  Divider
} from "@mui/material";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import ErrorIcon from "@mui/icons-material/Error";
import { Panel } from "../../../components/Panel";
import type { DMSProcessingRequest } from "../../../api/dms";

interface ProcessingViewProps {
  data: DMSProcessingRequest | null;
  loading: boolean;
  error: string | null;
}

export function ProcessingView({ data, loading, error }: ProcessingViewProps) {
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

  if (!data) {
    return (
      <Panel title="Processing Status" dense>
        <Typography variant="body2" color="text.secondary" sx={{ textAlign: "center", py: 4 }}>
          No processing data available. Enter a request ID to view status.
        </Typography>
      </Panel>
    );
  }

  const stats = data.statistics || {
    documents_processed: 0,
    documents_succeeded: 0,
    documents_failed: 0,
    steps_completed: 0
  };

  return (
    <Stack spacing={2}>
      <Panel title="Processing Status" dense>
        <Stack spacing={2}>
          <Box>
            <Typography variant="h6" gutterBottom>
              Request: {data.request_id}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {data.query}
            </Typography>
          </Box>

          <Stack direction="row" spacing={2}>
            <Card variant="outlined" sx={{ flex: 1 }}>
              <CardContent>
                <Typography variant="h4" color="primary">
                  {stats.documents_processed}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Documents Processed
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

          {data.progress_percent !== undefined && (
            <Box>
              <Box display="flex" justifyContent="space-between" mb={1}>
                <Typography variant="body2" color="text.secondary">
                  Progress
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {data.progress_percent.toFixed(1)}%
                </Typography>
              </Box>
              <LinearProgress variant="determinate" value={data.progress_percent} />
              {data.estimated_time_remaining_ms && (
                <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: "block" }}>
                  Estimated time remaining: {Math.round(data.estimated_time_remaining_ms / 1000)}s
                </Typography>
              )}
            </Box>
          )}

          {data.current_step && (
            <Box sx={{ p: 2, bgcolor: "grey.100", borderRadius: 1 }}>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Current Step
              </Typography>
              <Typography variant="h6">{data.current_step}</Typography>
            </Box>
          )}

          <Chip
            label={data.status}
            color={
              data.status === "completed"
                ? "success"
                : data.status === "failed"
                ? "error"
                : data.status === "processing"
                ? "info"
                : "default"
            }
            sx={{ alignSelf: "flex-start" }}
          />
        </Stack>
      </Panel>

      {data.errors && data.errors.length > 0 && (
        <Panel title="Errors" dense>
          <Alert severity="error" sx={{ mb: 1 }}>
            {data.errors.length} error(s) occurred during processing
          </Alert>
          <List dense>
            {data.errors.map((error, index) => {
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

      {data.documents && data.documents.length > 0 && (
        <Panel title="Documents" dense>
          <List dense>
            {data.documents.map((doc, index) => (
              <React.Fragment key={doc.id}>
                <ListItem>
                  <Box sx={{ flex: 1 }}>
                    <Box display="flex" alignItems="center" gap={1} mb={0.5}>
                      {doc.status === "succeeded" ? (
                        <CheckCircleIcon color="success" sx={{ fontSize: 16 }} />
                      ) : doc.status === "failed" ? (
                        <ErrorIcon color="error" sx={{ fontSize: 16 }} />
                      ) : null}
                      <Typography variant="body2" fontWeight={500}>
                        {doc.title || doc.id}
                      </Typography>
                      <Chip label={doc.status} size="small" sx={{ ml: "auto" }} />
                    </Box>
                    {doc.error && (
                      <Typography variant="caption" color="error">
                        {doc.error}
                      </Typography>
                    )}
                  </Box>
                </ListItem>
                {index < data.documents!.length - 1 && <Divider />}
              </React.Fragment>
            ))}
          </List>
        </Panel>
      )}
    </Stack>
  );
}

