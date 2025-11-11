import { useMemo } from "react";
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
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import { Panel } from "../../../components/Panel";
import type { ProcessingRequest } from "../../../api/perplexity";

interface ProcessingViewProps {
  requestId: string;
  data: ProcessingRequest | null;
  loading: boolean;
  error: Error | null;
}

export function ProcessingView({ requestId, data, loading, error }: ProcessingViewProps) {
  const progress = useMemo(() => {
    if (!data) return 0;
    return data.progress_percent || 0;
  }, [data]);

  if (!requestId) {
    return (
      <Alert severity="info">
        Enter a request ID above to view processing status, or submit a new query to get started.
      </Alert>
    );
  }

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 200 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error">
        Failed to load processing status: {error.message}
      </Alert>
    );
  }

  if (!data) {
    return (
      <Alert severity="warning">
        No data found for request ID: {requestId}
      </Alert>
    );
  }

  return (
    <Box>
      {/* Status Summary */}
      <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} sx={{ mb: 3 }}>
        <Card sx={{ flex: 1 }}>
          <CardContent>
            <Typography variant="h4" color="primary" gutterBottom>
              {data.statistics?.documents_processed || 0}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Documents Processed
            </Typography>
          </CardContent>
        </Card>
        <Card sx={{ flex: 1 }}>
          <CardContent>
            <Typography variant="h4" color="success.main" gutterBottom>
              {data.statistics?.documents_succeeded || 0}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Succeeded
            </Typography>
          </CardContent>
        </Card>
        <Card sx={{ flex: 1 }}>
          <CardContent>
            <Typography 
              variant="h4" 
              color={data.statistics?.documents_failed ? "error.main" : "text.secondary"} 
              gutterBottom
            >
              {data.statistics?.documents_failed || 0}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Failed
            </Typography>
          </CardContent>
        </Card>
      </Stack>

      {/* Progress */}
      <Panel title="Progress" dense>
        <Box sx={{ mb: 2 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
            <Typography variant="body2" color="text.secondary">
              {data.current_step || "Initializing..."}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {progress.toFixed(1)}%
            </Typography>
          </Box>
          <LinearProgress 
            variant="determinate" 
            value={progress} 
            sx={{ height: 8, borderRadius: 4 }}
          />
        </Box>

        {data.completed_steps && data.completed_steps.length > 0 && (
          <Box>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Completed Steps ({data.completed_steps.length} / {data.total_steps || 0})
            </Typography>
            <List dense>
              {data.completed_steps.map((step, index) => (
                <ListItem key={index} sx={{ py: 0.5 }}>
                  <CheckCircleIcon color="success" sx={{ fontSize: 16, mr: 1 }} />
                  <ListItemText 
                    primary={step}
                    primaryTypographyProps={{ variant: "body2" }}
                  />
                </ListItem>
              ))}
            </List>
          </Box>
        )}
      </Panel>

      {/* Errors */}
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

      {/* Status Info */}
      <Panel title="Request Information" dense>
        <Stack spacing={2}>
          <Box>
            <Typography variant="body2" color="text.secondary">
              Request ID
            </Typography>
            <Typography variant="body1" sx={{ fontFamily: 'monospace' }}>
              {data.request_id}
            </Typography>
          </Box>
          <Box>
            <Typography variant="body2" color="text.secondary">
              Status
            </Typography>
            <Chip 
              label={data.status} 
              color={
                data.status === "completed" ? "success" :
                data.status === "failed" ? "error" :
                data.status === "processing" ? "info" :
                "default"
              }
            />
          </Box>
          <Box>
            <Typography variant="body2" color="text.secondary">
              Query
            </Typography>
            <Typography variant="body1">
              {data.query}
            </Typography>
          </Box>
          {data.processing_time_ms && (
            <Box>
              <Typography variant="body2" color="text.secondary">
                Processing Time
              </Typography>
              <Typography variant="body1">
                {(data.processing_time_ms / 1000).toFixed(2)}s
              </Typography>
            </Box>
          )}
        </Stack>
      </Panel>
    </Box>
  );
}

