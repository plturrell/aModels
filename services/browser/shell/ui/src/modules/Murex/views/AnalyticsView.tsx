/**
 * Murex Analytics View
 * Displays analytics and history for Murex processing
 */

import React from "react";
import {
  Box,
  Typography,
  Stack,
  Card,
  CardContent,
  Alert,
  CircularProgress,
  Button,
  List,
  ListItem,
  ListItemText,
  Divider,
  Chip
} from "@mui/material";
import RefreshIcon from "@mui/icons-material/Refresh";
import { Panel } from "../../../components/Panel";
import type { MurexRequestHistory } from "../../../api/murex";

interface AnalyticsViewProps {
  history: MurexRequestHistory | null;
  loading: boolean;
  error: string | null;
  onRefresh: () => void;
}

export function AnalyticsView({
  history,
  loading,
  error,
  onRefresh
}: AnalyticsViewProps) {
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

  const completed = history?.requests.filter(r => r.status === "completed").length || 0;
  const failed = history?.requests.filter(r => r.status === "failed").length || 0;
  const total = history?.total || 0;
  const successRate = total > 0 ? ((completed / total) * 100).toFixed(1) : "0";

  return (
    <Stack spacing={2}>
      <Panel title="Analytics Summary" dense>
        <Stack spacing={2}>
          <Box display="flex" justifyContent="flex-end">
            <Button startIcon={<RefreshIcon />} onClick={onRefresh} size="small">
              Refresh
            </Button>
          </Box>

          <Stack direction="row" spacing={2}>
            <Card variant="outlined" sx={{ flex: 1 }}>
              <CardContent>
                <Typography variant="h4" color="primary">
                  {total}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Total Requests
                </Typography>
              </CardContent>
            </Card>

            <Card variant="outlined" sx={{ flex: 1 }}>
              <CardContent>
                <Typography variant="h4" color="success.main">
                  {completed}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Completed
                </Typography>
              </CardContent>
            </Card>

            <Card variant="outlined" sx={{ flex: 1 }}>
              <CardContent>
                <Typography variant="h4" color="error">
                  {failed}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Failed
                </Typography>
              </CardContent>
            </Card>

            <Card variant="outlined" sx={{ flex: 1 }}>
              <CardContent>
                <Typography variant="h4" color="warning.main">
                  {successRate}%
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Success Rate
                </Typography>
              </CardContent>
            </Card>
          </Stack>
        </Stack>
      </Panel>

      <Panel title="Recent Activity" dense>
        {history && history.requests.length > 0 ? (
          <List>
            {history.requests.slice(0, 10).map((request, index) => (
              <React.Fragment key={request.request_id}>
                <ListItem>
                  <Box sx={{ flex: 1 }}>
                    <Box display="flex" alignItems="center" gap={1} mb={0.5}>
                      <Typography variant="body2" fontWeight={500}>
                        {request.request_id}
                      </Typography>
                      <Chip
                        label={request.status}
                        size="small"
                        color={
                          request.status === "completed"
                            ? "success"
                            : request.status === "failed"
                            ? "error"
                            : "default"
                        }
                        sx={{ ml: "auto" }}
                      />
                    </Box>
                    <Typography variant="caption" color="text.secondary">
                      {request.query} • {new Date(request.created_at).toLocaleString()}
                    </Typography>
                  </Box>
                </ListItem>
                {index < Math.min(9, history.requests.length - 1) && <Divider />}
              </React.Fragment>
            ))}
          </List>
        ) : (
          <Typography variant="body2" color="text.secondary" sx={{ textAlign: "center", py: 4 }}>
            No processing history available.
          </Typography>
        )}
      </Panel>
    </Stack>
  );
}

