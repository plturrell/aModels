/**
 * DMS Analytics View
 * Displays analytics and history for DMS requests
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
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper as MuiPaper
} from "@mui/material";
import { Panel } from "../../../components/common/Panel";
import type { DMSRequestHistory } from "../../../api/dms";

interface AnalyticsViewProps {
  history: DMSRequestHistory | null;
  loading: boolean;
  error: string | null;
}

export function AnalyticsView({ history, loading, error }: AnalyticsViewProps) {
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

  if (!history) {
    return (
      <Panel title="Analytics" dense>
        <Typography variant="body2" color="text.secondary" sx={{ textAlign: "center", py: 4 }}>
          No analytics data available
        </Typography>
      </Panel>
    );
  }

  const completed = history.requests.filter(r => r.status === "completed").length;
  const failed = history.requests.filter(r => r.status === "failed").length;
  const successRate = history.total > 0 ? ((completed / history.total) * 100).toFixed(1) : "0";

  return (
    <Stack spacing={2}>
      <Panel title="Analytics Summary" dense>
        <Stack direction="row" spacing={2}>
          <Card variant="outlined" sx={{ flex: 1 }}>
            <CardContent>
              <Typography variant="h4" color="primary">
                {history.total}
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
      </Panel>

      <Panel title="Recent Requests" dense>
        {history.requests && history.requests.length > 0 ? (
          <TableContainer component={MuiPaper} variant="outlined">
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Request ID</TableCell>
                  <TableCell>Query</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Documents</TableCell>
                  <TableCell>Created</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {history.requests.slice(0, 20).map((req) => (
                  <TableRow key={req.request_id}>
                    <TableCell>
                      <Typography variant="body2" fontFamily="monospace">
                        {req.request_id.substring(0, 8)}...
                      </Typography>
                    </TableCell>
                    <TableCell>{req.query}</TableCell>
                    <TableCell>
                      <Chip
                        label={req.status}
                        size="small"
                        color={
                          req.status === "completed"
                            ? "success"
                            : req.status === "failed"
                            ? "error"
                            : "default"
                        }
                      />
                    </TableCell>
                    <TableCell>{req.document_count}</TableCell>
                    <TableCell>
                      {new Date(req.created_at).toLocaleString()}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        ) : (
          <Typography variant="body2" color="text.secondary" sx={{ textAlign: "center", py: 4 }}>
            No requests found
          </Typography>
        )}
      </Panel>
    </Stack>
  );
}

