import { useMemo } from "react";
import {
  Box,
  Typography,
  Button,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Stack,
  Card,
  CardContent,
  Divider,
  List,
  ListItem
} from '@mui/material';
import ClearIcon from '@mui/icons-material/Clear';

import { Panel } from "../../components/Panel";
import { telemetryDefaults, telemetryRecordFields, telemetryConfigFields } from "../../data/telemetry";
import {
  useTelemetryStore,
  type TelemetryState,
  type InteractionMetric
} from "../../state/useTelemetryStore";

const formatNumber = (value: number | undefined) =>
  typeof value === "number" && Number.isFinite(value) ? value.toLocaleString() : "–";

const formatLatency = (value: number) => `${value.toLocaleString()} ms`;

const formatRelativeTime = (timestamp: number) => {
  const diff = Date.now() - timestamp;
  if (diff < 45_000) return "just now";
  if (diff < 90_000) return "1 min ago";
  const minutes = Math.round(diff / 60_000);
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} h ago`;
  const days = Math.round(hours / 24);
  return `${days} d ago`;
};

export function TelemetryModule() {
  const metrics = useTelemetryStore((state: TelemetryState) => state.metrics);
  const resetMetrics = useTelemetryStore((state: TelemetryState) => state.reset);

  const summary = useMemo(() => {
    if (!metrics.length) {
      return null;
    }

    const totalLatency = metrics.reduce<number>((sum, metric) => sum + metric.durationMs, 0);
    const maxLatency = metrics.reduce<number>(
      (max, metric) => Math.max(max, metric.durationMs),
      0
    );
    const totalPromptTokens = metrics.reduce<number>(
      (sum, metric) => sum + (metric.promptTokens ?? 0),
      0
    );
    const totalCompletionTokens = metrics.reduce<number>(
      (sum, metric) => sum + (metric.completionTokens ?? 0),
      0
    );
    const totalCitations = metrics.reduce<number>(
      (sum, metric) => sum + metric.citations,
      0
    );

    return {
      interactions: metrics.length,
      avgLatency: Math.round(totalLatency / metrics.length),
      maxLatency,
      avgPromptTokens: Math.round(totalPromptTokens / metrics.length),
      avgCompletionTokens: Math.round(totalCompletionTokens / metrics.length),
      avgCitations:
        metrics.length > 0 ? Number((totalCitations / metrics.length).toFixed(1)) : 0
    };
  }, [metrics]);

  const hasMetrics = metrics.length > 0;

  return (
    <Box>
      <Panel title="Service Defaults" subtitle="Telemetry baseline in extract service">
        <Stack spacing={2}>
          <Box>
            <Typography variant="caption" color="text.secondary">Library</Typography>
            <Typography variant="body2">
              {telemetryDefaults.library ?? "layer4_extract"}
            </Typography>
          </Box>
          <Divider />
          <Box>
            <Typography variant="caption" color="text.secondary">Operation</Typography>
            <Typography variant="body2">
              {telemetryDefaults.operation ?? "run_extract"}
            </Typography>
          </Box>
          <Divider />
          <Box>
            <Typography variant="caption" color="text.secondary">HTTP timeout</Typography>
            <Typography variant="body2">
              {telemetryDefaults.httpTimeout ?? "45 * time.Second"}
            </Typography>
          </Box>
          <Divider />
          <Box>
            <Typography variant="caption" color="text.secondary">Dial timeout</Typography>
            <Typography variant="body2">
              {telemetryDefaults.dialTimeout ?? "5 * time.Second"}
            </Typography>
          </Box>
          <Divider />
          <Box>
            <Typography variant="caption" color="text.secondary">Call timeout</Typography>
            <Typography variant="body2">
              {telemetryDefaults.callTimeout ?? "3 * time.Second"}
            </Typography>
          </Box>
        </Stack>
      </Panel>

      <Panel
        title="Session Pulse"
        subtitle={hasMetrics && summary ? "Live feel of the LocalAI assistant" : "No interactions recorded yet"}
        actions={
          hasMetrics ? (
            <Button
              variant="outlined"
              size="small"
              startIcon={<ClearIcon />}
              onClick={resetMetrics}
            >
              Clear history
            </Button>
          ) : null
        }
      >
        {hasMetrics && summary ? (
          <Stack direction="row" spacing={2} flexWrap="wrap">
            <Card variant="outlined" sx={{ flex: 1, minWidth: 150 }}>
              <CardContent>
                <Typography variant="caption" color="text.secondary">
                  Average latency
                </Typography>
                <Typography variant="h6">{formatLatency(summary.avgLatency)}</Typography>
                <Typography variant="caption" color="text.secondary">
                  Peak {formatLatency(summary.maxLatency)}
                </Typography>
              </CardContent>
            </Card>
            <Card variant="outlined" sx={{ flex: 1, minWidth: 150 }}>
              <CardContent>
                <Typography variant="caption" color="text.secondary">
                  Sessions today
                </Typography>
                <Typography variant="h6">{summary.interactions}</Typography>
                <Typography variant="caption" color="text.secondary">
                  {summary.avgCitations.toFixed(1)} citations per reply
                </Typography>
              </CardContent>
            </Card>
            <Card variant="outlined" sx={{ flex: 1, minWidth: 150 }}>
              <CardContent>
                <Typography variant="caption" color="text.secondary">
                  Token blend
                </Typography>
                <Typography variant="h6">
                  {summary.avgPromptTokens.toLocaleString()} /{" "}
                  {summary.avgCompletionTokens.toLocaleString()}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  prompt / completion
                </Typography>
              </CardContent>
            </Card>
          </Stack>
        ) : (
          <Paper variant="outlined" sx={{ p: 3, textAlign: 'center' }}>
            <Typography variant="body2" color="text.secondary">
              Chat with LocalAI to light up latency, token, and citation telemetry in real time.
            </Typography>
          </Paper>
        )}
      </Panel>

      <Panel title="Latest Sessions" subtitle="Rolling 25 interactions">
        {hasMetrics ? (
          <TableContainer component={Paper} variant="outlined">
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>When</TableCell>
                  <TableCell>Model</TableCell>
                  <TableCell>Latency</TableCell>
                  <TableCell>Prompt tokens</TableCell>
                  <TableCell>Completion tokens</TableCell>
                  <TableCell>Citations</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {metrics.map((metric: InteractionMetric) => (
                  <TableRow key={metric.id}>
                    <TableCell>
                      <Typography variant="body2">
                        {formatRelativeTime(metric.timestamp)}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip label={metric.model} size="small" />
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {formatLatency(metric.durationMs)}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {formatNumber(metric.promptTokens)}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {formatNumber(metric.completionTokens)}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">{metric.citations}</Typography>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        ) : (
          <Paper variant="outlined" sx={{ p: 3, textAlign: 'center' }}>
            <Typography variant="body2" color="text.secondary">
              Once you start chatting we will surface latency, token usage, and citation coverage for
              each exchange here.
            </Typography>
          </Paper>
        )}
      </Panel>

      <Panel title="Schema Reference" subtitle="Telemetry structs at a glance" dense>
        <Stack direction={{ xs: 'column', md: 'row' }} spacing={2}>
          <Box sx={{ flex: 1 }}>
            <Typography variant="subtitle2" gutterBottom>
              telemetryConfig
            </Typography>
            <List dense>
              {telemetryConfigFields.map((field) => (
                <ListItem key={field} sx={{ py: 0.25, px: 0 }}>
                  <Typography variant="body2" component="code" sx={{ fontFamily: 'monospace' }}>
                    {field}
                  </Typography>
                </ListItem>
              ))}
            </List>
          </Box>
          <Divider orientation="vertical" flexItem />
          <Box sx={{ flex: 1 }}>
            <Typography variant="subtitle2" gutterBottom>
              telemetryRecord
            </Typography>
            <List dense>
              {telemetryRecordFields.map((field) => (
                <ListItem key={field} sx={{ py: 0.25, px: 0 }}>
                  <Typography variant="body2" component="code" sx={{ fontFamily: 'monospace' }}>
                    {field}
                  </Typography>
                </ListItem>
              ))}
            </List>
          </Box>
        </Stack>
      </Panel>
    </Box>
  );
}
