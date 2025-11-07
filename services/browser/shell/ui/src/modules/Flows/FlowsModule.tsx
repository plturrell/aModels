import { useMemo, useState } from "react";
import {
  Box,
  Typography,
  Button,
  Alert,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  CircularProgress,
  Stack,
  Card,
  CardContent,
  TextField,
  Divider
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';

import { Panel } from "../../components/Panel";
import { useFlows, runFlow, type FlowInfo, type FlowRunResponse } from "../../api/agentflow";

const formatter = new Intl.DateTimeFormat(undefined, {
  year: "numeric",
  month: "short",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit"
});

const relativeFormatter = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });

const relativeTime = (iso?: string | null) => {
  if (!iso) return "—";
  const ts = Date.parse(iso);
  if (Number.isNaN(ts)) return "—";
  const diffMs = ts - Date.now();
  const minutes = Math.round(diffMs / (1000 * 60));
  if (Math.abs(minutes) < 60) return relativeFormatter.format(minutes, "minute");
  const hours = Math.round(minutes / 60);
  if (Math.abs(hours) < 24) return relativeFormatter.format(hours, "hour");
  const days = Math.round(hours / 24);
  return relativeFormatter.format(days, "day");
};

const truncate = (value?: string | null, length = 140) => {
  if (!value) return null;
  return value.length > length ? `${value.slice(0, length - 1)}…` : value;
};

export function FlowsModule() {
  const { data, loading, error, refresh } = useFlows();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [inputValue, setInputValue] = useState<string>("Hello AgentFlow");
  const [sending, setSending] = useState<boolean>(false);
  const [runOutput, setRunOutput] = useState<FlowRunResponse | null>(null);
  const [runError, setRunError] = useState<Error | null>(null);

  const flows = useMemo(() => data ?? [], [data]);
  const totalFlows = flows.length;
  const syncedFlows = flows.filter((flow) => Boolean(flow.remote_id)).length;
  const pendingSync = totalFlows - syncedFlows;
  const recentlyUpdated = flows.filter((flow) => {
    if (!flow.updated_at) return false;
    const updated = Date.parse(flow.updated_at);
    if (Number.isNaN(updated)) return false;
    const diff = Date.now() - updated;
    return diff <= 1000 * 60 * 60 * 24 * 7;
  }).length;

  const selectedFlow: FlowInfo | undefined = selectedId
    ? flows.find((flow) => flow.local_id === selectedId)
    : undefined;

  const handleSelect = (flow: FlowInfo) => {
    setSelectedId(flow.local_id);
    setRunOutput(null);
    setRunError(null);
  };

  const handleRun = async () => {
    if (!selectedFlow) return;
    setSending(true);
    setRunError(null);
    try {
      const response = await runFlow(selectedFlow.local_id, {
        input_value: inputValue,
        ensure: true
      });
      setRunOutput(response);
    } catch (err) {
      setRunOutput(null);
      setRunError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setSending(false);
    }
  };

  return (
    <Box>
      <Panel
        title="AgentFlow"
        subtitle="Manage, sync, and execute LangFlow pipelines"
        actions={
          <Button
            variant="outlined"
            size="small"
            startIcon={loading ? <CircularProgress size={16} /> : <RefreshIcon />}
            onClick={refresh}
            disabled={loading}
          >
            {loading ? "Refreshing…" : "Refresh"}
          </Button>
        }
      >
        <Box sx={{ mb: 3 }}>
          <Typography variant="h5" gutterBottom>
            Curate your orchestration lineup.
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            Preview local flows, monitor LangFlow sync status, and execute test runs without leaving
            the browser shell.
          </Typography>

          <Stack direction="row" spacing={2} sx={{ mt: 3, flexWrap: 'wrap' }}>
            <Card variant="outlined" sx={{ flex: 1, minWidth: 150 }}>
              <CardContent>
                <Typography variant="caption" color="text.secondary">
                  Total flows
                </Typography>
                <Typography variant="h6">{totalFlows}</Typography>
              </CardContent>
            </Card>
            <Card variant="outlined" sx={{ flex: 1, minWidth: 150 }}>
              <CardContent>
                <Typography variant="caption" color="text.secondary">
                  Synced to LangFlow
                </Typography>
                <Typography variant="h6">{syncedFlows}</Typography>
                <Typography variant="caption" color="text.secondary">
                  {pendingSync} pending sync
                </Typography>
              </CardContent>
            </Card>
            <Card variant="outlined" sx={{ flex: 1, minWidth: 150 }}>
              <CardContent>
                <Typography variant="caption" color="text.secondary">
                  Updated this week
                </Typography>
                <Typography variant="h6">{recentlyUpdated}</Typography>
              </CardContent>
            </Card>
            <Card variant="outlined" sx={{ flex: 1, minWidth: 150 }}>
              <CardContent>
                <Typography variant="caption" color="text.secondary">
                  Selected flow
                </Typography>
                <Typography variant="h6">{selectedFlow?.name ?? "—"}</Typography>
                <Typography variant="caption" color="text.secondary">
                  {selectedFlow ? selectedFlow.local_id : "Choose from the ledger"}
                </Typography>
              </CardContent>
            </Card>
          </Stack>

          {error ? (
            <Alert severity="error" sx={{ mt: 2 }}>
              Unable to load flows: {error.message}
            </Alert>
          ) : null}
        </Box>
      </Panel>

      <Stack direction={{ xs: 'column', md: 'row' }} spacing={2}>
        <Box sx={{ flex: 2 }}>
          <Panel title="Flow ledger" subtitle="Local specs and sync status">
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Name</TableCell>
                    <TableCell>Project</TableCell>
                    <TableCell>Folder</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Updated</TableCell>
                    <TableCell>Synced</TableCell>
                    <TableCell align="right">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {flows.map((flow) => (
                    <TableRow
                      key={flow.local_id}
                      hover
                      selected={flow.local_id === selectedId}
                      onClick={() => handleSelect(flow)}
                      sx={{ cursor: 'pointer' }}
                    >
                      <TableCell>
                        <Typography variant="body2" fontWeight={500}>
                          {flow.name ?? flow.local_id}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {truncate(flow.description)}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {flow.project_id ?? "—"}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {flow.folder_path ?? "—"}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        {flow.remote_id ? (
                          <Chip label="Synced" size="small" color="success" />
                        ) : (
                          <Chip label="Local only" size="small" color="warning" />
                        )}
                      </TableCell>
                      <TableCell>
                        {flow.updated_at ? (
                          <>
                            <Typography variant="body2">
                              {formatter.format(new Date(flow.updated_at))}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {relativeTime(flow.updated_at)}
                            </Typography>
                          </>
                        ) : (
                          "—"
                        )}
                      </TableCell>
                      <TableCell>
                        {flow.synced_at ? (
                          <>
                            <Typography variant="body2">
                              {formatter.format(new Date(flow.synced_at))}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {relativeTime(flow.synced_at)}
                            </Typography>
                          </>
                        ) : (
                          "—"
                        )}
                      </TableCell>
                      <TableCell align="right">
                        <Button
                          variant="outlined"
                          size="small"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleSelect(flow);
                          }}
                        >
                          View
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                  {!flows.length && !loading ? (
                    <TableRow>
                      <TableCell colSpan={7} align="center" sx={{ py: 4 }}>
                        <Typography variant="body2" color="text.secondary">
                          No flows discovered in the AgentFlow catalog.
                        </Typography>
                      </TableCell>
                    </TableRow>
                  ) : null}
                </TableBody>
              </Table>
            </TableContainer>
          </Panel>
        </Box>

        <Box sx={{ flex: 1 }}>
          <Panel title="Execution studio" subtitle="Run the selected flow">
            {selectedFlow ? (
              <Stack spacing={2}>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    Local ID
                  </Typography>
                  <Typography variant="body2">{selectedFlow.local_id}</Typography>
                </Box>
                <Divider />
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    Remote ID
                  </Typography>
                  <Typography variant="body2">
                    {selectedFlow.remote_id ?? "Pending sync"}
                  </Typography>
                </Box>
                <Divider />
                <TextField
                  fullWidth
                  label="Input value"
                  multiline
                  rows={4}
                  value={inputValue}
                  onChange={(event) => setInputValue(event.target.value)}
                  placeholder="Provide the primary input value for this flow run"
                />
                <Button
                  variant="contained"
                  startIcon={<PlayArrowIcon />}
                  onClick={handleRun}
                  disabled={!inputValue.trim() || sending}
                  fullWidth
                >
                  {sending ? "Running…" : "Run flow"}
                </Button>

                {runError ? (
                  <Alert severity="error">Run failed: {runError.message}</Alert>
                ) : null}

                {runOutput ? (
                  <Paper variant="outlined" sx={{ p: 2 }}>
                    <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
                      Result payload
                    </Typography>
                    <Box
                      component="pre"
                      sx={{
                        bgcolor: 'grey.100',
                        p: 1,
                        borderRadius: 1,
                        overflow: 'auto',
                        fontSize: '0.75rem',
                        fontFamily: 'monospace'
                      }}
                    >
                      {JSON.stringify(runOutput.result, null, 2)}
                    </Box>
                    {runOutput.deepagents_analysis ? (
                      <>
                        <Typography variant="caption" color="text.secondary" sx={{ mt: 2, mb: 1, display: 'block' }}>
                          DeepAgents analysis
                        </Typography>
                        <Box
                          component="pre"
                          sx={{
                            bgcolor: 'grey.100',
                            p: 1,
                            borderRadius: 1,
                            overflow: 'auto',
                            fontSize: '0.75rem',
                            fontFamily: 'monospace'
                          }}
                        >
                          {JSON.stringify(runOutput.deepagents_analysis, null, 2)}
                        </Box>
                      </>
                    ) : null}
                  </Paper>
                ) : null}
              </Stack>
            ) : (
              <Paper variant="outlined" sx={{ p: 3, textAlign: 'center' }}>
                <Typography variant="body2" color="text.secondary">
                  Select a flow from the ledger to run it against LangFlow.
                </Typography>
              </Paper>
            )}
          </Panel>
        </Box>
      </Stack>
    </Box>
  );
}
