import { useMemo } from "react";
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
  List,
  ListItem,
  ListItemText
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';

import { Panel } from "../../components/Panel";
import { useDocuments, type DocumentRecord } from "../../api/hooks";

const formatter = new Intl.DateTimeFormat(undefined, {
  year: "numeric",
  month: "short",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit"
});

const relativeFormatter = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });

const getRelativeTime = (iso: string) => {
  const now = Date.now();
  const timestamp = Date.parse(iso);
  if (Number.isNaN(timestamp)) return "unknown";
  const diffMs = timestamp - now;
  const diffMinutes = Math.round(diffMs / (1000 * 60));
  if (Math.abs(diffMinutes) < 60) {
    return relativeFormatter.format(diffMinutes, "minute");
  }
  const diffHours = Math.round(diffMinutes / 60);
  if (Math.abs(diffHours) < 24) {
    return relativeFormatter.format(diffHours, "hour");
  }
  const diffDays = Math.round(diffHours / 24);
  return relativeFormatter.format(diffDays, "day");
};

const createdThisWeek = (documents: DocumentRecord[]) => {
  const now = new Date();
  const startOfWeek = new Date(now);
  startOfWeek.setDate(now.getDate() - now.getDay());
  startOfWeek.setHours(0, 0, 0, 0);
  return documents.filter((doc) => {
    const created = Date.parse(doc.created_at);
    return !Number.isNaN(created) && created >= startOfWeek.getTime();
  }).length;
};

const truncate = (value?: string | null, length = 160) => {
  if (!value) return null;
  return value.length > length ? `${value.slice(0, length - 1)}…` : value;
};

export function DocumentsModule() {
  const { data, loading, error, refresh } = useDocuments();

  const documents = useMemo(() => {
    const items = data ?? [];
    return [...items].sort(
      (a, b) => Date.parse(b.created_at) - Date.parse(a.created_at)
    );
  }, [data]);

  const totalDocuments = documents.length;
  const newThisWeek = createdThisWeek(documents);
  const mostRecent = documents[0];

  return (
    <Box>
      <Panel
        title="Document Library"
        subtitle="Curate, relate, and explore the latest uploads"
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
            Everything you ingest lands here.
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            Track source material, surface relationships, and keep tabs on the freshest additions
            to your knowledge graph.
          </Typography>

          <Stack direction="row" spacing={2} sx={{ mt: 3, flexWrap: 'wrap' }}>
            <Card variant="outlined" sx={{ flex: 1, minWidth: 150 }}>
              <CardContent>
                <Typography variant="caption" color="text.secondary">
                  Total documents
                </Typography>
                <Typography variant="h6">{totalDocuments}</Typography>
              </CardContent>
            </Card>
            <Card variant="outlined" sx={{ flex: 1, minWidth: 150 }}>
              <CardContent>
                <Typography variant="caption" color="text.secondary">
                  Added this week
                </Typography>
                <Typography variant="h6">{newThisWeek}</Typography>
              </CardContent>
            </Card>
            <Card variant="outlined" sx={{ flex: 1, minWidth: 150 }}>
              <CardContent>
                <Typography variant="caption" color="text.secondary">
                  Latest upload
                </Typography>
                <Typography variant="h6">
                  {mostRecent ? formatter.format(new Date(mostRecent.created_at)) : "—"}
                </Typography>
              </CardContent>
            </Card>
            <Card variant="outlined" sx={{ flex: 1, minWidth: 150 }}>
              <CardContent>
                <Typography variant="caption" color="text.secondary">
                  Catalog coverage
                </Typography>
                <Typography variant="h6">
                  {documents.filter((doc) => Boolean(doc.catalog_identifier)).length} linked
                </Typography>
              </CardContent>
            </Card>
          </Stack>

          {error ? (
            <Alert severity="error" sx={{ mt: 2 }}>
              Unable to load documents: {error.message}
            </Alert>
          ) : null}

          {!totalDocuments && !loading ? (
            <Paper variant="outlined" sx={{ p: 3, mt: 2, textAlign: 'center' }}>
              <Chip label="Awaiting first upload" color="info" sx={{ mb: 2 }} />
              <Typography variant="body2" color="text.secondary">
                Once the ingestion pipeline lands a document, this space will highlight the latest
                additions along with their relationships.
              </Typography>
            </Paper>
          ) : null}
        </Box>
      </Panel>

      <Stack direction={{ xs: 'column', md: 'row' }} spacing={2}>
        <Box sx={{ flex: 2 }}>
          <Panel title="Ledger" subtitle="Recently captured documents">
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Name</TableCell>
                    <TableCell>Description</TableCell>
                    <TableCell>Data Product</TableCell>
                    <TableCell>Insights</TableCell>
                    <TableCell>Created</TableCell>
                    <TableCell>Updated</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {documents.slice(0, 12).map((doc) => (
                    <TableRow key={doc.id} hover>
                      <TableCell>
                        <Typography variant="body2" fontWeight={500}>
                          {doc.name}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {doc.id}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {doc.description ?? "—"}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        {doc.catalog_identifier ? (
                          <Chip label={doc.catalog_identifier} size="small" color="primary" />
                        ) : (
                          <Typography variant="caption" color="text.secondary">
                            Pending
                          </Typography>
                        )}
                      </TableCell>
                      <TableCell>
                        {doc.extraction_summary ? (
                          <Typography variant="body2" sx={{ maxWidth: 300 }}>
                            {truncate(doc.extraction_summary)}
                          </Typography>
                        ) : (
                          <Typography variant="caption" color="text.secondary">
                            Processing…
                          </Typography>
                        )}
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {formatter.format(new Date(doc.created_at))}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {getRelativeTime(doc.created_at)}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {formatter.format(new Date(doc.updated_at))}
                        </Typography>
                      </TableCell>
                    </TableRow>
                  ))}
                  {!documents.length ? (
                    <TableRow>
                      <TableCell colSpan={6} align="center" sx={{ py: 4 }}>
                        {loading ? (
                          <CircularProgress size={24} />
                        ) : (
                          <Typography variant="body2" color="text.secondary">
                            No documents yet.
                          </Typography>
                        )}
                      </TableCell>
                    </TableRow>
                  ) : null}
                </TableBody>
              </Table>
            </TableContainer>
          </Panel>
        </Box>

        <Box sx={{ flex: 1 }}>
          <Panel title="Next steps" subtitle="Bring the library to life" dense>
            <List>
              <ListItem>
                <ListItemText
                  primary="Upload via DMS API"
                  secondary="Use the `/documents` endpoint to push new files from pipelines or CLI scripts."
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="Annotate relationships"
                  secondary="Sync taxonomy edges into Neo4j so downstream agents can traverse them."
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="Enrich with embeddings"
                  secondary="Queue Celery jobs to generate pgvector embeddings for RAG-ready search."
                />
              </ListItem>
            </List>
          </Panel>
        </Box>
      </Stack>
    </Box>
  );
}
