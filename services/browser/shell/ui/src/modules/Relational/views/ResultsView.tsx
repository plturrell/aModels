/**
 * Relational Results View
 * Displays processing results and intelligence for relational tables
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
  TextField,
  Button,
  List,
  ListItem,
  ListItemText,
  Divider
} from "@mui/material";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import RefreshIcon from "@mui/icons-material/Refresh";
import { Panel } from "../../../components/Panel";
import type { RelationalTable, RelationalRequestIntelligence } from "../../../api/relational";

interface ResultsViewProps {
  requestId: string;
  tables: RelationalTable[] | null;
  intelligence: RelationalRequestIntelligence | null;
  loading: boolean;
  error: string | null;
  onRequestIdChange: (id: string) => void;
  onRefresh: () => void;
}

export function ResultsView({
  requestId,
  tables,
  intelligence,
  loading,
  error,
  onRequestIdChange,
  onRefresh
}: ResultsViewProps) {
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

  return (
    <Stack spacing={2}>
      <Panel title="Results" dense>
        <Stack spacing={2}>
          <Box display="flex" justifyContent="space-between" alignItems="center">
            <TextField
              label="Request ID"
              value={requestId}
              onChange={(e) => onRequestIdChange(e.target.value)}
              placeholder="Enter request ID"
              size="small"
              sx={{ flex: 1, mr: 2 }}
            />
            <Button startIcon={<RefreshIcon />} onClick={onRefresh} size="small">
              Refresh
            </Button>
          </Box>

          {intelligence && (
            <Card variant="outlined">
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Intelligence Summary
                </Typography>
                <Stack direction="row" spacing={2} mt={2}>
                  <Box>
                    <Typography variant="h4" color="primary">
                      {intelligence.domains?.length || 0}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Domains
                    </Typography>
                  </Box>
                  <Box>
                    <Typography variant="h4" color="success.main">
                      {intelligence.total_relationships || 0}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Relationships
                    </Typography>
                  </Box>
                  <Box>
                    <Typography variant="h4" color="warning.main">
                      {intelligence.total_patterns || 0}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Patterns
                    </Typography>
                  </Box>
                  <Box>
                    <Typography variant="h4" color="secondary">
                      {intelligence.knowledge_graph_nodes || 0}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      KG Nodes
                    </Typography>
                  </Box>
                </Stack>
              </CardContent>
            </Card>
          )}

          {tables && tables.length > 0 ? (
            <List>
              {tables.map((table, index) => (
                <React.Fragment key={table.id}>
                  <ListItem>
                    <Box sx={{ flex: 1 }}>
                      <Box display="flex" alignItems="center" gap={1} mb={0.5}>
                        {table.status === "succeeded" ? (
                          <CheckCircleIcon color="success" sx={{ fontSize: 16 }} />
                        ) : null}
                        <Typography variant="body2" fontWeight={500}>
                          {table.title || table.id}
                        </Typography>
                        <Chip label={table.status} size="small" sx={{ ml: "auto" }} />
                      </Box>
                      {table.intelligence?.domain && (
                        <Typography variant="caption" color="text.secondary">
                          Domain: {table.intelligence.domain}
                        </Typography>
                      )}
                    </Box>
                  </ListItem>
                  {index < tables.length - 1 && <Divider />}
                </React.Fragment>
              ))}
            </List>
          ) : (
            <Typography variant="body2" color="text.secondary" sx={{ textAlign: "center", py: 4 }}>
              No tables processed yet. Enter a request ID to view results.
            </Typography>
          )}
        </Stack>
      </Panel>
    </Stack>
  );
}

