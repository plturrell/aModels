/**
 * DMS Results View
 * Displays processing results and intelligence data
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
  List,
  ListItem,
  ListItemText,
  Divider,
  Paper
} from "@mui/material";
import { Panel } from "../../../components/common/Panel";
import type { DMSProcessedDocument, DMSRequestIntelligence } from "../../../api/dms";

interface ResultsViewProps {
  documents: DMSProcessedDocument[] | null;
  intelligence: DMSRequestIntelligence | null;
  loading: boolean;
  error: string | null;
}

export function ResultsView({ documents, intelligence, loading, error }: ResultsViewProps) {
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
      {intelligence && (
        <Panel title="Intelligence Summary" dense>
          <Stack direction="row" spacing={2}>
            <Card variant="outlined" sx={{ flex: 1 }}>
              <CardContent>
                <Typography variant="h4" color="primary">
                  {intelligence.domains?.length || 0}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Domains
                </Typography>
              </CardContent>
            </Card>

            <Card variant="outlined" sx={{ flex: 1 }}>
              <CardContent>
                <Typography variant="h4" color="success.main">
                  {intelligence.total_relationships || 0}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Relationships
                </Typography>
              </CardContent>
            </Card>

            <Card variant="outlined" sx={{ flex: 1 }}>
              <CardContent>
                <Typography variant="h4" color="warning.main">
                  {intelligence.total_patterns || 0}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Patterns
                </Typography>
              </CardContent>
            </Card>

            <Card variant="outlined" sx={{ flex: 1 }}>
              <CardContent>
                <Typography variant="h4" color="secondary">
                  {intelligence.knowledge_graph_nodes || 0}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  KG Nodes
                </Typography>
              </CardContent>
            </Card>
          </Stack>
        </Panel>
      )}

      {documents && documents.length > 0 ? (
        <Panel title="Processed Documents" dense>
          <List>
            {documents.map((doc, index) => (
              <React.Fragment key={doc.id}>
                <ListItem>
                  <Box sx={{ flex: 1 }}>
                    <Box display="flex" alignItems="center" gap={1} mb={1}>
                      <Typography variant="h6">{doc.title || doc.id}</Typography>
                      <Chip label={doc.status} size="small" color={doc.status === "succeeded" ? "success" : "error"} />
                    </Box>
                    {doc.intelligence?.domain && (
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        Domain: {doc.intelligence.domain}
                      </Typography>
                    )}
                    <Typography variant="body2" color="text.secondary">
                      {doc.intelligence?.relationships?.length || 0} relationships •
                      {doc.intelligence?.learned_patterns?.length || 0} patterns
                    </Typography>
                    {doc.error && (
                      <Alert severity="error" sx={{ mt: 1 }}>
                        {doc.error}
                      </Alert>
                    )}
                  </Box>
                </ListItem>
                {index < documents.length - 1 && <Divider />}
              </React.Fragment>
            ))}
          </List>
        </Panel>
      ) : (
        <Panel title="Processed Documents" dense>
          <Typography variant="body2" color="text.secondary" sx={{ textAlign: "center", py: 4 }}>
            No documents available
          </Typography>
        </Panel>
      )}
    </Stack>
  );
}

