/**
 * DMS Documents View
 * Displays the document library
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
  Divider
} from "@mui/material";
import { Panel } from "../../../components/Panel";
import type { DMSDocument } from "../../../api/dms";

interface DocumentsViewProps {
  documents: DMSDocument[] | null;
  loading: boolean;
  error: string | null;
}

export function DocumentsView({ documents, loading, error }: DocumentsViewProps) {
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

  if (!documents || documents.length === 0) {
    return (
      <Panel title="Documents" dense>
        <Typography variant="body2" color="text.secondary" sx={{ textAlign: "center", py: 4 }}>
          No documents found
        </Typography>
      </Panel>
    );
  }

  return (
    <Panel title={`Documents (${documents.length})`} dense>
      <List>
        {documents.map((doc, index) => (
          <React.Fragment key={doc.id}>
            <ListItem>
              <Box sx={{ flex: 1 }}>
                <Box display="flex" alignItems="center" gap={1} mb={1}>
                  <Typography variant="h6">{doc.name}</Typography>
                  <Chip
                    label={doc.catalog_identifier ? "Processed" : "Pending"}
                    size="small"
                    color={doc.catalog_identifier ? "success" : "warning"}
                  />
                </Box>
                {doc.description && (
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    {doc.description}
                  </Typography>
                )}
                <Typography variant="caption" color="text.secondary">
                  Created: {new Date(doc.created_at).toLocaleString()} • ID: {doc.id}
                </Typography>
              </Box>
            </ListItem>
            {index < documents.length - 1 && <Divider />}
          </React.Fragment>
        ))}
      </List>
    </Panel>
  );
}

