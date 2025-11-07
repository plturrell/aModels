import { useMemo } from "react";
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
} from '@mui/material';
import { Panel } from "../../../components/Panel";
import type { ProcessedDocument } from "../../../api/perplexity";

interface ResultsViewProps {
  requestId: string;
  resultsData: any;
  intelligenceData: any;
  loading: boolean;
  error: Error | null;
}

export function ResultsView({ 
  requestId, 
  resultsData, 
  intelligenceData, 
  loading, 
  error 
}: ResultsViewProps) {
  const documents = useMemo(() => {
    return resultsData?.documents || [];
  }, [resultsData]);

  const intelligence = useMemo(() => {
    return intelligenceData?.intelligence || resultsData?.intelligence;
  }, [intelligenceData, resultsData]);

  if (!requestId) {
    return (
      <Alert severity="info">
        Enter a request ID above to view results, or submit a new query to get started.
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
        Failed to load results: {error.message}
      </Alert>
    );
  }

  if (!resultsData && !intelligenceData) {
    return (
      <Alert severity="warning">
        No results found for request ID: {requestId}
      </Alert>
    );
  }

  return (
    <Box>
      {/* Intelligence Summary */}
      {intelligence && (
        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} sx={{ mb: 3 }}>
          <Card sx={{ flex: 1 }}>
            <CardContent>
              <Typography variant="h4" color="primary" gutterBottom>
                {intelligence.domains?.length || 0}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Domains
              </Typography>
            </CardContent>
          </Card>
          <Card sx={{ flex: 1 }}>
            <CardContent>
              <Typography variant="h4" color="primary" gutterBottom>
                {intelligence.total_relationships || 0}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Relationships
              </Typography>
            </CardContent>
          </Card>
          <Card sx={{ flex: 1 }}>
            <CardContent>
              <Typography variant="h4" color="primary" gutterBottom>
                {intelligence.total_patterns || 0}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Patterns
              </Typography>
            </CardContent>
          </Card>
          <Card sx={{ flex: 1 }}>
            <CardContent>
              <Typography variant="h4" color="primary" gutterBottom>
                {intelligence.knowledge_graph_nodes || 0}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                KG Nodes
              </Typography>
            </CardContent>
          </Card>
        </Stack>
      )}

      {/* Documents List */}
      <Panel title="Processed Documents" subtitle={`${documents.length} document(s)`}>
        {documents.length === 0 ? (
          <Alert severity="info">
            No documents processed yet for this request.
          </Alert>
        ) : (
          <List>
            {documents.map((doc: ProcessedDocument, index: number) => (
              <Box key={doc.id}>
                <ListItem
                  sx={{
                    py: 2,
                    border: '1px solid',
                    borderColor: 'divider',
                    borderRadius: 1,
                    mb: 1,
                    bgcolor: doc.status === 'success' ? 'success.50' : 'error.50'
                  }}
                >
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                        <Typography variant="h6">
                          {doc.title || doc.id}
                        </Typography>
                        <Chip 
                          label={doc.status} 
                          color={doc.status === 'success' ? 'success' : 'error'}
                          size="small"
                        />
                        {doc.intelligence?.domain && (
                          <Chip 
                            label={doc.intelligence.domain} 
                            size="small"
                            variant="outlined"
                          />
                        )}
                      </Box>
                    }
                    secondary={
                      <Box>
                        <Typography variant="body2" color="text.secondary">
                          {doc.intelligence?.relationships?.length || 0} relationships • 
                          {doc.intelligence?.learned_patterns?.length || 0} patterns
                        </Typography>
                        {doc.error && (
                          <Typography variant="body2" color="error" sx={{ mt: 0.5 }}>
                            Error: {doc.error}
                          </Typography>
                        )}
                      </Box>
                    }
                  />
                </ListItem>
                {index < documents.length - 1 && <Divider />}
              </Box>
            ))}
          </List>
        )}
      </Panel>

      {/* Domains */}
      {intelligence?.domains && intelligence.domains.length > 0 && (
        <Panel title="Domains" dense>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            {intelligence.domains.map((domain: string, index: number) => (
              <Chip key={index} label={domain} variant="outlined" />
            ))}
          </Box>
        </Panel>
      )}
    </Box>
  );
}

