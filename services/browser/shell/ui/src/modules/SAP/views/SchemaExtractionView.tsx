/**
 * SAP Schema Extraction View
 * 
 * Extract and view schemas from SAP Datasphere and HANA Data Lake
 */

import React, { useState } from 'react';
import {
  Box,
  Typography,
  TextField,
  Button,
  Stack,
  Alert,
  CircularProgress,
  Card,
  CardContent,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Checkbox,
  FormControlLabel,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import { Panel } from '../../../components/common/Panel';
import { extractFromSAPBDC, convertSAPSchemaToGraph } from '../../../api/sap';
import type { SAPBDCConnectionConfig, SAPBDCExtractRequest, SAPSchema } from '../../../api/sap';

interface SchemaExtractionViewProps {
  connection: SAPBDCConnectionConfig;
}

export function SchemaExtractionView({ connection }: SchemaExtractionViewProps) {
  const [extractRequest, setExtractRequest] = useState<SAPBDCExtractRequest>({
    formation_id: connection.formation_id,
    source_system: 'SAP S/4HANA Cloud',
    include_views: true,
  });
  const [extracting, setExtracting] = useState(false);
  const [extractResult, setExtractResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [graphData, setGraphData] = useState<{ nodes: any[]; edges: any[] } | null>(null);

  const handleExtract = async () => {
    setExtracting(true);
    setError(null);
    setExtractResult(null);
    setGraphData(null);

    try {
      const result = await extractFromSAPBDC(extractRequest);
      setExtractResult(result);

      if (result.success && result.schema) {
        // Convert schema to graph format
        const graph = convertSAPSchemaToGraph(result.schema);
        setGraphData(graph);
        // Store in localStorage for GraphIntegrationView
        try {
          localStorage.setItem('sap_extracted_graph', JSON.stringify(graph));
        } catch {
          // Ignore storage errors
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Extraction failed');
    } finally {
      setExtracting(false);
    }
  };

  return (
    <Stack spacing={3}>
      <Panel title="Schema Extraction" dense>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          Extract database schemas from SAP Datasphere spaces or SAP HANA Data Lake.
          Extracted schemas can be visualized as knowledge graphs.
        </Typography>

        <Stack spacing={2}>
          <TextField
            label="Formation ID"
            value={extractRequest.formation_id}
            onChange={(e) =>
              setExtractRequest({ ...extractRequest, formation_id: e.target.value })
            }
            fullWidth
            required
          />

          <TextField
            label="Source System"
            value={extractRequest.source_system}
            onChange={(e) =>
              setExtractRequest({ ...extractRequest, source_system: e.target.value })
            }
            fullWidth
            required
            helperText="e.g., SAP S/4HANA Cloud, SAP Datasphere"
          />

          <TextField
            label="Data Product ID (Optional)"
            value={extractRequest.data_product_id || ''}
            onChange={(e) =>
              setExtractRequest({ ...extractRequest, data_product_id: e.target.value || undefined })
            }
            fullWidth
            helperText="Specific data product to extract. Leave empty to extract all."
          />

          <TextField
            label="Space ID (Optional)"
            value={extractRequest.space_id || ''}
            onChange={(e) =>
              setExtractRequest({ ...extractRequest, space_id: e.target.value || undefined })
            }
            fullWidth
            helperText="SAP Datasphere space ID"
          />

          <FormControl fullWidth>
            <InputLabel>Database</InputLabel>
            <Select
              value={extractRequest.database || ''}
              onChange={(e) =>
                setExtractRequest({ ...extractRequest, database: e.target.value || undefined })
              }
              label="Database"
            >
              <MenuItem value="">All Databases</MenuItem>
              <MenuItem value="HANADB">HANA Database</MenuItem>
              <MenuItem value="DATASPHERE">Datasphere</MenuItem>
            </Select>
          </FormControl>

          <FormControlLabel
            control={
              <Checkbox
                checked={extractRequest.include_views}
                onChange={(e) =>
                  setExtractRequest({ ...extractRequest, include_views: e.target.checked })
                }
              />
            }
            label="Include Views"
          />

          <Button
            variant="contained"
            startIcon={extracting ? <CircularProgress size={20} /> : <PlayArrowIcon />}
            onClick={handleExtract}
            disabled={extracting || !extractRequest.formation_id || !extractRequest.source_system}
            fullWidth
          >
            {extracting ? 'Extracting Schema...' : 'Extract Schema'}
          </Button>
        </Stack>
      </Panel>

      {error && (
        <Alert severity="error">
          {error}
        </Alert>
      )}

      {extractResult && (
        <>
          {extractResult.success ? (
            <Alert severity="success">
              Schema extracted successfully! {extractResult.schema && `Found ${extractResult.schema.tables.length} tables.`}
            </Alert>
          ) : (
            <Alert severity="error">
              {extractResult.error || 'Extraction failed'}
            </Alert>
          )}

          {extractResult.schema && (
            <Panel title="Extracted Schema" dense>
              <Accordion defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="h6">
                    {extractResult.schema.database}.{extractResult.schema.schema}
                  </Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Stack spacing={2}>
                    <Box>
                      <Typography variant="subtitle2" gutterBottom>
                        Tables ({extractResult.schema.tables.length})
                      </Typography>
                      {extractResult.schema.tables.map((table: any, idx: number) => (
                        <Accordion key={idx} sx={{ mb: 1 }}>
                          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                            <Typography>{table.name}</Typography>
                            <Chip
                              label={`${table.columns.length} columns`}
                              size="small"
                              sx={{ ml: 2 }}
                            />
                          </AccordionSummary>
                          <AccordionDetails>
                            <TableContainer component={Paper} variant="outlined">
                              <Table size="small">
                                <TableHead>
                                  <TableRow>
                                    <TableCell>Column</TableCell>
                                    <TableCell>Type</TableCell>
                                    <TableCell>Nullable</TableCell>
                                    <TableCell>Default</TableCell>
                                  </TableRow>
                                </TableHead>
                                <TableBody>
                                  {table.columns.map((col: any, colIdx: number) => (
                                    <TableRow key={colIdx}>
                                      <TableCell>{col.name}</TableCell>
                                      <TableCell>
                                        <Chip label={col.type} size="small" variant="outlined" />
                                      </TableCell>
                                      <TableCell>{col.nullable ? 'Yes' : 'No'}</TableCell>
                                      <TableCell>{col.default || '-'}</TableCell>
                                    </TableRow>
                                  ))}
                                </TableBody>
                              </Table>
                            </TableContainer>
                          </AccordionDetails>
                        </Accordion>
                      ))}
                    </Box>

                    {extractResult.schema.views && extractResult.schema.views.length > 0 && (
                      <Box>
                        <Typography variant="subtitle2" gutterBottom>
                          Views ({extractResult.schema.views.length})
                        </Typography>
                        {extractResult.schema.views.map((view: any, idx: number) => (
                          <Card key={idx} sx={{ mb: 1 }}>
                            <CardContent>
                              <Typography variant="h6">{view.name}</Typography>
                              <Typography variant="body2" color="text.secondary">
                                {view.columns.length} columns
                              </Typography>
                            </CardContent>
                          </Card>
                        ))}
                      </Box>
                    )}

                    {graphData && (
                      <Alert severity="info">
                        Graph data ready! Switch to the "Graph Visualization" tab to view the schema as a knowledge graph.
                        ({graphData.nodes.length} nodes, {graphData.edges.length} edges)
                      </Alert>
                    )}
                  </Stack>
                </AccordionDetails>
              </Accordion>
            </Panel>
          )}

          {extractResult.data_products && extractResult.data_products.length > 0 && (
            <Panel title="Data Products" dense>
              <Stack spacing={1}>
                {extractResult.data_products.map((product: any, idx: number) => (
                  <Card key={idx}>
                    <CardContent>
                      <Typography variant="h6">{product.name}</Typography>
                      <Typography variant="body2" color="text.secondary">
                        {product.id}
                      </Typography>
                    </CardContent>
                  </Card>
                ))}
              </Stack>
            </Panel>
          )}
        </>
      )}
    </Stack>
  );
}

