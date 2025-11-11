/**
 * SAP Graph View
 * 
 * SAP-specific graph visualization with schema lineage and data product relationships
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  Alert,
  CircularProgress,
  Stack,
  Card,
  CardContent,
  Chip,
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import { GraphVisualization, LayoutType } from '../../../components/GraphVisualization';
import { convertSAPSchemaToGraph, SAPSchema } from '../../../api/sap';
import { GraphData } from '../../../api/graph';

interface SAPGraphViewProps {
  schema?: SAPSchema;
  projectId?: string;
  systemId?: string;
}

export function SAPGraphView({ schema, projectId, systemId }: SAPGraphViewProps) {
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], edges: [] });
  const [layout, setLayout] = useState<LayoutType>('hierarchical');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (schema) {
      loadGraphFromSchema();
    }
  }, [schema]);

  const loadGraphFromSchema = () => {
    if (!schema) return;

    setLoading(true);
    setError(null);

    try {
      const graph = convertSAPSchemaToGraph(schema);
      setGraphData({
        nodes: graph.nodes,
        edges: graph.edges.map(e => ({
          source_id: e.source,
          target_id: e.target,
          type: e.type,
          properties: e.properties
        })),
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to convert schema to graph');
    } finally {
      setLoading(false);
    }
  };

  if (!schema) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="info">
          No SAP schema available. Extract a schema from SAP BDC to visualize it here.
        </Alert>
      </Box>
    );
  }

  return (
    <Box>
      <Stack spacing={3}>
        {/* Schema Summary */}
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Schema Summary
            </Typography>
            <Stack direction="row" spacing={2} sx={{ mt: 2 }}>
              <Chip label={`Database: ${schema.database}`} />
              <Chip label={`Schema: ${schema.schema}`} />
              <Chip label={`Tables: ${schema.tables.length}`} />
              {schema.views && <Chip label={`Views: ${schema.views.length}`} />}
            </Stack>
          </CardContent>
        </Card>

        {/* Graph Controls */}
        <Paper sx={{ p: 2 }}>
          <Stack direction="row" spacing={2} alignItems="center">
            <FormControl size="small" sx={{ minWidth: 200 }}>
              <InputLabel>Layout</InputLabel>
              <Select
                value={layout}
                label="Layout"
                onChange={(e) => setLayout(e.target.value as LayoutType)}
              >
                <MenuItem value="hierarchical">Hierarchical</MenuItem>
                <MenuItem value="force-directed">Force-Directed</MenuItem>
                <MenuItem value="dagre">Dagre</MenuItem>
                <MenuItem value="breadthfirst">Breadth-First</MenuItem>
              </Select>
            </FormControl>
            <Button
              startIcon={<RefreshIcon />}
              onClick={loadGraphFromSchema}
              disabled={loading}
            >
              Refresh
            </Button>
          </Stack>
        </Paper>

        {/* Error Display */}
        {error && (
          <Alert severity="error" onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {/* Graph Visualization */}
        <Paper sx={{ p: 2, height: '600px' }}>
          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
              <CircularProgress />
            </Box>
          ) : graphData.nodes.length === 0 ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
              <Typography color="textSecondary">No graph data available</Typography>
            </Box>
          ) : (
            <GraphVisualization
              graphData={graphData}
              layout={layout}
              height={600}
            />
          )}
        </Paper>

        {/* Schema Lineage Info */}
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Schema Lineage
            </Typography>
            <Typography variant="body2" color="textSecondary">
              This graph shows the structure of your SAP schema including:
            </Typography>
            <Stack spacing={1} sx={{ mt: 2 }}>
              <Typography variant="body2">
                • Database and schema relationships
              </Typography>
              <Typography variant="body2">
                • Table and column hierarchies
              </Typography>
              <Typography variant="body2">
                • Foreign key relationships between tables
              </Typography>
              {schema.views && (
                <Typography variant="body2">
                  • View definitions and dependencies
                </Typography>
              )}
            </Stack>
          </CardContent>
        </Card>
      </Stack>
    </Box>
  );
}

