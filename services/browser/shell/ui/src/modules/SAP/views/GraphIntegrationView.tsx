/**
 * SAP Graph Integration View
 * 
 * Visualize extracted SAP schemas as knowledge graphs
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Button,
  Stack,
  Alert,
  Card,
  CardContent,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
} from '@mui/material';
import AccountTreeIcon from '@mui/icons-material/AccountTree';
import { Panel } from '../../../components/Panel';
import { GraphVisualization, LayoutType } from '../../../components/GraphVisualization';
import type { SAPBDCConnectionConfig } from '../../../api/sap';
import type { GraphData } from '../../../api/graph';

interface GraphIntegrationViewProps {
  connection: SAPBDCConnectionConfig;
}

export function GraphIntegrationView({ connection }: GraphIntegrationViewProps) {
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], edges: [] });
  const [layout, setLayout] = useState<LayoutType>('hierarchical');
  const [selectedNodes, setSelectedNodes] = useState<string[]>([]);

  // Try to load graph data from localStorage (set by SchemaExtractionView)
  useEffect(() => {
    try {
      const stored = localStorage.getItem('sap_extracted_graph');
      if (stored) {
        const parsed = JSON.parse(stored);
        setGraphData(parsed);
      }
    } catch {
      // Ignore parse errors
    }
  }, []);

  // Listen for graph data updates from schema extraction
  useEffect(() => {
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === 'sap_extracted_graph' && e.newValue) {
        try {
          const parsed = JSON.parse(e.newValue);
          setGraphData(parsed);
        } catch {
          // Ignore parse errors
        }
      }
    };

    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, []);

  const handleNodeClick = (nodeId: string, node: any) => {
    setSelectedNodes((prev) =>
      prev.includes(nodeId) ? prev.filter((id) => id !== nodeId) : [...prev, nodeId]
    );
  };

  if (graphData.nodes.length === 0) {
    return (
      <Stack spacing={3}>
        <Panel title="Graph Visualization" dense>
          <Alert severity="info">
            No graph data available. Extract a schema from the "Schema Extraction" tab first.
          </Alert>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
            To visualize SAP schemas as knowledge graphs:
          </Typography>
          <ol style={{ marginTop: 8, paddingLeft: 20 }}>
            <li>Go to the "Schema Extraction" tab</li>
            <li>Configure your extraction parameters</li>
            <li>Click "Extract Schema"</li>
            <li>Return to this tab to view the graph visualization</li>
          </ol>
        </Panel>
      </Stack>
    );
  }

  return (
    <Stack spacing={3}>
      <Panel
        title="SAP Schema Knowledge Graph"
        subtitle={`Visualizing ${graphData.nodes.length} nodes and ${graphData.edges.length} relationships`}
        dense
        actions={
          <FormControl size="small" sx={{ minWidth: 200 }}>
            <InputLabel>Layout</InputLabel>
            <Select
              value={layout}
              onChange={(e) => setLayout(e.target.value as LayoutType)}
              label="Layout"
            >
              <MenuItem value="hierarchical">Hierarchical</MenuItem>
              <MenuItem value="force-directed">Force-Directed</MenuItem>
              <MenuItem value="circular">Circular</MenuItem>
              <MenuItem value="breadthfirst">Breadth-First</MenuItem>
              <MenuItem value="dagre">DAG Layout</MenuItem>
              <MenuItem value="cola">Force-Atlas</MenuItem>
            </Select>
          </FormControl>
        }
      >
        <Box sx={{ height: 600, border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
          <GraphVisualization
            graphData={graphData}
            layout={layout}
            onNodeClick={handleNodeClick}
            selectedNodes={selectedNodes}
            height={600}
            showControls={true}
          />
        </Box>
      </Panel>

      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Graph Statistics
          </Typography>
          <Stack direction="row" spacing={4}>
            <Box>
              <Typography variant="caption" color="text.secondary">
                Total Nodes
              </Typography>
              <Typography variant="h5">{graphData.nodes.length}</Typography>
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary">
                Total Edges
              </Typography>
              <Typography variant="h5">{graphData.edges.length}</Typography>
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary">
                Node Types
              </Typography>
              <Typography variant="h5">
                {new Set(graphData.nodes.map((n) => n.type)).size}
              </Typography>
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary">
                Selected
              </Typography>
              <Typography variant="h5">{selectedNodes.length}</Typography>
            </Box>
          </Stack>
        </CardContent>
      </Card>

      {selectedNodes.length > 0 && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Selected Nodes
            </Typography>
            <Stack spacing={1}>
              {selectedNodes.map((nodeId) => {
                const node = graphData.nodes.find((n) => n.id === nodeId);
                return node ? (
                  <Card key={nodeId} variant="outlined">
                    <CardContent>
                      <Typography variant="subtitle1">{node.label || node.id}</Typography>
                      <Typography variant="caption" color="text.secondary">
                        Type: {node.type || 'unknown'} | ID: {node.id}
                      </Typography>
                    </CardContent>
                  </Card>
                ) : null;
              })}
            </Stack>
          </CardContent>
        </Card>
      )}
    </Stack>
  );
}


