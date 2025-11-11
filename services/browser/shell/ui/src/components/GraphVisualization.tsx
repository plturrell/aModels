/**
 * Phase 1.1: Graph Visualization Component
 * 
 * Interactive graph visualization using Cytoscape.js with support for:
 * - Force-directed, hierarchical, and circular layouts
 * - Zoom, pan, drag operations
 * - Node/edge selection and highlighting
 * - Performance optimization for 10K+ nodes
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import CytoscapeComponent from 'react-cytoscapejs';
import cytoscape from 'cytoscape';
import dagre from 'cytoscape-dagre';
import cola from 'cytoscape-cola';
import coseBilkent from 'cytoscape-cose-bilkent';
import { Box, Paper, Typography, Select, MenuItem, FormControl, InputLabel, Button, IconButton, Tooltip, Slider } from '@mui/material';
import ZoomInIcon from '@mui/icons-material/ZoomIn';
import ZoomOutIcon from '@mui/icons-material/ZoomOut';
import FitScreenIcon from '@mui/icons-material/FitScreen';
import CenterFocusStrongIcon from '@mui/icons-material/CenterFocusStrong';
import { GraphNode, GraphEdge, GraphData } from '../types/graph';

// Register Cytoscape extensions
cytoscape.use(dagre);
cytoscape.use(cola);
cytoscape.use(coseBilkent);

export type LayoutType = 'force-directed' | 'hierarchical' | 'circular' | 'breadthfirst' | 'cose-bilkent' | 'dagre' | 'cola';

export interface GraphVisualizationProps {
  graphData: GraphData;
  layout?: LayoutType;
  onNodeClick?: (nodeId: string, node: GraphNode) => void;
  onEdgeClick?: (edgeId: string, edge: GraphEdge) => void;
  onNodeSelect?: (nodeIds: string[]) => void;
  selectedNodes?: string[];
  height?: number;
  enableInteractions?: boolean;
  maxNodes?: number; // Performance limit
  showControls?: boolean;
}

const layoutConfigs: Record<LayoutType, any> = {
  'force-directed': {
    name: 'cose-bilkent',
    idealEdgeLength: 100,
    nodeRepulsion: 4500,
    gravity: 0.25,
    gravityRange: 3.8,
    numIter: 2500,
    tile: true,
    animate: true,
    animationDuration: 1000,
  },
  'hierarchical': {
    name: 'dagre',
    rankDir: 'TB',
    spacingFactor: 1.25,
    nodeSep: 50,
    edgeSep: 20,
    rankSep: 75,
  },
  'circular': {
    name: 'circle',
    spacingFactor: 1.75,
    radius: null,
    startAngle: 0,
    sweep: null,
    clockwise: true,
    sort: undefined,
  },
  'breadthfirst': {
    name: 'breadthfirst',
    directed: true,
    padding: 10,
    spacingFactor: 1.5,
    avoidOverlap: true,
    nodeDimensionsIncludeLabels: false,
  },
  'cose-bilkent': {
    name: 'cose-bilkent',
    idealEdgeLength: 100,
    nodeRepulsion: 4500,
    gravity: 0.25,
    gravityRange: 3.8,
    numIter: 2500,
    tile: true,
    animate: true,
  },
  'dagre': {
    name: 'dagre',
    rankDir: 'TB',
    spacingFactor: 1.25,
  },
  'cola': {
    name: 'cola',
    animate: true,
    maxSimulationTime: 4000,
    ungrabifyWhileSimulating: false,
    fit: true,
    padding: 30,
    nodeDimensionsIncludeLabels: false,
  },
};

const defaultStylesheet = [
  {
    selector: 'node',
    style: {
      'background-color': 'data(color)',
      'label': 'data(label)',
      'width': 30,
      'height': 30,
      'font-size': '12px',
      'text-valign': 'center',
      'text-halign': 'center',
      'text-wrap': 'wrap',
      'text-max-width': '100px',
      'color': '#000',
      'border-width': 2,
      'border-color': '#4A90E2',
      'shape': 'ellipse',
    },
  },
  {
    selector: 'node:selected',
    style: {
      'background-color': '#F5A623',
      'border-width': 3,
      'border-color': '#D68910',
    },
  },
  {
    selector: 'node[type="table"]',
    style: {
      'background-color': '#50C878',
      'shape': 'rectangle',
    },
  },
  {
    selector: 'node[type="column"]',
    style: {
      'background-color': '#87CEEB',
      'shape': 'diamond',
    },
  },
  {
    selector: 'edge',
    style: {
      'width': 2,
      'line-color': '#A0A0A0',
      'target-arrow-color': '#A0A0A0',
      'target-arrow-shape': 'triangle',
      'curve-style': 'bezier',
      'label': 'data(label)',
      'font-size': '10px',
      'text-rotation': 'autorotate',
      'text-margin-y': -10,
    },
  },
  {
    selector: 'edge:selected',
    style: {
      'width': 4,
      'line-color': '#F5A623',
      'target-arrow-color': '#F5A623',
    },
  },
];

export function GraphVisualization({
  graphData,
  layout = 'force-directed',
  onNodeClick,
  onEdgeClick,
  onNodeSelect,
  selectedNodes = [],
  height = 600,
  enableInteractions = true,
  maxNodes = 10000,
  showControls = true,
}: GraphVisualizationProps) {
  const cyRef = useRef<cytoscape.Core | null>(null);
  const [currentLayout, setCurrentLayout] = useState<LayoutType>(layout);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [nodeCount, setNodeCount] = useState(0);
  const [edgeCount, setEdgeCount] = useState(0);

  // Convert graph data to Cytoscape format
  const elements = React.useMemo(() => {
    let nodes = graphData.nodes || [];
    let edges = graphData.edges || [];

    // Performance optimization: limit nodes if too many
    if (nodes.length > maxNodes) {
      // Prioritize nodes with more connections
      const nodeConnections = new Map<string, number>();
      edges.forEach(edge => {
        nodeConnections.set(edge.source_id, (nodeConnections.get(edge.source_id) || 0) + 1);
        nodeConnections.set(edge.target_id, (nodeConnections.get(edge.target_id) || 0) + 1);
      });
      
      nodes = nodes
        .map(node => ({ node, connections: nodeConnections.get(node.id) || 0 }))
        .sort((a, b) => b.connections - a.connections)
        .slice(0, maxNodes)
        .map(item => item.node);
      
      // Filter edges to only include selected nodes
      const nodeIds = new Set(nodes.map(n => n.id));
      edges = edges.filter(edge => 
        nodeIds.has(edge.source_id) && nodeIds.has(edge.target_id)
      );
    }

    setNodeCount(nodes.length);
    setEdgeCount(edges.length);

    const cyNodes = nodes.map(node => ({
      data: {
        id: node.id,
        label: node.label || node.id,
        type: node.type,
        color: node.properties?.color || '#619BD6', // Use color from properties if available
        ...(node.properties || {}),
      },
      classes: node.type ? `type-${node.type}` : '',
    }));

    const cyEdges = edges.map(edge => ({
      data: {
        id: edge.id || `${edge.source_id}-${edge.target_id}`,
        source: edge.source_id,
        target: edge.target_id,
        label: edge.label,
        ...(edge.properties || {}),
      },
    }));

    return [...cyNodes, ...cyEdges];
  }, [graphData, maxNodes]);

  // Handle graph initialization
  const handleCyInit = useCallback((cy: cytoscape.Core) => {
    cyRef.current = cy;

    // Set up event handlers
    if (enableInteractions) {
      cy.on('tap', 'node', (evt) => {
        const node = evt.target;
        const nodeData = node.data();
        if (onNodeClick) {
          const graphNode = graphData.nodes.find(n => n.id === nodeData.id);
          if (graphNode) {
            onNodeClick(nodeData.id, graphNode);
          }
        }
      });

      cy.on('tap', 'edge', (evt) => {
        const edge = evt.target;
        const edgeData = edge.data();
        if (onEdgeClick) {
          const graphEdge = graphData.edges.find(e => 
            e.id === edgeData.id || 
            (e.source_id === edgeData.source && e.target_id === edgeData.target)
          );
          if (graphEdge) {
            onEdgeClick(edgeData.id, graphEdge);
          }
        }
      });

      cy.on('select', 'node', () => {
        if (onNodeSelect) {
          const selected = cy.$('node:selected').map(node => node.id());
          onNodeSelect(selected);
        }
      });

      // Update zoom level
      cy.on('zoom', () => {
        setZoomLevel(cy.zoom());
      });
    }

    // Apply layout
    const layoutConfig = layoutConfigs[currentLayout];
    if (layoutConfig) {
      const layoutInstance = cy.layout(layoutConfig);
      layoutInstance.run();
    }

    // Fit to viewport
    cy.fit(undefined, 50);
  }, [graphData, currentLayout, enableInteractions, onNodeClick, onEdgeClick, onNodeSelect]);

  // Update selected nodes
  useEffect(() => {
    if (cyRef.current && selectedNodes.length > 0) {
      cyRef.current.$('node').removeClass('selected');
      selectedNodes.forEach(nodeId => {
        cyRef.current?.$(`#${nodeId}`).addClass('selected');
      });
    }
  }, [selectedNodes]);

  // Handle layout change
  const handleLayoutChange = (newLayout: LayoutType) => {
    setCurrentLayout(newLayout);
    if (cyRef.current) {
      const layoutConfig = layoutConfigs[newLayout];
      if (layoutConfig) {
        const layoutInstance = cyRef.current.layout(layoutConfig);
        layoutInstance.run();
      }
    }
  };

  // Zoom controls
  const handleZoomIn = () => {
    if (cyRef.current) {
      cyRef.current.zoom(cyRef.current.zoom() * 1.2);
      setZoomLevel(cyRef.current.zoom());
    }
  };

  const handleZoomOut = () => {
    if (cyRef.current) {
      cyRef.current.zoom(cyRef.current.zoom() * 0.8);
      setZoomLevel(cyRef.current.zoom());
    }
  };

  const handleFit = () => {
    if (cyRef.current) {
      cyRef.current.fit(undefined, 50);
      setZoomLevel(cyRef.current.zoom());
    }
  };

  const handleCenter = () => {
    if (cyRef.current && selectedNodes.length > 0) {
      const firstSelected = cyRef.current.$(`#${selectedNodes[0]}`);
      if (firstSelected.length > 0) {
        cyRef.current.center(firstSelected);
        cyRef.current.zoom(2);
        setZoomLevel(cyRef.current.zoom());
      }
    } else {
      cyRef.current?.center();
    }
  };

  if (elements.length === 0) {
    return (
      <Paper sx={{ p: 3, height, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Typography variant="body1" color="text.secondary">
          No graph data available
        </Typography>
      </Paper>
    );
  }

  return (
    <Box sx={{ position: 'relative', height, width: '100%' }}>
      {showControls && (
        <Box
          sx={{
            position: 'absolute',
            top: 10,
            right: 10,
            zIndex: 1000,
            display: 'flex',
            flexDirection: 'column',
            gap: 1,
            backgroundColor: 'rgba(255, 255, 255, 0.9)',
            padding: 1,
            borderRadius: 1,
            boxShadow: 2,
          }}
        >
          <FormControl size="small" sx={{ minWidth: 150 }}>
            <InputLabel>Layout</InputLabel>
            <Select
              value={currentLayout}
              label="Layout"
              onChange={(e) => handleLayoutChange(e.target.value as LayoutType)}
            >
              <MenuItem value="force-directed">Force-Directed</MenuItem>
              <MenuItem value="hierarchical">Hierarchical</MenuItem>
              <MenuItem value="circular">Circular</MenuItem>
              <MenuItem value="breadthfirst">Breadth-First</MenuItem>
              <MenuItem value="cose-bilkent">COSE-Bilkent</MenuItem>
              <MenuItem value="dagre">Dagre</MenuItem>
              <MenuItem value="cola">Cola</MenuItem>
            </Select>
          </FormControl>

          <Box sx={{ display: 'flex', gap: 0.5 }}>
            <Tooltip title="Zoom In">
              <IconButton size="small" onClick={handleZoomIn}>
                <ZoomInIcon />
              </IconButton>
            </Tooltip>
            <Tooltip title="Zoom Out">
              <IconButton size="small" onClick={handleZoomOut}>
                <ZoomOutIcon />
              </IconButton>
            </Tooltip>
            <Tooltip title="Fit to Screen">
              <IconButton size="small" onClick={handleFit}>
                <FitScreenIcon />
              </IconButton>
            </Tooltip>
            <Tooltip title="Center Selected">
              <IconButton size="small" onClick={handleCenter}>
                <CenterFocusStrongIcon />
              </IconButton>
            </Tooltip>
          </Box>

          <Typography variant="caption" sx={{ textAlign: 'center', color: 'text.secondary' }}>
            {nodeCount} nodes, {edgeCount} edges
            {graphData.nodes.length > maxNodes && ` (showing top ${maxNodes})`}
          </Typography>
        </Box>
      )}

      <CytoscapeComponent
        elements={elements}
        style={{ width: '100%', height: '100%' }}
        cy={(cy) => handleCyInit(cy)}
        stylesheet={defaultStylesheet}
        layout={layoutConfigs[currentLayout]}
        wheelSensitivity={0.1}
        minZoom={0.1}
        maxZoom={2}
      />
    </Box>
  );
}

