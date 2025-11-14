/**
 * GraphVisualization Component
 * 
 * Interactive graph visualization using Cytoscape.js
 */

import React, { useEffect, useRef } from 'react';
import CytoscapeComponent from 'react-cytoscapejs';
import cytoscape from 'cytoscape';
import cola from 'cytoscape-cola';
import dagre from 'cytoscape-dagre';
import coseBilkent from 'cytoscape-cose-bilkent';
import { Box } from '@mui/material';
import { GraphData } from '../types/graph';

// Register Cytoscape extensions
cytoscape.use(cola);
cytoscape.use(dagre);
cytoscape.use(coseBilkent);

export type LayoutType = 'force-directed' | 'hierarchical' | 'circular' | 'breadthfirst' | 'cose-bilkent' | 'dagre' | 'cola';

interface GraphVisualizationProps {
  graphData: GraphData;
  layout?: LayoutType;
  onNodeClick?: (nodeId: string, node: any) => void;
  onNodeSelect?: (nodeIds: string[]) => void;
  selectedNodes?: string[];
  height?: number;
}

export function GraphVisualization({
  graphData,
  layout = 'force-directed',
  onNodeClick,
  onNodeSelect,
  selectedNodes = [],
  height = 400
}: GraphVisualizationProps) {
  const cyRef = useRef<cytoscape.Core>();

  const getLayout = (layoutType: LayoutType) => {
    switch (layoutType) {
      case 'force-directed':
        return { name: 'cose' };
      case 'hierarchical':
        return { name: 'dagre', rankDir: 'TB' };
      case 'circular':
        return { name: 'circle' };
      case 'breadthfirst':
        return { name: 'breadthfirst' };
      case 'cose-bilkent':
        return { name: 'cose-bilkent' };
      case 'dagre':
        return { name: 'dagre' };
      case 'cola':
        return { name: 'cola' };
      default:
        return { name: 'cose' };
    }
  };

  const handleNodeClick = (event: any) => {
    const node = event.target;
    if (onNodeClick) {
      onNodeClick(node.id(), node.data());
    }
  };

  const elements = [
    ...graphData.nodes.map(node => ({
      data: { id: node.id, label: node.label || node.id, ...node.properties },
      selected: selectedNodes.includes(node.id)
    })),
    ...graphData.edges.map(edge => ({
      data: { 
        id: edge.id, 
        source: edge.source, 
        target: edge.target, 
        label: edge.label || '',
        ...edge.properties
      }
    }))
  ];

  return (
    <Box sx={{ height, width: '100%' }}>
      <CytoscapeComponent
        elements={elements}
        style={{ width: '100%', height: '100%' }}
        layout={getLayout(layout)}
        stylesheet={[
          {
            selector: 'node',
            style: {
              'background-color': '#007AFF',
              'label': 'data(label)',
              'text-valign': 'center',
              'text-halign': 'center',
              'font-size': '12px',
              'width': '30px',
              'height': '30px',
              'color': '#fff',
              'text-outline-width': 2,
              'text-outline-color': '#007AFF'
            }
          },
          {
            selector: 'node:selected',
            style: {
              'background-color': '#FF3B30',
              'border-width': 3,
              'border-color': '#FF3B30'
            }
          },
          {
            selector: 'edge',
            style: {
              'width': 2,
              'line-color': '#ccc',
              'target-arrow-color': '#ccc',
              'target-arrow-shape': 'triangle',
              'curve-style': 'bezier',
              'label': 'data(label)',
              'font-size': '10px',
              'text-rotation': 'autorotate',
              'text-margin-y': -10
            }
          }
        ]}
        cy={(cy) => {
          cyRef.current = cy;
          cy.on('tap', 'node', handleNodeClick);
          cy.on('select', 'node', () => {
            if (onNodeSelect) {
              const selected = cy.$('node:selected').map((n) => n.id());
              onNodeSelect(selected);
            }
          });
          cy.on('unselect', 'node', () => {
            if (onNodeSelect) {
              const selected = cy.$('node:selected').map((n) => n.id());
              onNodeSelect(selected);
            }
          });
        }}
      />
    </Box>
  );
}
