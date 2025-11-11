declare module 'react-cytoscapejs' {
  import { Component } from 'react';
  import cytoscape from 'cytoscape';

  export interface CytoscapeComponentProps {
    elements: cytoscape.ElementDefinition[];
    style?: React.CSSProperties;
    stylesheet?: cytoscape.Stylesheet[];
    layout?: cytoscape.LayoutOptions;
    cy?: (cy: cytoscape.Core) => void;
    wheelSensitivity?: number;
    minZoom?: number;
    maxZoom?: number;
  }

  export default class CytoscapeComponent extends Component<CytoscapeComponentProps> {}
}

