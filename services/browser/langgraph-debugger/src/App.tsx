import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Paper,
  Typography,
  Button,
  Grid,
  Card,
  CardContent,
  Chip,
  AppBar,
  Toolbar,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Tabs,
  Tab,
  List,
  ListItem,
  ListItemText,
  Divider
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  SkipNext as StepIcon,
  Refresh as RefreshIcon,
  BugReport as DebugIcon
} from '@mui/icons-material';
import CytoscapeComponent from 'react-cytoscapejs';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import axios from 'axios';

interface GraphNode {
  id: string;
  type: string;
  data: any;
}

interface GraphEdge {
  source: string;
  target: string;
  label?: string;
}

interface ExecutionState {
  node: string;
  inputs: any;
  outputs: any;
  timestamp: Date;
  duration: number;
}

const LANGGRAPH_API = process.env.REACT_APP_LANGGRAPH_API || 'http://localhost:2024';

function App() {
  const [graphData, setGraphData] = useState<{ nodes: GraphNode[]; edges: GraphEdge[] } | null>(null);
  const [execution, setExecution] = useState<ExecutionState[]>([]);
  const [currentNode, setCurrentNode] = useState<string | null>(null);
  const [debugMode, setDebugMode] = useState<'run' | 'step' | 'stopped'>('stopped');
  const [selectedTab, setSelectedTab] = useState(0);
  const [graphs, setGraphs] = useState<string[]>([]);
  const [selectedGraph, setSelectedGraph] = useState<string>('');

  useEffect(() => {
    loadAvailableGraphs();
  }, []);

  const loadAvailableGraphs = async () => {
    try {
      const response = await axios.get(`${LANGGRAPH_API}/graphs`);
      setGraphs(response.data.graphs || []);
      if (response.data.graphs.length > 0) {
        setSelectedGraph(response.data.graphs[0]);
        loadGraph(response.data.graphs[0]);
      }
    } catch (error) {
      console.error('Error loading graphs:', error);
    }
  };

  const loadGraph = async (graphId: string) => {
    try {
      const response = await axios.get(`${LANGGRAPH_API}/graphs/${graphId}`);
      setGraphData(response.data);
    } catch (error) {
      console.error('Error loading graph:', error);
    }
  };

  const runGraph = async () => {
    setDebugMode('run');
    try {
      const response = await axios.post(`${LANGGRAPH_API}/run`, {
        graph_id: selectedGraph,
        input: {},
        debug: true
      });
      
      setExecution(response.data.execution);
    } catch (error) {
      console.error('Error running graph:', error);
    } finally {
      setDebugMode('stopped');
    }
  };

  const stepGraph = async () => {
    setDebugMode('step');
    try {
      const response = await axios.post(`${LANGGRAPH_API}/step`, {
        graph_id: selectedGraph,
        current_node: currentNode
      });
      
      setExecution(prev => [...prev, response.data.state]);
      setCurrentNode(response.data.next_node);
    } catch (error) {
      console.error('Error stepping graph:', error);
    }
  };

  const cytoscapeElements = graphData ? [
    ...graphData.nodes.map(node => ({
      data: { id: node.id, label: node.id, ...node.data }
    })),
    ...graphData.edges.map(edge => ({
      data: { source: edge.source, target: edge.target, label: edge.label || '' }
    }))
  ] : [];

  const cytoscapeStylesheet = [
    {
      selector: 'node',
      style: {
        'background-color': '#667eea',
        'label': 'data(label)',
        'color': '#fff',
        'text-valign': 'center',
        'text-halign': 'center',
        'width': 60,
        'height': 60
      }
    },
    {
      selector: 'node[id="' + currentNode + '"]',
      style: {
        'background-color': '#f59e0b',
        'border-width': 3,
        'border-color': '#dc2626'
      }
    },
    {
      selector: 'edge',
      style: {
        'width': 2,
        'line-color': '#cbd5e1',
        'target-arrow-color': '#cbd5e1',
        'target-arrow-shape': 'triangle',
        'curve-style': 'bezier',
        'label': 'data(label)',
        'font-size': 10,
        'text-rotation': 'autorotate'
      }
    }
  ];

  const performanceData = execution.map((state, idx) => ({
    name: state.node,
    duration: state.duration,
    index: idx
  }));

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100vh', bgcolor: '#f8fafc' }}>
      <AppBar position="static">
        <Toolbar>
          <DebugIcon sx={{ mr: 2 }} />
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            LangGraph Debugger
          </Typography>
          <FormControl sx={{ minWidth: 200, mr: 2 }} size="small">
            <InputLabel sx={{ color: 'white' }}>Graph</InputLabel>
            <Select
              value={selectedGraph}
              onChange={(e) => {
                setSelectedGraph(e.target.value);
                loadGraph(e.target.value);
              }}
              sx={{ color: 'white', '.MuiOutlinedInput-notchedOutline': { borderColor: 'white' } }}
            >
              {graphs.map(graph => (
                <MenuItem key={graph} value={graph}>{graph}</MenuItem>
              ))}
            </Select>
          </FormControl>
          <Button
            variant="contained"
            color="success"
            startIcon={<PlayIcon />}
            onClick={runGraph}
            disabled={debugMode === 'run'}
            sx={{ mr: 1 }}
          >
            Run
          </Button>
          <Button
            variant="contained"
            color="warning"
            startIcon={<StepIcon />}
            onClick={stepGraph}
            disabled={debugMode === 'run'}
            sx={{ mr: 1 }}
          >
            Step
          </Button>
          <Button
            variant="contained"
            color="error"
            startIcon={<StopIcon />}
            onClick={() => setDebugMode('stopped')}
            disabled={debugMode === 'stopped'}
          >
            Stop
          </Button>
        </Toolbar>
      </AppBar>

      <Container maxWidth="xl" sx={{ flexGrow: 1, py: 3 }}>
        <Grid container spacing={3} sx={{ height: '100%' }}>
          {/* Graph Visualization */}
          <Grid item xs={12} md={8}>
            <Paper sx={{ height: '100%', p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Graph Structure
              </Typography>
              <Box sx={{ height: 500, border: 1, borderColor: 'divider', borderRadius: 1 }}>
                {graphData && (
                  <CytoscapeComponent
                    elements={cytoscapeElements}
                    style={{ width: '100%', height: '100%' }}
                    stylesheet={cytoscapeStylesheet}
                    layout={{ name: 'dagre', rankDir: 'TB' }}
                  />
                )}
              </Box>
            </Paper>
          </Grid>

          {/* Execution State */}
          <Grid item xs={12} md={4}>
            <Paper sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
              <Tabs value={selectedTab} onChange={(_, val) => setSelectedTab(val)}>
                <Tab label="Execution" />
                <Tab label="State" />
                <Tab label="Performance" />
              </Tabs>

              {selectedTab === 0 && (
                <Box sx={{ flexGrow: 1, overflow: 'auto', p: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Execution Steps
                  </Typography>
                  <List>
                    {execution.map((state, idx) => (
                      <React.Fragment key={idx}>
                        <ListItem>
                          <ListItemText
                            primary={
                              <Box display="flex" alignItems="center" gap={1}>
                                <Chip label={idx + 1} size="small" />
                                <Typography variant="body2">{state.node}</Typography>
                              </Box>
                            }
                            secondary={
                              <Typography variant="caption" color="text.secondary">
                                {state.duration}ms • {state.timestamp.toLocaleTimeString()}
                              </Typography>
                            }
                          />
                        </ListItem>
                        <Divider />
                      </React.Fragment>
                    ))}
                  </List>
                </Box>
              )}

              {selectedTab === 1 && (
                <Box sx={{ flexGrow: 1, overflow: 'auto', p: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Current State
                  </Typography>
                  {currentNode && (
                    <Card sx={{ mb: 2 }}>
                      <CardContent>
                        <Typography variant="caption" color="text.secondary">
                          Current Node
                        </Typography>
                        <Typography variant="h6">{currentNode}</Typography>
                      </CardContent>
                    </Card>
                  )}
                  {execution.length > 0 && (
                    <Card>
                      <CardContent>
                        <Typography variant="caption" color="text.secondary">
                          Last Output
                        </Typography>
                        <pre style={{ fontSize: 12, overflow: 'auto' }}>
                          {JSON.stringify(execution[execution.length - 1].outputs, null, 2)}
                        </pre>
                      </CardContent>
                    </Card>
                  )}
                </Box>
              )}

              {selectedTab === 2 && (
                <Box sx={{ flexGrow: 1, p: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Performance Metrics
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={performanceData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis label={{ value: 'Duration (ms)', angle: -90, position: 'insideLeft' }} />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="duration" stroke="#8884d8" />
                    </LineChart>
                  </ResponsiveContainer>
                  
                  {execution.length > 0 && (
                    <Card sx={{ mt: 2 }}>
                      <CardContent>
                        <Typography variant="caption" color="text.secondary">
                          Total Execution Time
                        </Typography>
                        <Typography variant="h5">
                          {execution.reduce((sum, state) => sum + state.duration, 0)}ms
                        </Typography>
                      </CardContent>
                    </Card>
                  )}
                </Box>
              )}
            </Paper>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
}

export default App;
