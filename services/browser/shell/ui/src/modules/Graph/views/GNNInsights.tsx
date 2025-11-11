/**
 * Phase 3.1: GNN Insights Dashboard View
 * 
 * Displays GNN predictions, anomalies, and pattern discoveries
 */

import React, { useState, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Tabs,
  Tab,
  Grid,
  Alert,
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Stack,
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import { GNNPredictionCard } from '../../../components/GNNPredictionCard';
import { GNNExplanation } from '../../../components/GNNExplanation';
import {
  getGNNEmbeddings,
  classifyNodes,
  predictLinks,
  getStructuralInsights,
  GNNEmbeddingsRequest,
  GNNClassifyRequest,
  GNNPredictLinksRequest,
  GNNStructuralInsightsRequest,
} from '../../../api/gnn';
import { GraphNode, GraphEdge } from '../../../types/graph';

export interface GNNInsightsProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  projectId?: string;
  onNodeClick?: (nodeId: string) => void;
}

export function GNNInsights({
  nodes,
  edges,
  projectId,
  onNodeClick,
}: GNNInsightsProps) {
  const [activeTab, setActiveTab] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Classification state
  const [classifications, setClassifications] = useState<any[]>([]);
  const [classificationExplanation, setClassificationExplanation] = useState<any>(null);

  // Link prediction state
  const [linkPredictions, setLinkPredictions] = useState<any[]>([]);
  const [topK, setTopK] = useState(10);

  // Anomaly detection state
  const [anomalies, setAnomalies] = useState<any>(null);
  const [anomalyThreshold, setAnomalyThreshold] = useState(0.5);

  // Embeddings state
  const [embeddings, setEmbeddings] = useState<any>(null);
  const [graphLevelEmbedding, setGraphLevelEmbedding] = useState(false);

  const handleClassify = useCallback(async () => {
    if (nodes.length === 0) {
      setError('No nodes available for classification');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const request: GNNClassifyRequest = {
        nodes: nodes.map(n => ({
          id: n.id,
          type: n.type,
          label: n.label,
          properties: n.properties,
        })),
        edges: edges.map(e => ({
          source_id: e.source_id,
          target_id: e.target_id,
          label: e.label,
          type: e.type,
          properties: e.properties,
        })),
      };

      const response = await classifyNodes(request);
      if (response.classifications) {
        setClassifications(response.classifications);
        setClassificationExplanation({
          model_info: {
            model_type: 'GNN Node Classifier',
            confidence: response.classifications.reduce((acc: number, c: any) => acc + c.confidence, 0) / response.classifications.length,
          },
        });
      }
    } catch (err: any) {
      setError(err.message || 'Failed to classify nodes');
      console.error('Classification error:', err);
    } finally {
      setLoading(false);
    }
  }, [nodes, edges]);

  const handlePredictLinks = useCallback(async () => {
    if (nodes.length === 0 || edges.length === 0) {
      setError('Nodes and edges required for link prediction');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const request: GNNPredictLinksRequest = {
        nodes: nodes.map(n => ({
          id: n.id,
          type: n.type,
          label: n.label,
          properties: n.properties,
        })),
        edges: edges.map(e => ({
          source_id: e.source_id,
          target_id: e.target_id,
          label: e.label,
          type: e.type,
          properties: e.properties,
        })),
        top_k: topK,
      };

      const response = await predictLinks(request);
      if (response.predictions) {
        setLinkPredictions(response.predictions);
      }
    } catch (err: any) {
      setError(err.message || 'Failed to predict links');
      console.error('Link prediction error:', err);
    } finally {
      setLoading(false);
    }
  }, [nodes, edges, topK]);

  const handleDetectAnomalies = useCallback(async () => {
    if (nodes.length === 0) {
      setError('No nodes available for anomaly detection');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const request: GNNStructuralInsightsRequest = {
        nodes: nodes.map(n => ({
          id: n.id,
          type: n.type,
          label: n.label,
          properties: n.properties,
        })),
        edges: edges.map(e => ({
          source_id: e.source_id,
          target_id: e.target_id,
          label: e.label,
          type: e.type,
          properties: e.properties,
        })),
        insight_type: 'anomalies',
        threshold: anomalyThreshold,
      };

      const response = await getStructuralInsights(request);
      if (response.insights?.anomalies) {
        setAnomalies(response.insights.anomalies);
      }
    } catch (err: any) {
      setError(err.message || 'Failed to detect anomalies');
      console.error('Anomaly detection error:', err);
    } finally {
      setLoading(false);
    }
  }, [nodes, edges, anomalyThreshold]);

  const handleGetEmbeddings = useCallback(async () => {
    if (nodes.length === 0) {
      setError('No nodes available for embeddings');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const request: GNNEmbeddingsRequest = {
        nodes: nodes.map(n => ({
          id: n.id,
          type: n.type,
          label: n.label,
          properties: n.properties,
        })),
        edges: edges.map(e => ({
          source_id: e.source_id,
          target_id: e.target_id,
          label: e.label,
          type: e.type,
          properties: e.properties,
        })),
        graph_level: graphLevelEmbedding,
      };

      const response = await getGNNEmbeddings(request);
      setEmbeddings(response);
    } catch (err: any) {
      setError(err.message || 'Failed to get embeddings');
      console.error('Embeddings error:', err);
    } finally {
      setLoading(false);
    }
  }, [nodes, edges, graphLevelEmbedding]);

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">GNN Insights</Typography>
        <Typography variant="body2" color="text.secondary">
          {nodes.length} nodes, {edges.length} edges
        </Typography>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Tabs value={activeTab} onChange={(_, v) => setActiveTab(v)} sx={{ mb: 3 }}>
        <Tab label="Classifications" />
        <Tab label="Link Predictions" />
        <Tab label="Anomaly Detection" />
        <Tab label="Embeddings" />
      </Tabs>

      {/* Classifications Tab */}
      {activeTab === 0 && (
        <Grid container spacing={2}>
          <Grid item xs={12} md={3}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Node Classification
              </Typography>
              <Button
                fullWidth
                variant="contained"
                onClick={handleClassify}
                disabled={loading || nodes.length === 0}
                startIcon={loading ? <CircularProgress size={16} /> : <RefreshIcon />}
                sx={{ mb: 2 }}
              >
                Classify Nodes
              </Button>
              {classifications.length > 0 && (
                <Typography variant="caption" color="text.secondary">
                  {classifications.length} nodes classified
                </Typography>
              )}
            </Paper>
          </Grid>
          <Grid item xs={12} md={9}>
            {classifications.length > 0 ? (
              <Box>
                <Typography variant="h6" gutterBottom>
                  Classifications ({classifications.length})
                </Typography>
                <Box sx={{ maxHeight: 500, overflow: 'auto' }}>
                  {classifications.map((classification, idx) => (
                    <GNNPredictionCard
                      key={idx}
                      prediction={classification}
                      type="classification"
                      onNodeClick={onNodeClick}
                    />
                  ))}
                </Box>
                {classificationExplanation && (
                  <Box sx={{ mt: 3 }}>
                    <GNNExplanation
                      predictionType="classification"
                      explanation={classificationExplanation}
                    />
                  </Box>
                )}
              </Box>
            ) : (
              <Paper sx={{ p: 4, textAlign: 'center' }}>
                <Typography variant="body2" color="text.secondary">
                  Click "Classify Nodes" to get GNN-based node classifications
                </Typography>
              </Paper>
            )}
          </Grid>
        </Grid>
      )}

      {/* Link Predictions Tab */}
      {activeTab === 1 && (
        <Grid container spacing={2}>
          <Grid item xs={12} md={3}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Link Prediction
              </Typography>
              <TextField
                fullWidth
                label="Top K"
                type="number"
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value) || 10)}
                sx={{ mb: 2 }}
                size="small"
              />
              <Button
                fullWidth
                variant="contained"
                onClick={handlePredictLinks}
                disabled={loading || nodes.length === 0 || edges.length === 0}
                startIcon={loading ? <CircularProgress size={16} /> : <RefreshIcon />}
              >
                Predict Links
              </Button>
            </Paper>
          </Grid>
          <Grid item xs={12} md={9}>
            {linkPredictions.length > 0 ? (
              <Box>
                <Typography variant="h6" gutterBottom>
                  Predicted Links ({linkPredictions.length})
                </Typography>
                <Box sx={{ maxHeight: 500, overflow: 'auto' }}>
                  {linkPredictions.map((prediction, idx) => (
                    <GNNPredictionCard
                      key={idx}
                      prediction={prediction}
                      type="link"
                      onNodeClick={onNodeClick}
                    />
                  ))}
                </Box>
              </Box>
            ) : (
              <Paper sx={{ p: 4, textAlign: 'center' }}>
                <Typography variant="body2" color="text.secondary">
                  Click "Predict Links" to get GNN-based link predictions
                </Typography>
              </Paper>
            )}
          </Grid>
        </Grid>
      )}

      {/* Anomaly Detection Tab */}
      {activeTab === 2 && (
        <Grid container spacing={2}>
          <Grid item xs={12} md={3}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Anomaly Detection
              </Typography>
              <TextField
                fullWidth
                label="Threshold"
                type="number"
                value={anomalyThreshold}
                onChange={(e) => setAnomalyThreshold(parseFloat(e.target.value) || 0.5)}
                sx={{ mb: 2 }}
                size="small"
                inputProps={{ min: 0, max: 1, step: 0.1 }}
              />
              <Button
                fullWidth
                variant="contained"
                onClick={handleDetectAnomalies}
                disabled={loading || nodes.length === 0}
                startIcon={loading ? <CircularProgress size={16} /> : <RefreshIcon />}
              >
                Detect Anomalies
              </Button>
            </Paper>
          </Grid>
          <Grid item xs={12} md={9}>
            {anomalies ? (
              <Box>
                <Typography variant="h6" gutterBottom>
                  Anomalies Detected ({anomalies.num_anomalies || 0})
                </Typography>
                {anomalies.anomalous_nodes && anomalies.anomalous_nodes.length > 0 && (
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="subtitle1" gutterBottom>
                      Anomalous Nodes ({anomalies.anomalous_nodes.length})
                    </Typography>
                    <Box sx={{ maxHeight: 400, overflow: 'auto' }}>
                      {anomalies.anomalous_nodes.map((anomaly: any, idx: number) => (
                        <GNNPredictionCard
                          key={idx}
                          prediction={anomaly}
                          type="anomaly"
                          onNodeClick={onNodeClick}
                        />
                      ))}
                    </Box>
                  </Box>
                )}
                {anomalies.anomalous_edges && anomalies.anomalous_edges.length > 0 && (
                  <Box>
                    <Typography variant="subtitle1" gutterBottom>
                      Anomalous Edges ({anomalies.anomalous_edges.length})
                    </Typography>
                    <Box sx={{ maxHeight: 400, overflow: 'auto' }}>
                      {anomalies.anomalous_edges.map((anomaly: any, idx: number) => (
                        <GNNPredictionCard
                          key={idx}
                          prediction={anomaly}
                          type="anomaly"
                          onNodeClick={onNodeClick}
                        />
                      ))}
                    </Box>
                  </Box>
                )}
              </Box>
            ) : (
              <Paper sx={{ p: 4, textAlign: 'center' }}>
                <Typography variant="body2" color="text.secondary">
                  Click "Detect Anomalies" to find anomalous nodes and edges
                </Typography>
              </Paper>
            )}
          </Grid>
        </Grid>
      )}

      {/* Embeddings Tab */}
      {activeTab === 3 && (
        <Grid container spacing={2}>
          <Grid item xs={12} md={3}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Graph Embeddings
              </Typography>
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Embedding Type</InputLabel>
                <Select
                  value={graphLevelEmbedding ? 'graph' : 'node'}
                  label="Embedding Type"
                  onChange={(e) => setGraphLevelEmbedding(e.target.value === 'graph')}
                  size="small"
                >
                  <MenuItem value="node">Node-level</MenuItem>
                  <MenuItem value="graph">Graph-level</MenuItem>
                </Select>
              </FormControl>
              <Button
                fullWidth
                variant="contained"
                onClick={handleGetEmbeddings}
                disabled={loading || nodes.length === 0}
                startIcon={loading ? <CircularProgress size={16} /> : <RefreshIcon />}
              >
                Get Embeddings
              </Button>
            </Paper>
          </Grid>
          <Grid item xs={12} md={9}>
            {embeddings ? (
              <Paper sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Embeddings
                </Typography>
                {embeddings.graph_embedding && (
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Graph Embedding
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Dimension: {embeddings.embedding_dim || embeddings.graph_embedding.length}
                    </Typography>
                    <Typography variant="caption" color="text.secondary" component="pre" sx={{ mt: 1, p: 1, bgcolor: 'grey.50', borderRadius: 1, overflow: 'auto' }}>
                      {JSON.stringify(embeddings.graph_embedding.slice(0, 20), null, 2)}
                      {embeddings.graph_embedding.length > 20 && '\n... (truncated)'}
                    </Typography>
                  </Box>
                )}
                {embeddings.node_embeddings && (
                  <Box>
                    <Typography variant="subtitle2" gutterBottom>
                      Node Embeddings ({Object.keys(embeddings.node_embeddings).length} nodes)
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Dimension: {embeddings.embedding_dim || (embeddings.node_embeddings[Object.keys(embeddings.node_embeddings)[0]]?.length || 0)}
                    </Typography>
                  </Box>
                )}
                {embeddings.cached && (
                  <Alert severity="info" sx={{ mt: 2 }}>
                    Results retrieved from cache
                  </Alert>
                )}
              </Paper>
            ) : (
              <Paper sx={{ p: 4, textAlign: 'center' }}>
                <Typography variant="body2" color="text.secondary">
                  Click "Get Embeddings" to generate graph or node embeddings
                </Typography>
              </Paper>
            )}
          </Grid>
        </Grid>
      )}
    </Box>
  );
}

