/**
 * Narrative Insights View
 * 
 * Main component for narrative GNN features including explanations, predictions,
 * anomaly detection, MCTS what-if analysis, and storyline exploration.
 */

import React, { useState, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  Tabs,
  Tab,
  Button,
  TextField,
  Alert,
  CircularProgress,
  Chip,
  Stack,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  Divider,
  Card,
  CardContent,
} from '@mui/material';
import { GridLegacy as Grid } from '@mui/material';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import TimelineIcon from '@mui/icons-material/Timeline';
import WarningIcon from '@mui/icons-material/Warning';
import PsychologyIcon from '@mui/icons-material/Psychology';
import BookIcon from '@mui/icons-material/Book';
import { GraphNode, GraphEdge } from '../../../types/graph';
import {
  explainNarrative,
  predictNarrative,
  detectNarrativeAnomalies,
  narrativeMCTS,
  narrativeStoryline,
} from '../../../api/narrative';

export interface NarrativeInsightsProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  projectId?: string;
  systemId?: string;
  onNodeClick?: (nodeId: string) => void;
}

export function NarrativeInsights({
  nodes,
  edges,
  projectId,
  systemId,
  onNodeClick,
}: NarrativeInsightsProps) {
  const [activeTab, setActiveTab] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Explanation state
  const [explanation, setExplanation] = useState<any>(null);
  const [focusNodeId, setFocusNodeId] = useState('');
  const [storylineId, setStorylineId] = useState('');

  // Prediction state
  const [prediction, setPrediction] = useState<any>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [futureTime, setFutureTime] = useState(10);

  // Anomaly state
  const [anomalies, setAnomalies] = useState<any>(null);
  const [anomalyThreshold, setAnomalyThreshold] = useState(0.5);

  // MCTS state
  const [mctsResult, setMctsResult] = useState<any>(null);
  const [mctsRollouts, setMctsRollouts] = useState(100);

  // Storyline state
  const [storylines, setStorylines] = useState<any>(null);
  const [selectedStoryline, setSelectedStoryline] = useState<string>('');

  const handleExplain = useCallback(async () => {
    if (nodes.length === 0) {
      setError('No nodes available for explanation');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await explainNarrative({
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
        storyline_id: storylineId || undefined,
        focus_node_id: focusNodeId || undefined,
      });

      setExplanation(result);
    } catch (err: any) {
      setError(err.message || 'Failed to generate explanation');
      console.error('Explanation error:', err);
    } finally {
      setLoading(false);
    }
  }, [nodes, edges, storylineId, focusNodeId]);

  const handlePredict = useCallback(async () => {
    if (nodes.length === 0) {
      setError('No nodes available for prediction');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await predictNarrative({
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
        storyline_id: storylineId || undefined,
        current_time: currentTime,
        future_time: futureTime,
        num_trajectories: 5,
      });

      setPrediction(result);
    } catch (err: any) {
      setError(err.message || 'Failed to generate prediction');
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  }, [nodes, edges, storylineId, currentTime, futureTime]);

  const handleDetectAnomalies = useCallback(async () => {
    if (nodes.length === 0) {
      setError('No nodes available for anomaly detection');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await detectNarrativeAnomalies({
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
        storyline_id: storylineId || undefined,
        threshold: anomalyThreshold,
      });

      setAnomalies(result);
    } catch (err: any) {
      setError(err.message || 'Failed to detect anomalies');
      console.error('Anomaly detection error:', err);
    } finally {
      setLoading(false);
    }
  }, [nodes, edges, storylineId, anomalyThreshold]);

  const handleMCTS = useCallback(async () => {
    if (nodes.length === 0) {
      setError('No nodes available for MCTS analysis');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await narrativeMCTS({
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
        storyline_id: storylineId || undefined,
        current_time: currentTime,
        num_rollouts: mctsRollouts,
        max_depth: 10,
        exploration_c: 1.414,
      });

      setMctsResult(result);
    } catch (err: any) {
      setError(err.message || 'Failed to run MCTS analysis');
      console.error('MCTS error:', err);
    } finally {
      setLoading(false);
    }
  }, [nodes, edges, storylineId, currentTime, mctsRollouts]);

  const handleLoadStorylines = useCallback(async () => {
    if (nodes.length === 0) {
      setError('No nodes available');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await narrativeStoryline({
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
        operation: 'list',
      });

      setStorylines(result);
    } catch (err: any) {
      setError(err.message || 'Failed to load storylines');
      console.error('Storyline error:', err);
    } finally {
      setLoading(false);
    }
  }, [nodes, edges]);

  return (
    <Box>
      <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <AutoAwesomeIcon /> Narrative Intelligence
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Explore graph dynamics through narrative explanations, predictions, and what-if analysis
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Common Controls */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Storyline ID (optional)"
              value={storylineId}
              onChange={(e) => setStorylineId(e.target.value)}
              size="small"
              placeholder="e.g., merger_story"
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Focus Node ID (optional)"
              value={focusNodeId}
              onChange={(e) => setFocusNodeId(e.target.value)}
              size="small"
              placeholder="Node ID to focus on"
            />
          </Grid>
        </Grid>
      </Paper>

      <Tabs value={activeTab} onChange={(_, v) => setActiveTab(v)} sx={{ mb: 2 }}>
        <Tab icon={<AutoAwesomeIcon />} label="Explanations" />
        <Tab icon={<TimelineIcon />} label="Predictions" />
        <Tab icon={<WarningIcon />} label="Anomalies" />
        <Tab icon={<PsychologyIcon />} label="What-If (MCTS)" />
        <Tab icon={<BookIcon />} label="Storylines" />
      </Tabs>

      {/* Explanations Tab */}
      {activeTab === 0 && (
        <Paper sx={{ p: 2 }}>
          <Box sx={{ mb: 2 }}>
            <Button
              variant="contained"
              onClick={handleExplain}
              disabled={loading || nodes.length === 0}
              startIcon={loading ? <CircularProgress size={16} /> : <AutoAwesomeIcon />}
            >
              Generate Explanation
            </Button>
          </Box>

          {explanation && (
            <Box>
              <Typography variant="h6" gutterBottom>
                Explanation
              </Typography>
              <Card sx={{ mb: 2 }}>
                <CardContent>
                  <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
                    {explanation.explanation || 'No explanation generated'}
                  </Typography>
                </CardContent>
              </Card>

              {explanation.key_actors && explanation.key_actors.length > 0 && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    Key Actors
                  </Typography>
                  <Stack direction="row" spacing={1} sx={{ flexWrap: 'wrap', gap: 1 }}>
                    {explanation.key_actors.map((actor: any, idx: number) => (
                      <Chip
                        key={idx}
                        label={actor.node_id}
                        onClick={() => onNodeClick?.(actor.node_id)}
                        sx={{ cursor: 'pointer' }}
                      />
                    ))}
                  </Stack>
                </Box>
              )}

              {explanation.turning_points && explanation.turning_points.length > 0 && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    Turning Points
                  </Typography>
                  <List>
                    {explanation.turning_points.map((point: any, idx: number) => (
                      <ListItem key={idx}>
                        <ListItemText
                          primary={point.node_id}
                          secondary={`Significance: ${point.significance?.toFixed(2) || 'N/A'}`}
                        />
                      </ListItem>
                    ))}
                  </List>
                </Box>
              )}
            </Box>
          )}
        </Paper>
      )}

      {/* Predictions Tab */}
      {activeTab === 1 && (
        <Paper sx={{ p: 2 }}>
          <Grid container spacing={2} sx={{ mb: 2 }}>
            <Grid item xs={12} sm={4}>
              <TextField
                fullWidth
                label="Current Time"
                type="number"
                value={currentTime}
                onChange={(e) => setCurrentTime(parseFloat(e.target.value) || 0)}
                size="small"
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                fullWidth
                label="Future Time"
                type="number"
                value={futureTime}
                onChange={(e) => setFutureTime(parseFloat(e.target.value) || 10)}
                size="small"
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <Button
                fullWidth
                variant="contained"
                onClick={handlePredict}
                disabled={loading || nodes.length === 0}
                startIcon={loading ? <CircularProgress size={16} /> : <TimelineIcon />}
              >
                Predict Future
              </Button>
            </Grid>
          </Grid>

          {prediction && (
            <Box>
              <Typography variant="h6" gutterBottom>
                Predictions
              </Typography>
              {prediction.trajectories && prediction.trajectories.length > 0 ? (
                <List>
                  {prediction.trajectories.map((trajectory: any, idx: number) => (
                    <Card key={idx} sx={{ mb: 2 }}>
                      <CardContent>
                        <Typography variant="subtitle1">
                          Trajectory {idx + 1} (Score: {prediction.scores?.[idx]?.toFixed(2) || 'N/A'})
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {JSON.stringify(trajectory, null, 2)}
                        </Typography>
                      </CardContent>
                    </Card>
                  ))}
                </List>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No trajectories generated
                </Typography>
              )}
            </Box>
          )}
        </Paper>
      )}

      {/* Anomalies Tab */}
      {activeTab === 2 && (
        <Paper sx={{ p: 2 }}>
          <Grid container spacing={2} sx={{ mb: 2 }}>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Anomaly Threshold"
                type="number"
                value={anomalyThreshold}
                onChange={(e) => setAnomalyThreshold(parseFloat(e.target.value) || 0.5)}
                size="small"
                inputProps={{ min: 0, max: 1, step: 0.1 }}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <Button
                fullWidth
                variant="contained"
                color="warning"
                onClick={handleDetectAnomalies}
                disabled={loading || nodes.length === 0}
                startIcon={loading ? <CircularProgress size={16} /> : <WarningIcon />}
              >
                Detect Anomalies
              </Button>
            </Grid>
          </Grid>

          {anomalies && (
            <Box>
              <Typography variant="h6" gutterBottom>
                Detected Anomalies
              </Typography>
              {anomalies.anomalies && anomalies.anomalies.length > 0 ? (
                <List>
                  {anomalies.anomalies.map((anomaly: any, idx: number) => (
                    <Card key={idx} sx={{ mb: 1, bgcolor: 'error.light' }}>
                      <CardContent>
                        <Typography variant="body2">
                          {JSON.stringify(anomaly, null, 2)}
                        </Typography>
                      </CardContent>
                    </Card>
                  ))}
                </List>
              ) : (
                <Alert severity="success">No anomalies detected</Alert>
              )}

              {anomalies.violations && anomalies.violations.length > 0 && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    Violations
                  </Typography>
                  <List>
                    {anomalies.violations.map((violation: any, idx: number) => (
                      <ListItem key={idx}>
                        <ListItemText
                          primary={violation.type || 'Unknown violation'}
                          secondary={JSON.stringify(violation)}
                        />
                      </ListItem>
                    ))}
                  </List>
                </Box>
              )}
            </Box>
          )}
        </Paper>
      )}

      {/* MCTS What-If Tab */}
      {activeTab === 3 && (
        <Paper sx={{ p: 2 }}>
          <Grid container spacing={2} sx={{ mb: 2 }}>
            <Grid item xs={12} sm={4}>
              <TextField
                fullWidth
                label="Current Time"
                type="number"
                value={currentTime}
                onChange={(e) => setCurrentTime(parseFloat(e.target.value) || 0)}
                size="small"
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                fullWidth
                label="MCTS Rollouts"
                type="number"
                value={mctsRollouts}
                onChange={(e) => setMctsRollouts(parseInt(e.target.value) || 100)}
                size="small"
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <Button
                fullWidth
                variant="contained"
                onClick={handleMCTS}
                disabled={loading || nodes.length === 0}
                startIcon={loading ? <CircularProgress size={16} /> : <PsychologyIcon />}
              >
                Run What-If Analysis
              </Button>
            </Grid>
          </Grid>

          {mctsResult && (
            <Box>
              <Typography variant="h6" gutterBottom>
                MCTS Analysis Results
              </Typography>
              <Card sx={{ mb: 2 }}>
                <CardContent>
                  <Typography variant="body2">
                    <strong>Best Path Value:</strong> {mctsResult.path_value?.toFixed(2) || 'N/A'}
                  </Typography>
                  <Typography variant="body2">
                    <strong>Rollouts:</strong> {mctsResult.rollouts || 0}
                  </Typography>
                </CardContent>
              </Card>

              {mctsResult.best_path && mctsResult.best_path.length > 0 && (
                <Box>
                  <Typography variant="subtitle1" gutterBottom>
                    Best Path
                  </Typography>
                  <List>
                    {mctsResult.best_path.map((step: any, idx: number) => (
                      <ListItem key={idx}>
                        <ListItemText
                          primary={`Step ${idx + 1}`}
                          secondary={JSON.stringify(step)}
                        />
                      </ListItem>
                    ))}
                  </List>
                </Box>
              )}
            </Box>
          )}
        </Paper>
      )}

      {/* Storylines Tab */}
      {activeTab === 4 && (
        <Paper sx={{ p: 2 }}>
          <Box sx={{ mb: 2 }}>
            <Button
              variant="contained"
              onClick={handleLoadStorylines}
              disabled={loading || nodes.length === 0}
              startIcon={loading ? <CircularProgress size={16} /> : <BookIcon />}
            >
              Load Storylines
            </Button>
          </Box>

          {storylines && (
            <Box>
              <Typography variant="h6" gutterBottom>
                Storylines
              </Typography>
              {storylines.storylines && Array.isArray(storylines.storylines) && storylines.storylines.length > 0 ? (
                <List>
                  {storylines.storylines.map((sid: string) => (
                    <ListItemButton
                      key={sid}
                      onClick={() => setSelectedStoryline(sid)}
                      selected={selectedStoryline === sid}
                    >
                      <ListItemText primary={sid} />
                    </ListItemButton>
                  ))}
                </List>
              ) : storylines.storylines && typeof storylines.storylines === 'object' ? (
                <List>
                  {Object.entries(storylines.storylines).map(([sid, story]: [string, any]) => (
                    <Card key={sid} sx={{ mb: 1 }}>
                      <CardContent>
                        <Typography variant="subtitle1">{sid}</Typography>
                        <Typography variant="body2" color="text.secondary">
                          Theme: {story.theme || 'N/A'}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Type: {story.narrative_type || 'N/A'}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Nodes: {story.nodes?.length || 0}
                        </Typography>
                      </CardContent>
                    </Card>
                  ))}
                </List>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No storylines found
                </Typography>
              )}
            </Box>
          )}
        </Paper>
      )}
    </Box>
  );
}

