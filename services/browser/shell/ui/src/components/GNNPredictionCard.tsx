/**
 * Phase 3.1: GNN Prediction Card Component
 * 
 * Displays GNN predictions in a card format
 */

import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  LinearProgress,
  Tooltip,
} from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import WarningIcon from '@mui/icons-material/Warning';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';

export interface GNNPrediction {
  node_id?: string;
  source_id?: string;
  target_id?: string;
  predicted_class?: string;
  probability: number;
  confidence?: number;
  predicted_label?: string;
  anomaly_score?: number;
  reason?: string;
}

export interface GNNPredictionCardProps {
  prediction: GNNPrediction;
  type: 'classification' | 'link' | 'anomaly';
  onNodeClick?: (nodeId: string) => void;
}

export function GNNPredictionCard({
  prediction,
  type,
  onNodeClick,
}: GNNPredictionCardProps) {
  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'success';
    if (confidence >= 0.5) return 'warning';
    return 'error';
  };

  const getAnomalyColor = (score: number) => {
    if (score >= 0.8) return 'error';
    if (score >= 0.5) return 'warning';
    return 'info';
  };

  const confidence = prediction.confidence ?? prediction.probability;
  const isAnomaly = type === 'anomaly' && (prediction.anomaly_score ?? 0) > 0.5;

  return (
    <Card
      sx={{
        mb: 2,
        border: isAnomaly ? '2px solid' : '1px solid',
        borderColor: isAnomaly ? 'error.main' : 'divider',
        '&:hover': {
          boxShadow: 4,
          cursor: onNodeClick ? 'pointer' : 'default',
        },
      }}
      onClick={() => {
        if (onNodeClick && prediction.node_id) {
          onNodeClick(prediction.node_id);
        }
      }}
    >
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', mb: 2 }}>
          <Box>
            {type === 'classification' && prediction.predicted_class && (
              <>
                <Typography variant="h6" gutterBottom>
                  {prediction.predicted_class}
                </Typography>
                {prediction.node_id && (
                  <Typography variant="caption" color="text.secondary">
                    Node: {prediction.node_id}
                  </Typography>
                )}
              </>
            )}
            {type === 'link' && (
              <>
                <Typography variant="h6" gutterBottom>
                  Link Prediction
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {prediction.source_id} → {prediction.target_id}
                </Typography>
                {prediction.predicted_label && (
                  <Chip
                    label={prediction.predicted_label}
                    size="small"
                    sx={{ mt: 1 }}
                  />
                )}
              </>
            )}
            {type === 'anomaly' && (
              <>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <WarningIcon color="error" />
                  <Typography variant="h6" gutterBottom>
                    Anomaly Detected
                  </Typography>
                </Box>
                {prediction.node_id && (
                  <Typography variant="caption" color="text.secondary">
                    Node: {prediction.node_id}
                  </Typography>
                )}
                {prediction.source_id && prediction.target_id && (
                  <Typography variant="caption" color="text.secondary" display="block">
                    Edge: {prediction.source_id} → {prediction.target_id}
                  </Typography>
                )}
              </>
            )}
          </Box>
          <Box sx={{ textAlign: 'right' }}>
            {type === 'anomaly' ? (
              <Chip
                icon={<WarningIcon />}
                label={`${((prediction.anomaly_score ?? 0) * 100).toFixed(1)}%`}
                color={getAnomalyColor(prediction.anomaly_score ?? 0)}
                size="small"
              />
            ) : (
              <Chip
                icon={confidence >= 0.8 ? <CheckCircleIcon /> : <TrendingUpIcon />}
                label={`${(confidence * 100).toFixed(1)}%`}
                color={getConfidenceColor(confidence)}
                size="small"
              />
            )}
          </Box>
        </Box>

        {/* Confidence/Score Bar */}
        <Box sx={{ mb: 2 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
            <Typography variant="caption" color="text.secondary">
              {type === 'anomaly' ? 'Anomaly Score' : 'Confidence'}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              {type === 'anomaly'
                ? `${((prediction.anomaly_score ?? 0) * 100).toFixed(1)}%`
                : `${(confidence * 100).toFixed(1)}%`}
            </Typography>
          </Box>
          <LinearProgress
            variant="determinate"
            value={(type === 'anomaly' ? prediction.anomaly_score ?? 0 : confidence) * 100}
            color={type === 'anomaly' ? getAnomalyColor(prediction.anomaly_score ?? 0) : getConfidenceColor(confidence)}
            sx={{ height: 8, borderRadius: 1 }}
          />
        </Box>

        {/* Reason/Explanation */}
        {prediction.reason && (
          <Box sx={{ mt: 2, p: 1, bgcolor: 'grey.50', borderRadius: 1 }}>
            <Typography variant="caption" fontWeight="bold" display="block" gutterBottom>
              Explanation:
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {prediction.reason}
            </Typography>
          </Box>
        )}

        {/* Probabilities for classification */}
        {type === 'classification' && 'probabilities' in prediction && prediction.probabilities && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="caption" fontWeight="bold" display="block" gutterBottom>
              Class Probabilities:
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
              {Object.entries(prediction.probabilities)
                .sort(([, a], [, b]) => (b as number) - (a as number))
                .slice(0, 5)
                .map(([className, prob]) => (
                  <Tooltip key={className} title={`${((prob as number) * 100).toFixed(1)}%`}>
                    <Chip
                      label={`${className}: ${((prob as number) * 100).toFixed(0)}%`}
                      size="small"
                      variant="outlined"
                    />
                  </Tooltip>
                ))}
            </Box>
          </Box>
        )}
      </CardContent>
    </Card>
  );
}

