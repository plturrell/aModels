/**
 * Phase 3.1: GNN Explanation Component
 * 
 * Explains GNN reasoning and predictions
 */

import React from 'react';
import {
  Paper,
  Typography,
  Box,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemText,
  Chip,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import InfoIcon from '@mui/icons-material/Info';
import PsychologyIcon from '@mui/icons-material/Psychology';

export interface GNNExplanationProps {
  predictionType: 'classification' | 'link' | 'anomaly' | 'embedding';
  explanation?: {
    reasoning?: string;
    factors?: Array<{
      factor: string;
      importance: number;
      description?: string;
    }>;
    model_info?: {
      model_type?: string;
      confidence?: number;
      training_data_size?: number;
    };
    graph_features?: {
      node_degree?: number;
      clustering_coefficient?: number;
      centrality?: number;
    };
  };
}

export function GNNExplanation({
  predictionType,
  explanation,
}: GNNExplanationProps) {
  if (!explanation) {
    return (
      <Paper sx={{ p: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
          <PsychologyIcon color="primary" />
          <Typography variant="h6">GNN Explanation</Typography>
        </Box>
        <Typography variant="body2" color="text.secondary">
          No explanation available for this prediction.
        </Typography>
      </Paper>
    );
  }

  return (
    <Paper sx={{ p: 2 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
        <PsychologyIcon color="primary" />
        <Typography variant="h6">GNN Explanation</Typography>
      </Box>

      {/* Reasoning */}
      {explanation.reasoning && (
        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Reasoning
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {explanation.reasoning}
          </Typography>
        </Box>
      )}

      {/* Factors */}
      {explanation.factors && explanation.factors.length > 0 && (
        <Accordion sx={{ mb: 2 }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="subtitle2">
              Contributing Factors ({explanation.factors.length})
            </Typography>
          </AccordionSummary>
          <AccordionDetails>
            <List dense>
              {explanation.factors
                .sort((a, b) => b.importance - a.importance)
                .map((factor, idx) => (
                  <ListItem key={idx}>
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Typography variant="body2">{factor.factor}</Typography>
                          <Chip
                            label={`${(factor.importance * 100).toFixed(0)}%`}
                            size="small"
                            color={factor.importance > 0.7 ? 'primary' : 'default'}
                          />
                        </Box>
                      }
                      secondary={factor.description}
                    />
                  </ListItem>
                ))}
            </List>
          </AccordionDetails>
        </Accordion>
      )}

      {/* Model Info */}
      {explanation.model_info && (
        <Accordion sx={{ mb: 2 }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="subtitle2">Model Information</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              {explanation.model_info.model_type && (
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    Model Type:
                  </Typography>
                  <Chip label={explanation.model_info.model_type} size="small" sx={{ ml: 1 }} />
                </Box>
              )}
              {explanation.model_info.confidence !== undefined && (
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    Model Confidence:
                  </Typography>
                  <Chip
                    label={`${(explanation.model_info.confidence * 100).toFixed(1)}%`}
                    size="small"
                    sx={{ ml: 1 }}
                  />
                </Box>
              )}
              {explanation.model_info.training_data_size && (
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    Training Data Size:
                  </Typography>
                  <Typography variant="body2" component="span" sx={{ ml: 1 }}>
                    {explanation.model_info.training_data_size.toLocaleString()} samples
                  </Typography>
                </Box>
              )}
            </Box>
          </AccordionDetails>
        </Accordion>
      )}

      {/* Graph Features */}
      {explanation.graph_features && (
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="subtitle2">Graph Features</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              {explanation.graph_features.node_degree !== undefined && (
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    Node Degree:
                  </Typography>
                  <Typography variant="body2" component="span" sx={{ ml: 1 }}>
                    {explanation.graph_features.node_degree}
                  </Typography>
                </Box>
              )}
              {explanation.graph_features.clustering_coefficient !== undefined && (
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    Clustering Coefficient:
                  </Typography>
                  <Typography variant="body2" component="span" sx={{ ml: 1 }}>
                    {explanation.graph_features.clustering_coefficient.toFixed(3)}
                  </Typography>
                </Box>
              )}
              {explanation.graph_features.centrality !== undefined && (
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    Centrality:
                  </Typography>
                  <Typography variant="body2" component="span" sx={{ ml: 1 }}>
                    {explanation.graph_features.centrality.toFixed(3)}
                  </Typography>
                </Box>
              )}
            </Box>
          </AccordionDetails>
        </Accordion>
      )}

      {/* Info Note */}
      <Box sx={{ mt: 2, p: 1, bgcolor: 'info.light', borderRadius: 1, display: 'flex', gap: 1 }}>
        <InfoIcon color="info" fontSize="small" />
        <Typography variant="caption" color="text.secondary">
          GNN predictions are based on graph structure and learned patterns. Higher confidence
          indicates stronger patterns in the training data.
        </Typography>
      </Box>
    </Paper>
  );
}

