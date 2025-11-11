/**
 * Graph Generation Panel
 * 
 * Panel showing graph generation progress and statistics
 */

import React from 'react';
import {
  Box,
  Typography,
  Button,
  CircularProgress,
  LinearProgress,
  Stack,
  Card,
  CardContent,
} from '@mui/material';
import { GridLegacy as Grid } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import { ExtractResponse, KnowledgeGraphResponse } from '../../api/extract';

interface GraphGenerationPanelProps {
  extractResult: ExtractResponse;
  onGenerate: () => void;
  loading: boolean;
  graphResult: KnowledgeGraphResponse | null;
}

export function GraphGenerationPanel({
  extractResult,
  onGenerate,
  loading,
  graphResult,
}: GraphGenerationPanelProps) {
  const entityCount = extractResult.entities
    ? Object.values(extractResult.entities).reduce((sum, arr) => sum + arr.length, 0)
    : 0;
  const extractionCount = extractResult.extractions?.length || 0;

  return (
    <Box>
      <Stack spacing={3}>
        {/* Extraction Summary */}
        <Card>
          <CardContent>
            <Typography variant="subtitle1" gutterBottom>
              Extraction Summary
            </Typography>
            <Grid container spacing={2} sx={{ mt: 1 }}>
              <Grid item xs={6}>
                <Typography variant="body2" color="textSecondary">
                  Entity Types
                </Typography>
                <Typography variant="h6">
                  {extractResult.entities ? Object.keys(extractResult.entities).length : 0}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2" color="textSecondary">
                  Total Entities
                </Typography>
                <Typography variant="h6">{entityCount}</Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2" color="textSecondary">
                  Detailed Extractions
                </Typography>
                <Typography variant="h6">{extractionCount}</Typography>
              </Grid>
            </Grid>
          </CardContent>
        </Card>

        {/* Generation Status */}
        {loading && (
          <Box>
            <Typography variant="body2" color="textSecondary" gutterBottom>
              Generating knowledge graph...
            </Typography>
            <LinearProgress sx={{ mt: 1 }} />
          </Box>
        )}

        {graphResult && (
          <Card>
            <CardContent>
              <Typography variant="subtitle1" gutterBottom>
                Graph Generated Successfully
              </Typography>
              <Grid container spacing={2} sx={{ mt: 1 }}>
                <Grid item xs={6}>
                  <Typography variant="body2" color="textSecondary">
                    Nodes
                  </Typography>
                  <Typography variant="h6">{graphResult.nodes.length}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="textSecondary">
                    Edges
                  </Typography>
                  <Typography variant="h6">{graphResult.edges.length}</Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        )}

        {/* Generate Button */}
        {!graphResult && (
          <Button
            variant="contained"
            startIcon={loading ? <CircularProgress size={20} /> : <PlayArrowIcon />}
            onClick={onGenerate}
            disabled={loading}
            fullWidth
          >
            Generate Knowledge Graph
          </Button>
        )}
      </Stack>
    </Box>
  );
}

