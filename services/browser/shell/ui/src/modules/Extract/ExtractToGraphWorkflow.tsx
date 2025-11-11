/**
 * Extract to Graph Workflow
 * 
 * Unified workflow component connecting extraction results to graph generation
 */

import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Alert,
  Stepper,
  Step,
  StepLabel,
  Stack,
  CircularProgress,
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import { extractEntities, ExtractRequest, ExtractResponse } from '../../api/extract';
import { processKnowledgeGraph, KnowledgeGraphRequest, KnowledgeGraphResponse } from '../../api/extract';
import { WorkflowStatus } from './WorkflowStatus';
import { GraphGenerationPanel } from './GraphGenerationPanel';

interface ExtractToGraphWorkflowProps {
  projectId?: string;
  systemId?: string;
}

type WorkflowStep = 'extract' | 'generate' | 'visualize';

export function ExtractToGraphWorkflow({
  projectId,
  systemId,
}: ExtractToGraphWorkflowProps) {
  const [currentStep, setCurrentStep] = useState<WorkflowStep>('extract');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [extractResult, setExtractResult] = useState<ExtractResponse | null>(null);
  const [graphResult, setGraphResult] = useState<KnowledgeGraphResponse | null>(null);
  const [extractRequest, setExtractRequest] = useState<Partial<ExtractRequest>>({
    document: '',
    prompt_description: 'Extract entities and relationships from the provided text.',
    model_id: 'default',
  });

  const handleExtract = async () => {
    if (!extractRequest.document?.trim()) {
      setError('Please provide document text');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const request: ExtractRequest = {
        document: extractRequest.document,
        prompt_description: extractRequest.prompt_description || 'Extract entities and relationships',
        model_id: extractRequest.model_id || 'default',
      };

      const result = await extractEntities(request);
      setExtractResult(result);
      setCurrentStep('generate');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Extraction failed');
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateGraph = async () => {
    if (!projectId) {
      setError('Project ID is required');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Convert extraction results to knowledge graph request
      const graphRequest: KnowledgeGraphRequest = {
        project_id: projectId,
        system_id: systemId,
        // In a real implementation, we would convert entities to appropriate graph input
        // For now, we'll use the extracted text as input
        json_tables: extractResult
          ? [
              JSON.stringify({
                entities: extractResult.entities,
                extractions: extractResult.extractions,
              }),
            ]
          : [],
      };

      const result = await processKnowledgeGraph(graphRequest);
      setGraphResult(result);
      setCurrentStep('visualize');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Graph generation failed');
    } finally {
      setLoading(false);
    }
  };

  const handleVisualize = () => {
    // Navigate to graph module with the generated graph data
    // In a real implementation, this would pass the graph data to the GraphModule
    window.location.hash = '#graph';
  };

  const steps = [
    { label: 'Extract', step: 'extract' as WorkflowStep },
    { label: 'Generate Graph', step: 'generate' as WorkflowStep },
    { label: 'Visualize', step: 'visualize' as WorkflowStep },
  ];

  const activeStepIndex = steps.findIndex((s) => s.step === currentStep);

  return (
    <Box>
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Extract → Graph → Visualize Workflow
        </Typography>
        <Typography variant="body2" color="textSecondary" sx={{ mb: 3 }}>
          Complete workflow from data extraction to graph visualization
        </Typography>

        <Stepper activeStep={activeStepIndex} sx={{ mb: 4 }}>
          {steps.map((step) => (
            <Step key={step.step}>
              <StepLabel>{step.label}</StepLabel>
            </Step>
          ))}
        </Stepper>

        <WorkflowStatus
          currentStep={currentStep}
          extractResult={extractResult}
          graphResult={graphResult}
        />
      </Paper>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {currentStep === 'extract' && (
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Step 1: Extract Entities
          </Typography>
          <Box sx={{ mt: 2 }}>
            <textarea
              style={{
                width: '100%',
                minHeight: '200px',
                padding: '12px',
                fontFamily: 'monospace',
                fontSize: '14px',
                border: '1px solid #ccc',
                borderRadius: '4px',
              }}
              placeholder="Enter text to extract entities from..."
              value={extractRequest.document || ''}
              onChange={(e) =>
                setExtractRequest((prev) => ({
                  ...prev,
                  document: e.target.value,
                }))
              }
            />
          </Box>
          <Stack direction="row" spacing={2} sx={{ mt: 2 }}>
            <Button
              variant="contained"
              startIcon={loading ? <CircularProgress size={20} /> : <PlayArrowIcon />}
              onClick={handleExtract}
              disabled={loading || !extractRequest.document?.trim()}
            >
              Extract
            </Button>
          </Stack>
        </Paper>
      )}

      {currentStep === 'generate' && extractResult && (
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Step 2: Generate Knowledge Graph
          </Typography>
          <GraphGenerationPanel
            extractResult={extractResult}
            onGenerate={handleGenerateGraph}
            loading={loading}
            graphResult={graphResult}
          />
        </Paper>
      )}

      {currentStep === 'visualize' && graphResult && (
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Step 3: Visualize Graph
          </Typography>
          <Box sx={{ mt: 2 }}>
            <Alert severity="success" sx={{ mb: 2 }}>
              Graph generated successfully! {graphResult.nodes.length} nodes and{' '}
              {graphResult.edges.length} edges created.
            </Alert>
            <Button
              variant="contained"
              onClick={handleVisualize}
              fullWidth
            >
              Open in Graph Module
            </Button>
          </Box>
        </Paper>
      )}
    </Box>
  );
}

