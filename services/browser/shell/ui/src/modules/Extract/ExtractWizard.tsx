/**
 * Extract Wizard
 * 
 * Multi-step wizard for configuring extractions from various sources
 */

import React, { useState } from 'react';
import {
  Box,
  Stepper,
  Step,
  StepLabel,
  Button,
  Paper,
  Typography,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  CircularProgress,
  Stack,
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import { extractEntities, ExtractRequest } from '../../api/extract';
import { SAPExtractWizard } from './SAPExtractWizard';

interface ExtractWizardProps {
  projectId?: string;
  systemId?: string;
  onExtractionComplete?: (jobId: string) => void;
}

type SourceType = 'text' | 'documents' | 'sap_bdc' | 'ocr' | 'schema';

const steps = ['Select Source', 'Configure', 'Parameters', 'Review'];

export function ExtractWizard({
  projectId,
  systemId,
  onExtractionComplete,
}: ExtractWizardProps) {
  const [activeStep, setActiveStep] = useState(0);
  const [sourceType, setSourceType] = useState<SourceType>('text');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [extractRequest, setExtractRequest] = useState<Partial<ExtractRequest>>({
    prompt_description: 'Extract entities and relationships from the provided text.',
    model_id: 'default',
  });

  // Source-specific state
  const [documentText, setDocumentText] = useState('');
  const [documents, setDocuments] = useState<string[]>(['']);
  const [projectIdInput, setProjectIdInput] = useState(projectId || '');
  const [systemIdInput, setSystemIdInput] = useState(systemId || '');

  const handleNext = () => {
    if (activeStep === steps.length - 1) {
      handleSubmit();
    } else {
      setActiveStep((prevActiveStep) => prevActiveStep + 1);
    }
  };

  const handleBack = () => {
    setActiveStep((prevActiveStep) => prevActiveStep - 1);
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);

    try {
      const request: ExtractRequest = {
        ...extractRequest,
        prompt_description: extractRequest.prompt_description || 'Extract entities and relationships',
        model_id: extractRequest.model_id || 'default',
      };

      if (sourceType === 'text') {
        request.document = documentText;
      } else if (sourceType === 'documents') {
        request.documents = documents.filter((d) => d.trim() !== '');
      }

      const response = await extractEntities(request);
      
      // Generate a job ID (in a real implementation, this would come from the server)
      const jobId = `job_${Date.now()}`;
      
      if (onExtractionComplete) {
        onExtractionComplete(jobId);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Extraction failed');
    } finally {
      setLoading(false);
    }
  };

  const renderStepContent = () => {
    switch (activeStep) {
      case 0:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Select Extraction Source
            </Typography>
            <FormControl fullWidth sx={{ mt: 2 }}>
              <InputLabel>Source Type</InputLabel>
              <Select
                value={sourceType}
                label="Source Type"
                onChange={(e) => setSourceType(e.target.value as SourceType)}
              >
                <MenuItem value="text">Text Input</MenuItem>
                <MenuItem value="documents">Multiple Documents</MenuItem>
                <MenuItem value="sap_bdc">SAP Business Data Cloud</MenuItem>
                <MenuItem value="ocr">OCR (Image to Text)</MenuItem>
                <MenuItem value="schema">Schema Replication</MenuItem>
              </Select>
            </FormControl>
          </Box>
        );

      case 1:
        if (sourceType === 'sap_bdc') {
          return (
            <SAPExtractWizard
              projectId={projectIdInput}
              systemId={systemIdInput}
              onConfigChange={(config) => {
                // Store SAP config for later use
                setExtractRequest((prev) => ({
                  ...prev,
                  ...config,
                }));
              }}
            />
          );
        }

        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Configure {sourceType === 'text' ? 'Text' : 'Documents'}
            </Typography>
            {sourceType === 'text' ? (
              <TextField
                fullWidth
                multiline
                rows={10}
                label="Document Text"
                value={documentText}
                onChange={(e) => setDocumentText(e.target.value)}
                sx={{ mt: 2 }}
              />
            ) : (
              <Box sx={{ mt: 2 }}>
                {documents.map((doc, index) => (
                  <TextField
                    key={index}
                    fullWidth
                    multiline
                    rows={4}
                    label={`Document ${index + 1}`}
                    value={doc}
                    onChange={(e) => {
                      const newDocs = [...documents];
                      newDocs[index] = e.target.value;
                      setDocuments(newDocs);
                    }}
                    sx={{ mb: 2 }}
                  />
                ))}
                <Button
                  variant="outlined"
                  onClick={() => setDocuments([...documents, ''])}
                >
                  Add Document
                </Button>
              </Box>
            )}
          </Box>
        );

      case 2:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Extraction Parameters
            </Typography>
            <TextField
              fullWidth
              label="Project ID"
              value={projectIdInput}
              onChange={(e) => setProjectIdInput(e.target.value)}
              sx={{ mt: 2 }}
            />
            <TextField
              fullWidth
              label="System ID (optional)"
              value={systemIdInput}
              onChange={(e) => setSystemIdInput(e.target.value)}
              sx={{ mt: 2 }}
            />
            <TextField
              fullWidth
              multiline
              rows={3}
              label="Prompt Description"
              value={extractRequest.prompt_description || ''}
              onChange={(e) =>
                setExtractRequest((prev) => ({
                  ...prev,
                  prompt_description: e.target.value,
                }))
              }
              sx={{ mt: 2 }}
            />
            <TextField
              fullWidth
              label="Model ID"
              value={extractRequest.model_id || ''}
              onChange={(e) =>
                setExtractRequest((prev) => ({
                  ...prev,
                  model_id: e.target.value,
                }))
              }
              sx={{ mt: 2 }}
            />
          </Box>
        );

      case 3:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Review Configuration
            </Typography>
            <Paper sx={{ p: 2, mt: 2 }}>
              <Typography variant="body2" color="textSecondary">
                Source Type: {sourceType}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Project ID: {projectIdInput || 'Not set'}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                System ID: {systemIdInput || 'Not set'}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Model: {extractRequest.model_id || 'default'}
              </Typography>
            </Paper>
          </Box>
        );

      default:
        return null;
    }
  };

  return (
    <Box>
      <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
        {steps.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Paper sx={{ p: 3, mb: 3 }}>{renderStepContent()}</Paper>

      <Stack direction="row" spacing={2} justifyContent="flex-end">
        <Button
          disabled={activeStep === 0 || loading}
          onClick={handleBack}
          startIcon={<ArrowBackIcon />}
        >
          Back
        </Button>
        <Button
          variant="contained"
          onClick={handleNext}
          disabled={loading}
          endIcon={loading ? <CircularProgress size={20} /> : <ArrowForwardIcon />}
        >
          {activeStep === steps.length - 1 ? 'Submit' : 'Next'}
        </Button>
      </Stack>
    </Box>
  );
}

