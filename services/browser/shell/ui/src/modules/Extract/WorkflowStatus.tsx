/**
 * Workflow Status
 * 
 * Status indicator for multi-step workflow
 */

import React from 'react';
import {
  Box,
  Typography,
  Chip,
  Stack,
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import RadioButtonUncheckedIcon from '@mui/icons-material/RadioButtonUnchecked';
import { ExtractResponse } from '../../api/extract';
import { KnowledgeGraphResponse } from '../../api/extract';

interface WorkflowStatusProps {
  currentStep: 'extract' | 'generate' | 'visualize';
  extractResult: ExtractResponse | null;
  graphResult: KnowledgeGraphResponse | null;
}

export function WorkflowStatus({
  currentStep,
  extractResult,
  graphResult,
}: WorkflowStatusProps) {
  const steps = [
    {
      key: 'extract',
      label: 'Extraction',
      completed: !!extractResult,
      current: currentStep === 'extract',
    },
    {
      key: 'generate',
      label: 'Graph Generation',
      completed: !!graphResult,
      current: currentStep === 'generate',
    },
    {
      key: 'visualize',
      label: 'Visualization',
      completed: false,
      current: currentStep === 'visualize',
    },
  ];

  return (
    <Box>
      <Stack direction="row" spacing={2} alignItems="center">
        {steps.map((step, index) => (
          <React.Fragment key={step.key}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              {step.completed ? (
                <CheckCircleIcon color="success" />
              ) : step.current ? (
                <RadioButtonUncheckedIcon color="primary" />
              ) : (
                <RadioButtonUncheckedIcon color="disabled" />
              )}
              <Typography
                variant="body2"
                color={step.completed || step.current ? 'primary' : 'textSecondary'}
              >
                {step.label}
              </Typography>
            </Box>
            {index < steps.length - 1 && (
              <Box
                sx={{
                  width: '40px',
                  height: '2px',
                  bgcolor: step.completed ? 'success.main' : 'divider',
                }}
              />
            )}
          </React.Fragment>
        ))}
      </Stack>
    </Box>
  );
}

