/**
 * Extract Results
 * 
 * View extraction results (entities, metadata)
 */

import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Chip,
  Stack,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { ExtractJob, ExtractResponse } from '../../api/extract';

interface ExtractResultsProps {
  job: ExtractJob;
}

export function ExtractResults({ job }: ExtractResultsProps) {
  const result = job.result as ExtractResponse | undefined;

  if (!result) {
    return (
      <Box sx={{ p: 2 }}>
        <Typography color="textSecondary">No results available</Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Stack spacing={3}>
        {/* Entities Summary */}
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Extracted Entities
          </Typography>
          {result.entities && Object.keys(result.entities).length > 0 ? (
            <Stack direction="row" spacing={1} sx={{ mt: 2 }} flexWrap="wrap">
              {Object.entries(result.entities).map(([type, values]) => (
                <Chip
                  key={type}
                  label={`${type}: ${values.length}`}
                  variant="outlined"
                  sx={{ mb: 1 }}
                />
              ))}
            </Stack>
          ) : (
            <Typography color="textSecondary">No entities extracted</Typography>
          )}
        </Paper>

        {/* Entities Details */}
        {result.entities && Object.keys(result.entities).length > 0 && (
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="subtitle1">Entity Details</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Stack spacing={2}>
                {Object.entries(result.entities).map(([type, values]) => (
                  <Box key={type}>
                    <Typography variant="subtitle2" gutterBottom>
                      {type} ({values.length})
                    </Typography>
                    <Stack direction="row" spacing={1} flexWrap="wrap">
                      {values.map((value, idx) => (
                        <Chip key={idx} label={value} size="small" />
                      ))}
                    </Stack>
                  </Box>
                ))}
              </Stack>
            </AccordionDetails>
          </Accordion>
        )}

        {/* Extractions */}
        {result.extractions && result.extractions.length > 0 && (
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="subtitle1">
                Detailed Extractions ({result.extractions.length})
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Class</TableCell>
                      <TableCell>Text</TableCell>
                      <TableCell>Attributes</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {result.extractions.map((extraction, idx) => (
                      <TableRow key={idx}>
                        <TableCell>{extraction.extraction_class}</TableCell>
                        <TableCell>{extraction.extraction_text}</TableCell>
                        <TableCell>
                          {extraction.attributes ? (
                            <Typography variant="body2" component="pre">
                              {JSON.stringify(extraction.attributes, null, 2)}
                            </Typography>
                          ) : (
                            '-'
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </AccordionDetails>
          </Accordion>
        )}

        {/* Metadata */}
        {job.metadata && (
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="subtitle1">Metadata</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Typography variant="body2" component="pre">
                {JSON.stringify(job.metadata, null, 2)}
              </Typography>
            </AccordionDetails>
          </Accordion>
        )}
      </Stack>
    </Box>
  );
}

