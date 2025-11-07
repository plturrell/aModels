import React from 'react';
import { Box, Typography, Paper, Chip, Stack, Divider, Card, CardContent } from '@mui/material';
import { styled } from '@mui/material/styles';
import CodeIcon from '@mui/icons-material/Code';
import DataObjectIcon from '@mui/icons-material/DataObject';
import TimelineIcon from '@mui/icons-material/Timeline';

interface CanvasProps {
  session: any;
}

const StyledPre = styled('pre')(({ theme }) => ({
  backgroundColor: theme.palette.mode === 'dark' ? '#1e1e1e' : '#f5f5f5',
  padding: theme.spacing(2),
  borderRadius: theme.shape.borderRadius,
  overflow: 'auto',
  fontSize: '0.875rem',
  fontFamily: 'Monaco, "Courier New", monospace',
  maxHeight: '70vh',
  border: `1px solid ${theme.palette.divider}`,
}));

const DataCard = styled(Card)(({ theme }) => ({
  marginBottom: theme.spacing(2),
  '&:last-child': {
    marginBottom: 0,
  },
}));

export function Canvas({ session }: CanvasProps) {
  if (!session) {
    return (
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
          color: 'text.secondary',
        }}
      >
        <CodeIcon sx={{ fontSize: 64, mb: 2, opacity: 0.5 }} />
        <Typography variant="h6" gutterBottom>
          No Active Session
        </Typography>
        <Typography variant="body2">
          Select a session from the sidebar or start a new one with <Chip label="Cmd+K" size="small" /> (Mac) or <Chip label="Ctrl+K" size="small" /> (Windows/Linux)
        </Typography>
      </Box>
    );
  }

  const renderData = (data: any, depth = 0): React.ReactNode => {
    if (data === null || data === undefined) {
      return <Typography component="span" color="text.secondary">null</Typography>;
    }

    if (typeof data === 'string') {
      // Try to detect if it's JSON
      if (data.trim().startsWith('{') || data.trim().startsWith('[')) {
        try {
          const parsed = JSON.parse(data);
          return renderData(parsed, depth);
        } catch {
          return <Typography component="span">{data}</Typography>;
        }
      }
      return <Typography component="span">{data}</Typography>;
    }

    if (typeof data === 'number' || typeof data === 'boolean') {
      return <Typography component="span" color="primary">{String(data)}</Typography>;
    }

    if (Array.isArray(data)) {
      return (
        <Box sx={{ pl: depth > 0 ? 2 : 0 }}>
          {data.map((item, idx) => (
            <Box key={idx} sx={{ mb: 1 }}>
              <Chip label={idx} size="small" sx={{ mr: 1 }} />
              {renderData(item, depth + 1)}
            </Box>
          ))}
        </Box>
      );
    }

    if (typeof data === 'object') {
      const entries = Object.entries(data);
      if (entries.length === 0) {
        return <Typography component="span" color="text.secondary">{} (empty)</Typography>;
      }
      return (
        <Box sx={{ pl: depth > 0 ? 2 : 0 }}>
          {entries.map(([key, value]) => (
            <Box key={key} sx={{ mb: 1.5 }}>
              <Stack direction="row" spacing={1} alignItems="flex-start">
                <Chip
                  label={key}
                  size="small"
                  icon={<DataObjectIcon />}
                  color="primary"
                  variant="outlined"
                />
                <Box sx={{ flex: 1 }}>
                  {renderData(value, depth + 1)}
                </Box>
              </Stack>
            </Box>
          ))}
        </Box>
      );
    }

    return <Typography component="span">{String(data)}</Typography>;
  };

  return (
    <Box sx={{ p: 3, height: '100%', overflow: 'auto' }}>
      <Stack spacing={3}>
        <Box>
          <Stack direction="row" spacing={2} alignItems="center" sx={{ mb: 2 }}>
            <TimelineIcon color="primary" />
            <Typography variant="h5" component="h1">
              {session.command || 'Session'}
            </Typography>
            {session.timestamp && (
              <Chip
                label={new Date(session.timestamp).toLocaleString()}
                size="small"
                variant="outlined"
              />
            )}
          </Stack>
          <Divider />
        </Box>

        <DataCard>
          <CardContent>
            <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <DataObjectIcon fontSize="small" />
              Session Data
            </Typography>
            <Divider sx={{ my: 2 }} />
            {session.data ? (
              <Box>
                {renderData(session.data)}
              </Box>
            ) : (
              <Typography color="text.secondary">No data available</Typography>
            )}
          </CardContent>
        </DataCard>

        {session.data && typeof session.data === 'object' && (
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Raw JSON
            </Typography>
            <StyledPre>{JSON.stringify(session.data, null, 2)}</StyledPre>
          </Paper>
        )}
      </Stack>
    </Box>
  );
}
