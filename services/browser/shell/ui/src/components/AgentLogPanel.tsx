import React from 'react';
import {
  Box,
  Typography,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
  Divider,
  Paper,
  Stack,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import PendingIcon from '@mui/icons-material/Pending';
import ErrorIcon from '@mui/icons-material/Error';
import WarningIcon from '@mui/icons-material/Warning';
import InfoIcon from '@mui/icons-material/Info';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import CodeIcon from '@mui/icons-material/Code';
import HttpIcon from '@mui/icons-material/Http';
import { styled } from '@mui/material/styles';

interface AgentLogPanelProps {
  session: any;
}

interface LogEntry {
  id: string;
  timestamp: Date;
  level: 'success' | 'error' | 'warning' | 'info' | 'pending';
  message: string;
  details?: any;
  type?: 'api' | 'agent' | 'system';
}

const LogItem = styled(ListItem)(({ theme }) => ({
  borderLeft: `3px solid ${theme.palette.divider}`,
  paddingLeft: theme.spacing(2),
  '&:hover': {
    backgroundColor: theme.palette.action.hover,
  },
}));

const getLogIcon = (level: LogEntry['level']) => {
  switch (level) {
    case 'success':
      return <CheckCircleIcon color="success" />;
    case 'error':
      return <ErrorIcon color="error" />;
    case 'warning':
      return <WarningIcon color="warning" />;
    case 'pending':
      return <PendingIcon color="action" />;
    default:
      return <InfoIcon color="info" />;
  }
};

const getLogColor = (level: LogEntry['level']) => {
  switch (level) {
    case 'success':
      return 'success';
    case 'error':
      return 'error';
    case 'warning':
      return 'warning';
    case 'pending':
      return 'default';
    default:
      return 'info';
  }
};

export function AgentLogPanel({ session }: AgentLogPanelProps) {
  const [expandedLog, setExpandedLog] = React.useState<string | false>(false);

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
          p: 2,
        }}
      >
        <InfoIcon sx={{ fontSize: 48, mb: 2, opacity: 0.5 }} />
        <Typography variant="body2" align="center">
          No active session
        </Typography>
        <Typography variant="caption" align="center" sx={{ mt: 1 }}>
          Agent logs will appear here when a session is active
        </Typography>
      </Box>
    );
  }

  // Generate log entries from session data
  const logEntries: LogEntry[] = React.useMemo(() => {
    const entries: LogEntry[] = [];

    // Initial command log
    entries.push({
      id: 'command',
      timestamp: new Date(session.timestamp || Date.now()),
      level: 'info',
      message: `Command executed: ${session.command}`,
      type: 'api',
      details: { command: session.command },
    });

    // Parse session data for additional logs
    if (session.data) {
      if (typeof session.data === 'object') {
        // Check for error indicators
        if (session.data.error) {
          entries.push({
            id: 'error',
            timestamp: new Date(),
            level: 'error',
            message: session.data.error,
            type: 'system',
            details: session.data,
          });
        }

        // Check for success indicators
        if (session.data.status === 'success' || session.data.result) {
          entries.push({
            id: 'success',
            timestamp: new Date(),
            level: 'success',
            message: 'Operation completed successfully',
            type: 'agent',
            details: session.data,
          });
        }

        // Check for API response
        if (session.data.response || session.data.data) {
          entries.push({
            id: 'response',
            timestamp: new Date(),
            level: 'info',
            message: 'Received API response',
            type: 'api',
            details: session.data.response || session.data.data,
          });
        }

        // Check for metadata
        if (session.data.metadata) {
          entries.push({
            id: 'metadata',
            timestamp: new Date(),
            level: 'info',
            message: 'Metadata available',
            type: 'system',
            details: session.data.metadata,
          });
        }
      } else if (typeof session.data === 'string') {
        entries.push({
          id: 'response',
          timestamp: new Date(),
          level: 'info',
          message: 'Response received',
          type: 'api',
          details: { content: session.data },
        });
      }
    }

    return entries;
  }, [session]);

  const handleAccordionChange = (panel: string) => (event: React.SyntheticEvent, isExpanded: boolean) => {
    setExpandedLog(isExpanded ? panel : false);
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Stack direction="row" spacing={1} alignItems="center">
          <CodeIcon color="primary" />
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Agent Log
          </Typography>
          <Chip label={logEntries.length} size="small" color="primary" variant="outlined" />
        </Stack>
      </Box>

      <Box sx={{ flex: 1, overflowY: 'auto', p: 1 }}>
        {logEntries.length === 0 ? (
          <Paper sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="body2" color="text.secondary">
              No log entries available
            </Typography>
          </Paper>
        ) : (
          <List sx={{ p: 0 }}>
            {logEntries.map((entry, index) => (
              <React.Fragment key={entry.id}>
                <Accordion
                  expanded={expandedLog === entry.id}
                  onChange={handleAccordionChange(entry.id)}
                  sx={{ mb: 1, boxShadow: 1 }}
                >
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Stack direction="row" spacing={1} alignItems="center" sx={{ width: '100%', pr: 2 }}>
                      <ListItemIcon sx={{ minWidth: 'auto' }}>
                        {getLogIcon(entry.level)}
                      </ListItemIcon>
                      <Box sx={{ flex: 1, minWidth: 0 }}>
                        <Typography variant="body2" noWrap>
                          {entry.message}
                        </Typography>
                        <Stack direction="row" spacing={1} sx={{ mt: 0.5 }}>
                          <Chip
                            label={entry.level}
                            size="small"
                            color={getLogColor(entry.level) as any}
                            variant="outlined"
                          />
                          {entry.type && (
                            <Chip
                              label={entry.type}
                              size="small"
                              variant="outlined"
                              icon={entry.type === 'api' ? <HttpIcon /> : undefined}
                            />
                          )}
                          <Typography variant="caption" color="text.secondary">
                            {entry.timestamp.toLocaleTimeString()}
                          </Typography>
                        </Stack>
                      </Box>
                    </Stack>
                  </AccordionSummary>
                  <AccordionDetails>
                    {entry.details && (
                      <Paper
                        sx={{
                          p: 2,
                          bgcolor: 'background.default',
                          maxHeight: 300,
                          overflow: 'auto',
                        }}
                      >
                        <Typography variant="caption" color="text.secondary" gutterBottom display="block">
                          Details:
                        </Typography>
                        <pre style={{ margin: 0, fontSize: '0.75rem' }}>
                          {JSON.stringify(entry.details, null, 2)}
                        </pre>
                      </Paper>
                    )}
                  </AccordionDetails>
                </Accordion>
                {index < logEntries.length - 1 && <Divider />}
              </React.Fragment>
            ))}
          </List>
        )}
      </Box>
    </Box>
  );
}
