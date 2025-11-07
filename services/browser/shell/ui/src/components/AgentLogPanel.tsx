import React from 'react';
import { Box, Typography, List, ListItem, ListItemIcon, ListItemText, Chip } from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import PendingIcon from '@mui/icons-material/Pending';
import ErrorIcon from '@mui/icons-material/Error';

interface AgentLogPanelProps {
  session: any;
}

export function AgentLogPanel({ session }: AgentLogPanelProps) {
  if (!session) {
    return null;
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <Box sx={{ p: 2 }}>
        <Typography variant="h6" sx={{ fontWeight: 600 }}>Agent Log</Typography>
      </Box>
      <List sx={{ flex: 1, overflowY: 'auto' }}>
        <ListItem>
          <ListItemIcon><CheckCircleIcon color="success" /></ListItemIcon>
          <ListItemText primary={`API Call: ${session.command}`} />
        </ListItem>
      </List>
    </Box>
  );
}
