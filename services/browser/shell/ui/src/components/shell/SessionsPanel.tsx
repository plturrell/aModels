import React from 'react';
import { Box, Typography, List, ListItem, ListItemButton, ListItemText, Button } from '@mui/material';
import AddIcon from '@mui/icons-material/Add';

interface SessionsPanelProps {
  sessions: any[];
  onSelect: (session: any) => void;
}

export function SessionsPanel({ sessions, onSelect }: SessionsPanelProps) {
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <Box sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h6" sx={{ fontWeight: 600 }}>Sessions</Typography>
        <Button variant="contained" size="small" startIcon={<AddIcon />}>New</Button>
      </Box>
      <List sx={{ flex: 1, overflowY: 'auto' }}>
        {sessions.map((session) => (
          <ListItem key={session.id} disablePadding>
            <ListItemButton onClick={() => onSelect(session)}>
              <ListItemText primary={session.command} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
    </Box>
  );
}
