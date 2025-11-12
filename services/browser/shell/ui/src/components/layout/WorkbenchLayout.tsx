import React from 'react';
import { Box } from '@mui/material';

interface WorkbenchLayoutProps {
  sessions: React.ReactNode;
  canvas: React.ReactNode;
  agentLog: React.ReactNode;
}

export function WorkbenchLayout({ sessions, canvas, agentLog }: WorkbenchLayoutProps) {
  return (
    <Box sx={{ display: 'flex', height: '100vh' }}>
      <Box sx={{ width: '240px', borderRight: '1px solid', borderColor: 'divider', p: 2 }}>
        {sessions}
      </Box>
      <Box sx={{ flex: 1, p: 4 }}>
        {canvas}
      </Box>
      <Box sx={{ width: '320px', borderLeft: '1px solid', borderColor: 'divider', p: 2 }}>
        {agentLog}
      </Box>
    </Box>
  );
}
