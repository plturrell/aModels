import React from 'react';
import { Box, Typography } from '@mui/material';

interface CanvasProps {
  session: any;
}

export function Canvas({ session }: CanvasProps) {
  if (!session) {
    return <Typography>Select a session or start a new one with Cmd+K</Typography>;
  }

  return (
    <Box sx={{ p: 4 }}>
      <Typography variant="h4" sx={{ mb: 4 }}>{session.command}</Typography>
      <pre>{JSON.stringify(session.data, null, 2)}</pre>
    </Box>
  );
}
