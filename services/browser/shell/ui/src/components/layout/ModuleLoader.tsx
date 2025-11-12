/**
 * Module Loading Fallback
 * 
 * Displays while lazy-loaded modules are being fetched
 */

import { Box, CircularProgress, Typography, alpha } from '@mui/material';

export function ModuleLoader() {
  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        height: '100%',
        minHeight: '400px',
        gap: 2,
      }}
    >
      <CircularProgress size={48} thickness={4} />
      <Typography 
        variant="body2" 
        color="text.secondary"
        sx={{ 
          mt: 2,
          animation: 'pulse 1.5s ease-in-out infinite',
          '@keyframes pulse': {
            '0%, 100%': { opacity: 1 },
            '50%': { opacity: 0.5 },
          },
        }}
      >
        Loading module...
      </Typography>
    </Box>
  );
}
