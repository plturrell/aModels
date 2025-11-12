/**
 * Graph Module Loading Skeleton
 * 
 * Displays while graph data is being loaded
 */

import { Box, Skeleton, Stack, Paper } from '@mui/material';

export function GraphSkeleton() {
  return (
    <Box sx={{ p: 3 }}>
      <Stack spacing={2}>
        {/* Title */}
        <Skeleton variant="text" width="30%" height={40} />
        
        {/* Tabs */}
        <Stack direction="row" spacing={2}>
          {[1, 2, 3, 4, 5].map((i) => (
            <Skeleton key={i} variant="rectangular" width={100} height={36} sx={{ borderRadius: 1 }} />
          ))}
        </Stack>

        {/* Main content area */}
        <Paper sx={{ p: 2 }}>
          <Stack direction="row" spacing={2}>
            {/* Left sidebar */}
            <Box sx={{ width: '20%' }}>
              <Skeleton variant="text" width="80%" height={30} sx={{ mb: 2 }} />
              <Stack spacing={1.5}>
                {[1, 2, 3, 4].map((i) => (
                  <Skeleton key={i} variant="rectangular" height={40} sx={{ borderRadius: 1 }} />
                ))}
              </Stack>
              <Skeleton variant="rectangular" height={48} sx={{ mt: 2, borderRadius: 1 }} />
            </Box>

            {/* Main visualization area */}
            <Box sx={{ flex: 1 }}>
              <Skeleton 
                variant="rectangular" 
                height={500} 
                sx={{ borderRadius: 2 }}
                animation="wave"
              />
              
              {/* Stats below graph */}
              <Stack direction="row" spacing={2} sx={{ mt: 2 }}>
                {[1, 2, 3, 4].map((i) => (
                  <Skeleton 
                    key={i} 
                    variant="rectangular" 
                    width={80} 
                    height={32} 
                    sx={{ borderRadius: 1 }} 
                  />
                ))}
              </Stack>
            </Box>

            {/* Right sidebar */}
            <Box sx={{ width: '25%' }}>
              <Skeleton variant="text" width="80%" height={30} sx={{ mb: 2 }} />
              <Stack spacing={2}>
                {[1, 2, 3].map((i) => (
                  <Paper key={i} variant="outlined" sx={{ p: 2 }}>
                    <Skeleton variant="text" width="60%" height={24} sx={{ mb: 1 }} />
                    <Skeleton variant="text" width="100%" />
                    <Skeleton variant="text" width="80%" />
                  </Paper>
                ))}
              </Stack>
            </Box>
          </Stack>
        </Paper>
      </Stack>
    </Box>
  );
}
