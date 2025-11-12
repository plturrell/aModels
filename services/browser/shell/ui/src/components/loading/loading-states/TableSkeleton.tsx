/**
 * Table Loading Skeleton
 * 
 * Displays while table data is being loaded
 */

import { Box, Skeleton, Stack, Paper } from '@mui/material';

interface TableSkeletonProps {
  rows?: number;
  columns?: number;
  showHeader?: boolean;
}

export function TableSkeleton({ 
  rows = 5, 
  columns = 4,
  showHeader = true 
}: TableSkeletonProps) {
  return (
    <Paper sx={{ p: 2 }}>
      {showHeader && (
        <Stack direction="row" spacing={2} sx={{ mb: 2, pb: 2, borderBottom: 1, borderColor: 'divider' }}>
          {Array.from({ length: columns }).map((_, i) => (
            <Skeleton 
              key={i} 
              variant="text" 
              width={`${100 / columns}%`} 
              height={24}
            />
          ))}
        </Stack>
      )}
      
      <Stack spacing={1.5}>
        {Array.from({ length: rows }).map((_, rowIndex) => (
          <Stack key={rowIndex} direction="row" spacing={2} alignItems="center">
            {Array.from({ length: columns }).map((_, colIndex) => (
              <Skeleton 
                key={colIndex} 
                variant="rectangular" 
                width={`${100 / columns}%`} 
                height={40}
                sx={{ borderRadius: 1 }}
              />
            ))}
          </Stack>
        ))}
      </Stack>
    </Paper>
  );
}
