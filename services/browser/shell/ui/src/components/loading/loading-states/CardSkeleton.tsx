/**
 * Card Loading Skeleton
 * 
 * Displays while card content is being loaded
 */

import { Card, CardContent, Skeleton, Stack } from '@mui/material';

interface CardSkeletonProps {
  hasImage?: boolean;
  rows?: number;
}

export function CardSkeleton({ hasImage = false, rows = 3 }: CardSkeletonProps) {
  return (
    <Card>
      {hasImage && (
        <Skeleton variant="rectangular" height={140} animation="wave" />
      )}
      <CardContent>
        <Stack spacing={1.5}>
          {/* Title */}
          <Skeleton variant="text" width="70%" height={28} />
          
          {/* Content rows */}
          {Array.from({ length: rows }).map((_, i) => (
            <Skeleton 
              key={i} 
              variant="text" 
              width={i === rows - 1 ? '50%' : '100%'} 
            />
          ))}
          
          {/* Actions */}
          <Stack direction="row" spacing={1} sx={{ mt: 1 }}>
            <Skeleton variant="rectangular" width={80} height={32} sx={{ borderRadius: 1 }} />
            <Skeleton variant="rectangular" width={80} height={32} sx={{ borderRadius: 1 }} />
          </Stack>
        </Stack>
      </CardContent>
    </Card>
  );
}
