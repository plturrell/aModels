import React from 'react';
import { Box, CircularProgress, Skeleton, Typography } from '@mui/material';

export interface LoadingStateProps {
  variant?: 'spinner' | 'skeleton' | 'pulse';
  size?: 'small' | 'medium' | 'large';
  message?: string;
  fullScreen?: boolean;
}

/**
 * Loading State Component
 * Apple-standard loading indicators with accessibility
 */
export function LoadingState({
  variant = 'spinner',
  size = 'medium',
  message,
  fullScreen = false,
}: LoadingStateProps) {
  const sizeMap = {
    small: 24,
    medium: 40,
    large: 60,
  };

  if (variant === 'skeleton') {
    return (
      <Box sx={{ width: '100%', p: 2 }}>
        <Skeleton variant="text" width="60%" height={40} />
        <Skeleton variant="text" width="80%" />
        <Skeleton variant="text" width="70%" />
        <Skeleton variant="rectangular" height={200} sx={{ mt: 2, borderRadius: 2 }} />
      </Box>
    );
  }

  if (variant === 'pulse') {
    return (
      <Box
        sx={{
          width: '100%',
          height: fullScreen ? '100vh' : 200,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          animation: 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
          '@keyframes pulse': {
            '0%, 100%': { opacity: 1 },
            '50%': { opacity: 0.5 },
          },
        }}
      >
        <Typography color="text.secondary">{message || 'Loading...'}</Typography>
      </Box>
    );
  }

  // Default: spinner
  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 2,
        height: fullScreen ? '100vh' : 'auto',
        p: 4,
      }}
      role="status"
      aria-live="polite"
      aria-label={message || 'Loading'}
    >
      <CircularProgress
        size={sizeMap[size]}
        aria-label="Loading indicator"
      />
      {message && (
        <Typography variant="body2" color="text.secondary">
          {message}
        </Typography>
      )}
    </Box>
  );
}

/**
 * Inline Loading Indicator
 * For smaller, inline loading states
 */
export function InlineLoading({ message }: { message?: string }) {
  return (
    <Box
      sx={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 1,
      }}
      role="status"
      aria-live="polite"
    >
      <CircularProgress size={16} />
      {message && (
        <Typography variant="caption" color="text.secondary">
          {message}
        </Typography>
      )}
    </Box>
  );
}

/**
 * Table Loading Skeleton
 * For data tables and lists
 */
export function TableLoadingSkeleton({ rows = 5 }: { rows?: number }) {
  return (
    <Box sx={{ width: '100%' }}>
      {Array.from({ length: rows }).map((_, index) => (
        <Box
          key={index}
          sx={{
            display: 'flex',
            gap: 2,
            p: 2,
            borderBottom: '1px solid',
            borderColor: 'divider',
          }}
        >
          <Skeleton variant="circular" width={40} height={40} />
          <Box sx={{ flex: 1 }}>
            <Skeleton variant="text" width="40%" />
            <Skeleton variant="text" width="60%" />
          </Box>
          <Skeleton variant="rectangular" width={80} height={32} />
        </Box>
      ))}
    </Box>
  );
}
