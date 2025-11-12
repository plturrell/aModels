import React from 'react';
import { Box, Button, Paper, Typography } from '@mui/material';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import RefreshIcon from '@mui/icons-material/Refresh';
import HomeIcon from '@mui/icons-material/Home';

export interface ErrorStateProps {
  title?: string;
  message?: string;
  error?: Error | string;
  onRetry?: () => void;
  onGoHome?: () => void;
  showDetails?: boolean;
  fullScreen?: boolean;
}

/**
 * Error State Component
 * Apple-standard error display with recovery options
 */
export function ErrorState({
  title = 'Something went wrong',
  message = 'We encountered an unexpected error. Please try again.',
  error,
  onRetry,
  onGoHome,
  showDetails = false,
  fullScreen = false,
}: ErrorStateProps) {
  const errorMessage = typeof error === 'string' ? error : error?.message;

  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: fullScreen ? '100vh' : 400,
        p: 3,
      }}
      role="alert"
      aria-live="assertive"
    >
      <Paper
        elevation={3}
        sx={{
          p: 4,
          maxWidth: 600,
          textAlign: 'center',
        }}
      >
        <ErrorOutlineIcon
          sx={{
            fontSize: 64,
            color: 'error.main',
            mb: 2,
          }}
          aria-hidden="true"
        />

        <Typography variant="h5" gutterBottom>
          {title}
        </Typography>

        <Typography variant="body1" color="text.secondary" paragraph>
          {message}
        </Typography>

        {showDetails && errorMessage && (
          <Paper
            variant="outlined"
            sx={{
              p: 2,
              mb: 2,
              textAlign: 'left',
              bgcolor: 'background.default',
              maxHeight: 200,
              overflow: 'auto',
            }}
          >
            <Typography variant="caption" component="pre" sx={{ whiteSpace: 'pre-wrap' }}>
              {errorMessage}
            </Typography>
          </Paper>
        )}

        <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', flexWrap: 'wrap' }}>
          {onRetry && (
            <Button variant="contained" startIcon={<RefreshIcon />} onClick={onRetry}>
              Try Again
            </Button>
          )}

          {onGoHome && (
            <Button variant="outlined" startIcon={<HomeIcon />} onClick={onGoHome}>
              Go Home
            </Button>
          )}

          {!onRetry && !onGoHome && (
            <Button variant="contained" onClick={() => window.location.reload()}>
              Reload Page
            </Button>
          )}
        </Box>
      </Paper>
    </Box>
  );
}

/**
 * Inline Error Message
 * For smaller, inline error states
 */
export function InlineError({
  message,
  onDismiss,
}: {
  message: string;
  onDismiss?: () => void;
}) {
  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 1,
        p: 1,
        bgcolor: 'error.light',
        color: 'error.contrastText',
        borderRadius: 1,
      }}
      role="alert"
    >
      <ErrorOutlineIcon fontSize="small" />
      <Typography variant="body2" sx={{ flex: 1 }}>
        {message}
      </Typography>
      {onDismiss && (
        <Button size="small" onClick={onDismiss}>
          Dismiss
        </Button>
      )}
    </Box>
  );
}

/**
 * Empty State Component
 * For when there's no data to display
 */
export function EmptyState({
  icon,
  title,
  message,
  action,
}: {
  icon?: React.ReactNode;
  title: string;
  message: string;
  action?: { label: string; onClick: () => void };
}) {
  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: 300,
        p: 4,
        textAlign: 'center',
      }}
    >
      {icon && (
        <Box sx={{ fontSize: 64, mb: 2, opacity: 0.5 }} aria-hidden="true">
          {icon}
        </Box>
      )}

      <Typography variant="h6" gutterBottom>
        {title}
      </Typography>

      <Typography variant="body2" color="text.secondary" paragraph>
        {message}
      </Typography>

      {action && (
        <Button variant="contained" onClick={action.onClick}>
          {action.label}
        </Button>
      )}
    </Box>
  );
}
