/**
 * Visually Hidden Component
 * 
 * Hides content visually but keeps it available to screen readers
 * Common pattern for accessibility
 */

import { Box } from '@mui/material';

interface VisuallyHiddenProps {
  children: React.ReactNode;
  component?: React.ElementType;
}

export function VisuallyHidden({ children, component = 'span' }: VisuallyHiddenProps) {
  return (
    <Box
      component={component}
      sx={{
        position: 'absolute',
        width: '1px',
        height: '1px',
        padding: 0,
        margin: '-1px',
        overflow: 'hidden',
        clip: 'rect(0, 0, 0, 0)',
        whiteSpace: 'nowrap',
        border: 0,
      }}
    >
      {children}
    </Box>
  );
}

/**
 * Usage:
 * 
 * <button>
 *   <DeleteIcon />
 *   <VisuallyHidden>Delete item</VisuallyHidden>
 * </button>
 * 
 * Screen readers will announce "Delete item" even though it's not visible
 */
