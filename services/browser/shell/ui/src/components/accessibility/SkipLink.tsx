/**
 * Skip Link Component
 * 
 * Allows keyboard users to skip to main content
 * Essential for accessibility
 */

import { Box } from '@mui/material';

interface SkipLinkProps {
  targetId?: string;
}

export function SkipLink({ targetId = 'main-content' }: SkipLinkProps) {
  const handleClick = (e: React.MouseEvent<HTMLAnchorElement>) => {
    e.preventDefault();
    const target = document.getElementById(targetId);
    if (target) {
      target.focus();
      target.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <Box
      component="a"
      href={`#${targetId}`}
      onClick={handleClick}
      sx={{
        position: 'absolute',
        left: '-9999px',
        top: '0',
        zIndex: 9999,
        padding: '8px 16px',
        backgroundColor: 'primary.main',
        color: 'primary.contrastText',
        textDecoration: 'none',
        borderRadius: '0 0 4px 4px',
        fontWeight: 600,
        '&:focus': {
          left: '0',
          outline: '2px solid',
          outlineColor: 'primary.dark',
          outlineOffset: '2px',
        },
      }}
    >
      Skip to main content
    </Box>
  );
}
