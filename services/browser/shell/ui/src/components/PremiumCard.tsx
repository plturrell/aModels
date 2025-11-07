/**
 * Premium Card Component - Apple-style elevated card
 * Glassmorphism, depth, beautiful shadows
 */

import React from 'react';
import { Card, CardProps, alpha } from '@mui/material';

interface PremiumCardProps extends CardProps {
  glass?: boolean;
  hoverable?: boolean;
  interactive?: boolean;
}

export function PremiumCard({ 
  glass = false, 
  hoverable = true, 
  interactive = false,
  children, 
  sx = {},
  ...props 
}: PremiumCardProps) {
  return (
    <Card
      {...props}
      sx={{
        borderRadius: 4,
        border: '1px solid',
        borderColor: alpha('#000', 0.06),
        background: glass 
          ? `linear-gradient(135deg, 
              ${alpha('#fff', 0.9)} 0%, 
              ${alpha('#fff', 0.7)} 100%)`
          : '#fff',
        backdropFilter: glass ? 'blur(20px)' : 'none',
        boxShadow: '0px 4px 20px rgba(0, 0, 0, 0.06)',
        transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
        position: 'relative',
        overflow: 'hidden',
        cursor: interactive ? 'pointer' : 'default',
        
        // Subtle inner glow
        '&::before': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: '1px',
          background: `linear-gradient(90deg, 
            transparent 0%, 
            ${alpha('#fff', 0.8)} 50%, 
            transparent 100%)`,
          opacity: 0.6,
        },
        
        ...(hoverable && {
          '&:hover': {
            transform: 'translateY(-4px) scale(1.01)',
            boxShadow: '0px 12px 40px rgba(0, 0, 0, 0.12)',
            borderColor: alpha('#007AFF', 0.2),
            
            '&::after': {
              opacity: 1,
            },
          },
        }),
        
        // Gradient overlay on hover
        '&::after': hoverable ? {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: `radial-gradient(circle at 50% 0%, 
            ${alpha('#007AFF', 0.05)} 0%, 
            transparent 60%)`,
          opacity: 0,
          transition: 'opacity 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
          pointerEvents: 'none',
        } : {},
        
        ...sx,
      }}
    >
      {children}
    </Card>
  );
}
