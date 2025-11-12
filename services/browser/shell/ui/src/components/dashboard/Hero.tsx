/**
 * Hero Component - Apple-style Hero Section
 * Big, bold, beautiful - makes you want to explore
 */

import React from 'react';
import { Box, Typography, Container } from '@mui/material';

interface HeroProps {
  title: string;
  subtitle?: string;
  gradient?: boolean;
  children?: React.ReactNode;
}

export function Hero({ title, subtitle, gradient = false, children }: HeroProps) {
  return (
    <Box
      sx={{
        position: 'relative',
        py: { xs: 8, md: 12 },
        mb: 6,
        background: gradient
          ? 'linear-gradient(135deg, rgba(0, 122, 255, 0.05) 0%, rgba(88, 86, 214, 0.05) 100%)'
          : 'transparent',
        borderRadius: 4,
        overflow: 'hidden',
        '&::before': gradient ? {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'radial-gradient(circle at 30% 50%, rgba(0, 122, 255, 0.1) 0%, transparent 50%)',
          pointerEvents: 'none',
        } : {},
      }}
    >
      <Container maxWidth="lg">
        <Box sx={{ textAlign: 'center', maxWidth: 800, mx: 'auto' }}>
          <Typography
            variant="h1"
            sx={{
              fontSize: { xs: '2.5rem', sm: '3.5rem', md: '4.5rem' },
              fontWeight: 700,
              lineHeight: 1.1,
              letterSpacing: '-0.02em',
              mb: 2,
              background: gradient
                ? 'linear-gradient(135deg, #007AFF 0%, #5856D6 100%)'
                : 'inherit',
              backgroundClip: gradient ? 'text' : 'inherit',
              WebkitBackgroundClip: gradient ? 'text' : 'inherit',
              WebkitTextFillColor: gradient ? 'transparent' : 'inherit',
              animation: 'fadeInUp 0.6s ease-out',
              '@keyframes fadeInUp': {
                from: {
                  opacity: 0,
                  transform: 'translateY(20px)',
                },
                to: {
                  opacity: 1,
                  transform: 'translateY(0)',
                },
              },
            }}
          >
            {title}
          </Typography>
          
          {subtitle && (
            <Typography
              variant="h5"
              sx={{
                fontSize: { xs: '1.125rem', sm: '1.375rem' },
                fontWeight: 400,
                lineHeight: 1.5,
                color: 'text.secondary',
                mb: 4,
                animation: 'fadeInUp 0.6s ease-out 0.1s both',
              }}
            >
              {subtitle}
            </Typography>
          )}
          
          {children && (
            <Box sx={{ animation: 'fadeInUp 0.6s ease-out 0.2s both' }}>
              {children}
            </Box>
          )}
        </Box>
      </Container>
    </Box>
  );
}
