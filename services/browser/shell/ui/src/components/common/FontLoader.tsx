import React, { useEffect, useState } from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';

interface FontLoaderProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

export const FontLoader: React.FC<FontLoaderProps> = ({ children, fallback }) => {
  const [fontsLoaded, setFontsLoaded] = useState(false);

  useEffect(() => {
    // Check if fonts are already loaded
    const checkFonts = async () => {
      try {
        // Check Inter font
        await document.fonts.load('1em Inter');
        
        // Check JetBrains Mono font
        await document.fonts.load('1em "JetBrains Mono"');
        
        // Check SAP-icons font
        await document.fonts.load('1em SAP-icons');
        
        setFontsLoaded(true);
      } catch (error) {
        console.warn('Some fonts failed to load:', error);
        setFontsLoaded(true); // Continue with fallback fonts
      }
    };

    checkFonts();

    // Listen for font loading events
    document.fonts.ready.then(() => {
      setFontsLoaded(true);
    });
  }, []);

  if (!fontsLoaded && fallback) {
    return <>{fallback}</>;
  }

  if (!fontsLoaded) {
    return (
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100vh',
          gap: 2,
        }}
      >
        <CircularProgress size={32} />
        <Typography variant="body2" color="text.secondary">
          Loading fonts...
        </Typography>
      </Box>
    );
  }

  return <>{children}</>;
};
