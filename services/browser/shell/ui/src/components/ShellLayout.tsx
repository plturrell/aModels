import React from 'react';
import { Box, Drawer, CssBaseline, AppBar, Toolbar, Typography } from '@mui/material';

interface ShellLayoutProps {
  nav: React.ReactNode;
  children: React.ReactNode;
}

const drawerWidth = 240;

export const ShellLayout: React.FC<ShellLayoutProps> = ({ nav, children }) => {
  return (
    <Box sx={{ display: 'flex' }}>
      <CssBaseline />
      <AppBar
        position="fixed"
        sx={{ width: `calc(100% - ${drawerWidth}px)`, ml: `${drawerWidth}px` }}
      >
        <Toolbar>
          <Typography variant="h6" noWrap component="div">
            aModels Shell
          </Typography>
        </Toolbar>
      </AppBar>
      <Drawer
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: drawerWidth,
            boxSizing: 'border-box',
          },
        }}
        variant="permanent"
        anchor="left"
      >
        {nav}
      </Drawer>
      <Box
        component="main"
        sx={{ flexGrow: 1, bgcolor: 'background.default', p: 3, width: `calc(100% - ${drawerWidth}px)` }}
      >
        <Toolbar /> {/* This is to offset the content below the AppBar */}
        {children}
      </Box>
    </Box>
  );
};