import React, { useState, useEffect, Suspense, lazy } from 'react';
import {
  Box,
  AppBar,
  Toolbar,
  Typography,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  IconButton,
  CircularProgress,
  useMediaQuery,
  useTheme,
  Avatar,
  Menu,
  MenuItem,
  Divider,
  Container,
  CssBaseline,
  ThemeProvider,
  createTheme
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  Code as CodeIcon,
  Build as BuildIcon,
  Storage as StorageIcon,
  MonitorHeart as MonitorIcon,
  Settings as SettingsIcon,
  Person as PersonIcon,
  Logout as LogoutIcon
} from '@mui/icons-material';
import { BrowserRouter as Router, Routes, Route, useNavigate, useLocation } from 'react-router-dom';
import { registerApplication, start as startSingleSpa } from 'single-spa';
import { getMuiTheme } from '../../../theme/theme-config';

// Micro-frontend routes
interface MicroFrontend {
  name: string;
  path: string;
  icon: React.ReactNode;
  url: string;
  activeWhen: string;
}

const microFrontends: MicroFrontend[] = [
  {
    name: 'Dashboard',
    path: '/dashboard',
    icon: <DashboardIcon />,
    url: 'http://localhost:4173/main.js',
    activeWhen: '/dashboard'
  },
  {
    name: 'Open Canvas',
    path: '/canvas',
    icon: <CodeIcon />,
    url: 'http://localhost:3000/main.js',
    activeWhen: '/canvas'
  },
  {
    name: 'LangFlow',
    path: '/langflow',
    icon: <BuildIcon />,
    url: 'http://localhost:7860/main.js',
    activeWhen: '/langflow'
  },
  {
    name: 'Database',
    path: '/database',
    icon: <StorageIcon />,
    url: 'http://localhost:7474/main.js',
    activeWhen: '/database'
  },
  {
    name: 'Monitoring',
    path: '/monitoring',
    icon: <MonitorIcon />,
    url: 'http://localhost:16686/main.js',
    activeWhen: '/monitoring'
  }
];

const DRAWER_WIDTH = 240;

const theme = createTheme(getMuiTheme());

function MicroFrontendContainer() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [loading, setLoading] = useState(false);
  const muiTheme = useTheme();
  const isMobile = useMediaQuery(muiTheme.breakpoints.down('md'));
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    // Register micro-frontends with single-spa
    microFrontends.forEach(mfe => {
      registerApplication({
        name: mfe.name,
        app: () => System.import(mfe.url),
        activeWhen: mfe.activeWhen,
        customProps: { domElement: document.getElementById('micro-frontend-root') }
      });
    });

    // Start single-spa
    startSingleSpa();
  }, []);

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const handleProfileMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleProfileMenuClose = () => {
    setAnchorEl(null);
  };

  const handleNavigation = (path: string) => {
    setLoading(true);
    navigate(path);
    if (isMobile) {
      setMobileOpen(false);
    }
    setTimeout(() => setLoading(false), 500);
  };

  const drawer = (
    <Box>
      <Toolbar>
        <Typography variant="h6" noWrap component="div">
          aModels
        </Typography>
      </Toolbar>
      <Divider />
      <List>
        {microFrontends.map((mfe) => (
          <ListItem key={mfe.name} disablePadding>
            <ListItemButton
              selected={location.pathname === mfe.path}
              onClick={() => handleNavigation(mfe.path)}
            >
              <ListItemIcon>{mfe.icon}</ListItemIcon>
              <ListItemText primary={mfe.name} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
      <Divider />
      <List>
        <ListItem disablePadding>
          <ListItemButton>
            <ListItemIcon><SettingsIcon /></ListItemIcon>
            <ListItemText primary="Settings" />
          </ListItemButton>
        </ListItem>
      </List>
    </Box>
  );

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      <CssBaseline />
      
      {/* AppBar */}
      <AppBar
        position="fixed"
        sx={{
          width: { md: `calc(100% - ${DRAWER_WIDTH}px)` },
          ml: { md: `${DRAWER_WIDTH}px` }
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { md: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            {microFrontends.find(mfe => mfe.path === location.pathname)?.name || 'aModels'}
          </Typography>

          <IconButton color="inherit" onClick={handleProfileMenuOpen}>
            <Avatar sx={{ width: 32, height: 32 }}>
              <PersonIcon />
            </Avatar>
          </IconButton>
        </Toolbar>
      </AppBar>

      {/* Profile Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleProfileMenuClose}
      >
        <MenuItem onClick={handleProfileMenuClose}>
          <ListItemIcon><PersonIcon /></ListItemIcon>
          Profile
        </MenuItem>
        <MenuItem onClick={handleProfileMenuClose}>
          <ListItemIcon><SettingsIcon /></ListItemIcon>
          Settings
        </MenuItem>
        <Divider />
        <MenuItem onClick={handleProfileMenuClose}>
          <ListItemIcon><LogoutIcon /></ListItemIcon>
          Logout
        </MenuItem>
      </Menu>

      {/* Drawer */}
      <Box
        component="nav"
        sx={{ width: { md: DRAWER_WIDTH }, flexShrink: { md: 0 } }}
      >
        {/* Mobile drawer */}
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{ keepMounted: true }}
          sx={{
            display: { xs: 'block', md: 'none' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: DRAWER_WIDTH }
          }}
        >
          {drawer}
        </Drawer>
        
        {/* Desktop drawer */}
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', md: 'block' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: DRAWER_WIDTH }
          }}
          open
        >
          {drawer}
        </Drawer>
      </Box>

      {/* Main content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          width: { md: `calc(100% - ${DRAWER_WIDTH}px)` },
          minHeight: '100vh',
          bgcolor: 'background.default'
        }}
      >
        <Toolbar /> {/* Spacer for AppBar */}
        
        {loading ? (
          <Box display="flex" justifyContent="center" alignItems="center" minHeight="80vh">
            <CircularProgress />
          </Box>
        ) : (
          <Container maxWidth="xl" sx={{ py: 3 }}>
            <div id="micro-frontend-root" style={{ minHeight: '80vh' }} />
          </Container>
        )}
      </Box>
    </Box>
  );
}

function App() {
  return (
    <ThemeProvider theme={theme}>
      <Router>
        <Routes>
          <Route path="/*" element={<MicroFrontendContainer />} />
        </Routes>
      </Router>
    </ThemeProvider>
  );
}

export default App;
