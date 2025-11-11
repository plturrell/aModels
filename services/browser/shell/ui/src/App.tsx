import { useState } from 'react';
import { ThemeProvider, CssBaseline, Typography, Box, Drawer, List, ListItem, ListItemButton, ListItemIcon, ListItemText, AppBar, Toolbar } from "@mui/material";
import { WorkbenchLayout } from "./components/WorkbenchLayout";
import { SessionsPanel } from "./components/SessionsPanel";
import { AgentLogPanel } from "./components/AgentLogPanel";
import { Canvas } from "./components/Canvas";
import { CommandPalette } from "./components/CommandPalette";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { useShellStore, ShellModuleId } from "./state/useShellStore";
import { GraphModule } from "./modules/Graph/GraphModule";
import { LocalAIModule } from "./modules/LocalAI/LocalAIModule";
import { DMSModule } from "./modules/DMS/DMSModule";
import { SAPModule } from "./modules/SAP/SAPModule";
import AccountTreeIcon from '@mui/icons-material/AccountTree';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import StorageIcon from '@mui/icons-material/Storage';
import CloudIcon from '@mui/icons-material/Cloud';
import HomeIcon from '@mui/icons-material/Home';
import theme from "./theme-sap";

const drawerWidth = 240;

function App() {
  const { activeModule, setActiveModule } = useShellStore();
  const [sessions, setSessions] = useState<any[]>([]);
  const [activeSession, setActiveSession] = useState<any>(null);

  const handleCommand = (command: string, data: any) => {
    const newSession = { id: Date.now(), command, data };
    setSessions([...sessions, newSession]);
    setActiveSession(newSession);
  };

  const handleError = (error: Error) => {
    console.error('Application error:', error);
    // TODO: Send to error tracking service
  };

  const renderModule = () => {
    switch (activeModule) {
      case 'graph':
        return <GraphModule />;
      case 'localai':
        return <LocalAIModule />;
      case 'dms':
        return <DMSModule />;
      case 'sap':
        return <SAPModule />;
      case 'home':
      default:
        return (
          <Box sx={{ p: 3, height: '100%', overflow: 'auto' }}>
            <Typography variant="h4" gutterBottom>aModels Shell</Typography>
            <Typography variant="body1" color="text.secondary">
              Select a module from the sidebar to get started.
            </Typography>
          </Box>
        );
    }
  };

  const navigationItems: Array<{ id: ShellModuleId; label: string; icon: React.ReactNode }> = [
    { id: 'home', label: 'Home', icon: <HomeIcon /> },
    { id: 'graph', label: 'Graph', icon: <AccountTreeIcon /> },
    { id: 'localai', label: 'LocalAI', icon: <SmartToyIcon /> },
    { id: 'dms', label: 'DMS', icon: <StorageIcon /> },
    { id: 'sap', label: 'SAP', icon: <CloudIcon /> },
  ];

  return (
    <ErrorBoundary onError={handleError}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <CommandPalette onCommand={handleCommand} />
        <Box sx={{ display: 'flex', height: '100vh' }}>
          <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
            <Toolbar>
              <Typography variant="h6" noWrap component="div">
                aModels Shell
              </Typography>
            </Toolbar>
          </AppBar>
          <Drawer
            variant="permanent"
            sx={{
              width: drawerWidth,
              flexShrink: 0,
              '& .MuiDrawer-paper': {
                width: drawerWidth,
                boxSizing: 'border-box',
              },
            }}
          >
            <Toolbar />
            <List>
              {navigationItems.map((item) => (
                <ListItem key={item.id} disablePadding>
                  <ListItemButton
                    selected={activeModule === item.id}
                    onClick={() => setActiveModule(item.id)}
                  >
                    <ListItemIcon>{item.icon}</ListItemIcon>
                    <ListItemText primary={item.label} />
                  </ListItemButton>
                </ListItem>
              ))}
            </List>
          </Drawer>
          <Box
            component="main"
            sx={{
              flexGrow: 1,
              p: 3,
              width: `calc(100% - ${drawerWidth}px)`,
              mt: '64px',
              height: 'calc(100vh - 64px)',
              overflow: 'auto',
            }}
          >
            <ErrorBoundary>
              {renderModule()}
            </ErrorBoundary>
          </Box>
        </Box>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;
