import { useState, lazy, Suspense } from 'react';
import { useGlobalShortcuts } from './hooks/useGlobalShortcuts';
import { ShortcutsDialog } from './components/ShortcutsDialog';
import { ThemeProvider, CssBaseline, Typography, Box, Drawer, List, ListItem, ListItemButton, ListItemIcon, ListItemText, AppBar, Toolbar } from "@mui/material";
import { WorkbenchLayout } from "./components/WorkbenchLayout";
import { SessionsPanel } from "./components/SessionsPanel";
import { AgentLogPanel } from "./components/AgentLogPanel";
import { Canvas } from "./components/Canvas";
import { CommandPalette } from "./components/CommandPalette";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { ModernHomePage } from "./components/ModernHomePage";
import { ModuleLoader } from "./components/ModuleLoader";
import { SkipLink } from "./components/SkipLink";
import { useShellStore, ShellModuleId } from "./state/useShellStore";
// Lazy load modules for better performance
const GraphModule = lazy(() => import('./modules/Graph/GraphModule').then(m => ({ default: m.GraphModule })));
const LocalAIModule = lazy(() => import('./modules/LocalAI/LocalAIModule').then(m => ({ default: m.LocalAIModule })));
const DMSModule = lazy(() => import('./modules/DMS/DMSModule').then(m => ({ default: m.DMSModule })));
const SAPModule = lazy(() => import('./modules/SAP/SAPModule').then(m => ({ default: m.SAPModule })));
const ExtractModule = lazy(() => import('./modules/Extract/ExtractModule').then(m => ({ default: m.ExtractModule })));
const TrainingModule = lazy(() => import('./modules/Training/TrainingModule').then(m => ({ default: m.TrainingModule })));
const PostgresModule = lazy(() => import('./modules/Postgres/PostgresModule').then(m => ({ default: m.PostgresModule })));
import AccountTreeIcon from '@mui/icons-material/AccountTree';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import StorageIcon from '@mui/icons-material/Storage';
import CloudIcon from '@mui/icons-material/Cloud';
import HomeIcon from '@mui/icons-material/Home';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import SchoolIcon from '@mui/icons-material/School';
import theme from "./theme-sap";

const drawerWidth = 240;

function App() {
  const { activeModule, setActiveModule } = useShellStore();
  const [sessions, setSessions] = useState<any[]>([]);
  const [activeSession, setActiveSession] = useState<any>(null);
  const [showShortcuts, setShowShortcuts] = useState(false);

  // Global keyboard shortcuts
  useGlobalShortcuts({
    onNavigate: (moduleId) => setActiveModule(moduleId as any),
    onHelp: () => setShowShortcuts(true),
    onEscape: () => setShowShortcuts(false),
  });

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
      case 'extract':
        return <ExtractModule />;
      case 'training':
        return <TrainingModule />;
      case 'postgres':
        return <PostgresModule />;
      case 'home':
      default:
        return <ModernHomePage />;
    }
  };

  const navigationItems: Array<{ id: ShellModuleId; label: string; icon: React.ReactNode }> = [
    { id: 'home', label: 'Home', icon: <HomeIcon /> },
    { id: 'graph', label: 'Graph', icon: <AccountTreeIcon /> },
    { id: 'extract', label: 'Extract', icon: <AutoAwesomeIcon /> },
    { id: 'training', label: 'Training', icon: <SchoolIcon /> },
    { id: 'postgres', label: 'Postgres', icon: <StorageIcon /> },
    { id: 'localai', label: 'LocalAI', icon: <SmartToyIcon /> },
    { id: 'dms', label: 'DMS', icon: <StorageIcon /> },
    { id: 'sap', label: 'SAP', icon: <CloudIcon /> },
  ];

  return (
    <ErrorBoundary onError={handleError}>
      <SkipLink />
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <CommandPalette onCommand={handleCommand} />
        <Box sx={{ display: 'flex', height: '100vh' }} role="application" aria-label="aModels Shell Application">
          <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }} component="header" role="banner">
            <Toolbar>
              <Typography variant="h6" noWrap component="div">
                aModels Shell
              </Typography>
            </Toolbar>
          </AppBar>
          <Drawer
            variant="permanent"
            component="nav"
            aria-label="Main navigation"
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
            <List component="nav" aria-label="Module navigation">
              {navigationItems.map((item) => (
                <ListItem key={item.id} disablePadding>
                  <ListItemButton
                    selected={activeModule === item.id}
                    onClick={() => setActiveModule(item.id)}
                    aria-label={`Navigate to ${item.label} module`}
                    aria-current={activeModule === item.id ? 'page' : undefined}
                  >
                    <ListItemIcon aria-hidden="true">{item.icon}</ListItemIcon>
                    <ListItemText primary={item.label} />
                  </ListItemButton>
                </ListItem>
              ))}
            </List>
          </Drawer>
          <Box
            component="main"
            id="main-content"
            tabIndex={-1}
            role="main"
            aria-label="Main content area"
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
              <Suspense fallback={<ModuleLoader />}>
                {renderModule()}
              </Suspense>
            </ErrorBoundary>
          </Box>
        </Box>
      </ThemeProvider>
      
      {/* Keyboard Shortcuts Help Dialog */}
      <ShortcutsDialog open={showShortcuts} onClose={() => setShowShortcuts(false)} />
    </ErrorBoundary>
  );
}

export default App;
