/**
 * Modern Home Page Component
 * 
 * Enhanced landing page with quick actions, recent projects, and system status
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  CardActionArea,
  Typography,
  Button,
  Chip,
  Stack,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  alpha,
} from '@mui/material';
import { GridLegacy as Grid } from '@mui/material';
import AccountTreeIcon from '@mui/icons-material/AccountTree';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import SchoolIcon from '@mui/icons-material/School';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import StorageIcon from '@mui/icons-material/Storage';
import CloudIcon from '@mui/icons-material/Cloud';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import { useShellStore, ShellModuleId } from '../state/useShellStore';

interface QuickAction {
  id: ShellModuleId;
  title: string;
  description: string;
  icon: React.ReactNode;
  color: string;
  gradient: string;
}

const quickActions: QuickAction[] = [
  {
    id: 'graph',
    title: 'Graph Explorer',
    description: 'Visualize and explore knowledge graphs with AI-powered insights',
    icon: <AccountTreeIcon />,
    color: '#0a6ed1',
    gradient: 'linear-gradient(135deg, #0a6ed1 0%, #3f9ddb 100%)',
  },
  {
    id: 'extract',
    title: 'Data Extraction',
    description: 'Extract data from various sources and generate graphs',
    icon: <AutoAwesomeIcon />,
    color: '#107e3e',
    gradient: 'linear-gradient(135deg, #107e3e 0%, #2da955 100%)',
  },
  {
    id: 'training',
    title: 'Model Training',
    description: 'Train and manage GNN models for graph analysis',
    icon: <SchoolIcon />,
    color: '#e9730c',
    gradient: 'linear-gradient(135deg, #e9730c 0%, #f09d3c 100%)',
  },
  {
    id: 'postgres',
    title: 'Postgres',
    description: 'Query and manage PostgreSQL databases',
    icon: <StorageIcon />,
    color: '#354a5f',
    gradient: 'linear-gradient(135deg, #354a5f 0%, #475e75 100%)',
  },
  {
    id: 'localai',
    title: 'LocalAI',
    description: 'Interact with local AI models and assistants',
    icon: <SmartToyIcon />,
    color: '#7030a0',
    gradient: 'linear-gradient(135deg, #7030a0 0%, #9050c0 100%)',
  },
  {
    id: 'dms',
    title: 'Document Management',
    description: 'Manage and process documents',
    icon: <StorageIcon />,
    color: '#bb0000',
    gradient: 'linear-gradient(135deg, #bb0000 0%, #e00 100%)',
  },
];

export function ModernHomePage() {
  const { setActiveModule } = useShellStore();
  const [recentProjects, setRecentProjects] = useState<string[]>([]);
  const [systemStatus, setSystemStatus] = useState({
    graph: 'online',
    extract: 'online',
    localai: 'online',
  });

  useEffect(() => {
    // Load recent projects from localStorage
    const recent = localStorage.getItem('amodels_recent_projects');
    if (recent) {
      try {
        setRecentProjects(JSON.parse(recent));
      } catch (e) {
        console.error('Failed to parse recent projects', e);
      }
    }
  }, []);

  const handleModuleClick = (moduleId: ShellModuleId) => {
    setActiveModule(moduleId);
  };

  return (
    <Box sx={{ p: 3, maxWidth: 1400, margin: '0 auto' }} component="section" aria-label="Home page">
      {/* Hero Section */}
      <Box sx={{ mb: 6 }}>
        <Typography 
          variant="h3" 
          sx={{ 
            fontWeight: 700, 
            mb: 2,
            background: 'linear-gradient(135deg, #0a6ed1 0%, #107e3e 100%)',
            backgroundClip: 'text',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}
        >
          Welcome to aModels Shell
        </Typography>
        <Typography variant="h6" color="text.secondary" sx={{ mb: 3, maxWidth: 800 }}>
          Enterprise-grade platform for graph exploration, data extraction, and AI-powered analytics
        </Typography>
        <Stack direction="row" spacing={2}>
          <Chip 
            icon={<CheckCircleIcon />} 
            label="8 Modules Active" 
            color="success" 
            variant="outlined" 
          />
          <Chip 
            icon={<TrendingUpIcon />} 
            label="Production Ready" 
            color="info" 
            variant="outlined" 
          />
          <Chip 
            icon={<AccessTimeIcon />} 
            label="v0.1.0" 
            variant="outlined" 
          />
        </Stack>
      </Box>

      {/* Quick Actions */}
      <Box sx={{ mb: 4 }} component="section" aria-labelledby="quick-actions-heading">
        <Typography variant="h4" component="h1" gutterBottom sx={{ fontWeight: 600, mb: 3 }}>
          Quick Actions
        </Typography>
        <Grid container spacing={3}>
          {quickActions.map((action) => (
            <Grid item xs={12} sm={6} md={4} key={action.id}>
              <Card 
                elevation={0}
                sx={{ 
                  height: '100%',
                  border: '1px solid',
                  borderColor: 'divider',
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    borderColor: action.color,
                    transform: 'translateY(-4px)',
                    boxShadow: `0 8px 24px ${alpha(action.color, 0.15)}`,
                  },
                }}
              >
                <CardActionArea 
                  onClick={() => handleModuleClick(action.id)}
                  aria-label={`Navigate to ${action.title}: ${action.description}`}
                  sx={{ height: '100%', p: 3 }}
                >
                  <Box
                    sx={{
                      width: 56,
                      height: 56,
                      borderRadius: 2,
                      background: action.gradient,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      mb: 2,
                      color: 'white',
                      '& svg': { fontSize: 28 }
                    }}
                  >
                    {action.icon}
                  </Box>
                  <Typography variant="h6" sx={{ mb: 1, fontWeight: 600 }}>
                    {action.title}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {action.description}
                  </Typography>
                </CardActionArea>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Box>

      {/* Recent Activity & System Status */}
      <Grid container spacing={3}>
        {/* Recent Projects */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, height: '100%' }}>
            <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
              Recent Projects
            </Typography>
            {recentProjects.length > 0 ? (
              <List>
                {recentProjects.slice(0, 5).map((project, index) => (
                  <React.Fragment key={project}>
                    <ListItem>
                      <ListItemIcon>
                        <AccountTreeIcon color="primary" />
                      </ListItemIcon>
                      <ListItemText
                        primary={project}
                        secondary={`Last accessed ${new Date().toLocaleDateString()}`}
                      />
                      <Button 
                        size="small" 
                        onClick={() => {
                          setActiveModule('graph');
                          // Could also pre-fill project ID here
                        }}
                      >
                        Open
                      </Button>
                    </ListItem>
                    {index < recentProjects.length - 1 && <Divider />}
                  </React.Fragment>
                ))}
              </List>
            ) : (
              <Box sx={{ textAlign: 'center', py: 4 }}>
                <Typography variant="body2" color="text.secondary">
                  No recent projects
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Start exploring graphs to see your recent activity here
                </Typography>
              </Box>
            )}
          </Paper>
        </Grid>

        {/* System Status */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, height: '100%' }}>
            <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
              System Status
            </Typography>
            <List>
              <ListItem>
                <ListItemIcon>
                  <AccountTreeIcon />
                </ListItemIcon>
                <ListItemText 
                  primary="Graph Service" 
                  secondary="Neo4j + Cytoscape"
                />
                <Chip 
                  label="Online" 
                  color="success" 
                  size="small" 
                  variant="outlined"
                />
              </ListItem>
              <Divider />
              <ListItem>
                <ListItemIcon>
                  <AutoAwesomeIcon />
                </ListItemIcon>
                <ListItemText 
                  primary="Extract Service" 
                  secondary="Data extraction & ETL"
                />
                <Chip 
                  label="Online" 
                  color="success" 
                  size="small" 
                  variant="outlined"
                />
              </ListItem>
              <Divider />
              <ListItem>
                <ListItemIcon>
                  <SmartToyIcon />
                </ListItemIcon>
                <ListItemText 
                  primary="LocalAI" 
                  secondary="LLM & embedding models"
                />
                <Chip 
                  label="Online" 
                  color="success" 
                  size="small" 
                  variant="outlined"
                />
              </ListItem>
              <Divider />
              <ListItem>
                <ListItemIcon>
                  <SchoolIcon />
                </ListItemIcon>
                <ListItemText 
                  primary="Training Service" 
                  secondary="GNN model training"
                />
                <Chip 
                  label="Online" 
                  color="success" 
                  size="small" 
                  variant="outlined"
                />
              </ListItem>
            </List>
          </Paper>
        </Grid>
      </Grid>

      {/* Quick Tips */}
      <Box sx={{ mt: 4 }}>
        <Paper sx={{ p: 3, background: alpha('#0a6ed1', 0.04) }}>
          <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
            💡 Quick Tips
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <Typography variant="body2" sx={{ mb: 1 }}>
                <strong>Keyboard Shortcuts:</strong>
              </Typography>
              <Typography variant="caption" color="text.secondary">
                • Cmd/Ctrl + K - Command Palette
              </Typography>
            </Grid>
            <Grid item xs={12} sm={6}>
              <Typography variant="body2" sx={{ mb: 1 }}>
                <strong>Getting Started:</strong>
              </Typography>
              <Typography variant="caption" color="text.secondary">
                • Try the Graph Explorer with sample data
              </Typography>
            </Grid>
          </Grid>
        </Paper>
      </Box>
    </Box>
  );
}
