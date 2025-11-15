import React from 'react';
import { 
  Dashboard, 
  Code, 
  Build, 
  Storage, 
  MonitorHeart,
  GitHub,
  Science,
  Hub
} from '@mui/icons-material';
import { 
  List, 
  ListItem, 
  ListItemButton, 
  ListItemIcon, 
  ListItemText,
  Divider,
  Typography,
  Chip
} from '@mui/material';

interface Service {
  name: string;
  url: string;
  icon: React.ReactNode;
  status: 'running' | 'available' | 'unavailable';
  category: string;
}

const services: Service[] = [
  // Core Dashboards
  {
    name: 'Service Portal',
    url: 'http://localhost:8888',
    icon: <Hub />,
    status: 'running',
    category: 'Core'
  },
  {
    name: 'Main Dashboard',
    url: 'http://localhost:4173',
    icon: <Dashboard />,
    status: 'running',
    category: 'Core'
  },
  {
    name: 'Open Canvas',
    url: 'http://localhost:3000',
    icon: <Code />,
    status: 'running',
    category: 'Core'
  },
  
  // Development Tools
  {
    name: 'LangFlow Builder',
    url: 'http://localhost:7860',
    icon: <Build />,
    status: 'available',
    category: 'Development'
  },
  {
    name: 'Gitea Repository',
    url: 'http://localhost:3003',
    icon: <GitHub />,
    status: 'running',
    category: 'Development'
  },
  {
    name: 'JupyterLab',
    url: 'http://ec2-50-17-166-162.compute-1.amazonaws.com:8888/lab',
    icon: <Science />,
    status: 'running',
    category: 'Development'
  },
  
  // Databases
  {
    name: 'Neo4j Browser',
    url: 'http://localhost:7474',
    icon: <Storage />,
    status: 'running',
    category: 'Database'
  },
  {
    name: 'PostgreSQL Admin',
    url: 'http://localhost:8082',
    icon: <Storage />,
    status: 'available',
    category: 'Database'
  },
  
  // Observability
  {
    name: 'Jaeger Tracing',
    url: 'http://localhost:16686',
    icon: <MonitorHeart />,
    status: 'available',
    category: 'Monitoring'
  },
  {
    name: 'Grafana',
    url: 'http://localhost:3001',
    icon: <MonitorHeart />,
    status: 'available',
    category: 'Monitoring'
  },
];

const getStatusColor = (status: string) => {
  switch (status) {
    case 'running': return 'success';
    case 'available': return 'warning';
    case 'unavailable': return 'error';
    default: return 'default';
  }
};

const ServiceNav: React.FC = () => {
  const categories = Array.from(new Set(services.map(s => s.category)));
  
  return (
    <List>
      {categories.map(category => (
        <React.Fragment key={category}>
          <ListItem>
            <Typography variant="caption" color="text.secondary" fontWeight="bold">
              {category}
            </Typography>
          </ListItem>
          
          {services
            .filter(s => s.category === category)
            .map(service => (
              <ListItem key={service.name} disablePadding>
                <ListItemButton 
                  component="a" 
                  href={service.url} 
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <ListItemIcon sx={{ minWidth: 40 }}>
                    {service.icon}
                  </ListItemIcon>
                  <ListItemText 
                    primary={service.name}
                    primaryTypographyProps={{ variant: 'body2' }}
                  />
                  <Chip 
                    size="small" 
                    label={service.status} 
                    color={getStatusColor(service.status) as any}
                    sx={{ height: 20, fontSize: '0.7rem' }}
                  />
                </ListItemButton>
              </ListItem>
            ))}
          
          <Divider sx={{ my: 1 }} />
        </React.Fragment>
      ))}
    </List>
  );
};

export default ServiceNav;
