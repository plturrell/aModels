/**
 * Extract Module
 * 
 * Main module for data extraction workflows, connecting extract service
 * with graph generation and visualization
 */

import React, { useState } from 'react';
import {
  Box,
  Typography,
  Tabs,
  Tab,
  Paper,
} from '@mui/material';
import DashboardIcon from '@mui/icons-material/Dashboard';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import ListIcon from '@mui/icons-material/List';
import HistoryIcon from '@mui/icons-material/History';
import TimelineIcon from '@mui/icons-material/Timeline';
import { ExtractDashboard } from './ExtractDashboard';
import { ExtractWizard } from './ExtractWizard';
import { ExtractJobs } from './ExtractJobs';
import { ExtractHistory } from './ExtractHistory';
import { ExtractToGraphWorkflow } from './ExtractToGraphWorkflow';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`extract-tabpanel-${index}`}
      aria-labelledby={`extract-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

interface ExtractModuleProps {
  projectId?: string;
  systemId?: string;
}

export function ExtractModule({ projectId, systemId }: ExtractModuleProps) {
  const [activeTab, setActiveTab] = useState(0);

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  return (
    <Box sx={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Paper sx={{ mb: 2 }}>
        <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
          <Typography variant="h4" gutterBottom>
            Data Extraction & Graph Generation
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Extract data from various sources and generate knowledge graphs for visualization
          </Typography>
        </Box>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs
            value={activeTab}
            onChange={handleTabChange}
            aria-label="extract module tabs"
            variant="scrollable"
            scrollButtons="auto"
          >
            <Tab
              icon={<DashboardIcon />}
              iconPosition="start"
              label="Dashboard"
              id="extract-tab-0"
              aria-controls="extract-tabpanel-0"
            />
            <Tab
              icon={<PlayArrowIcon />}
              iconPosition="start"
              label="New Extraction"
              id="extract-tab-1"
              aria-controls="extract-tabpanel-1"
            />
            <Tab
              icon={<ListIcon />}
              iconPosition="start"
              label="Jobs"
              id="extract-tab-2"
              aria-controls="extract-tabpanel-2"
            />
            <Tab
              icon={<HistoryIcon />}
              iconPosition="start"
              label="History"
              id="extract-tab-3"
              aria-controls="extract-tabpanel-3"
            />
            <Tab
              icon={<TimelineIcon />}
              iconPosition="start"
              label="Extract → Graph"
              id="extract-tab-4"
              aria-controls="extract-tabpanel-4"
            />
          </Tabs>
        </Box>
      </Paper>

      <Box sx={{ flex: 1, overflow: 'auto' }}>
        <TabPanel value={activeTab} index={0}>
          <ExtractDashboard projectId={projectId} systemId={systemId} />
        </TabPanel>

        <TabPanel value={activeTab} index={1}>
          <ExtractWizard
            projectId={projectId}
            systemId={systemId}
            onExtractionComplete={(jobId) => {
              // Switch to jobs tab to show the new extraction
              setActiveTab(2);
            }}
          />
        </TabPanel>

        <TabPanel value={activeTab} index={2}>
          <ExtractJobs projectId={projectId} systemId={systemId} />
        </TabPanel>

        <TabPanel value={activeTab} index={3}>
          <ExtractHistory projectId={projectId} systemId={systemId} />
        </TabPanel>

        <TabPanel value={activeTab} index={4}>
          <ExtractToGraphWorkflow projectId={projectId} systemId={systemId} />
        </TabPanel>
      </Box>
    </Box>
  );
}

