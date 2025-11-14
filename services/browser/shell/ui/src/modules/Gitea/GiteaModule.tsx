/**
 * Gitea Module
 * 
 * Main module for Gitea repository management with GitHub-like UI
 */

import React, { useState } from 'react';
import {
  Box,
  Typography,
  Tabs,
  Tab,
  Paper,
} from '@mui/material';
import ListIcon from '@mui/icons-material/List';
import AddIcon from '@mui/icons-material/Add';
import FolderIcon from '@mui/icons-material/Folder';
import { RepositoryList } from './views/RepositoryList';
import { CreateRepository } from './views/CreateRepository';
import { RepositoryDetail } from './views/RepositoryDetail';
import { FileBrowser } from './views/FileBrowser';
import type { GiteaRepository } from '../../api/gitea';

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
      id={`gitea-tabpanel-${index}`}
      aria-labelledby={`gitea-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

export function GiteaModule() {
  const [activeTab, setActiveTab] = useState(0);
  const [selectedRepo, setSelectedRepo] = useState<GiteaRepository | null>(null);
  const [viewMode, setViewMode] = useState<'list' | 'detail' | 'files'>('list');

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
    if (newValue === 0) {
      setViewMode('list');
      setSelectedRepo(null);
    }
  };

  const handleRepositorySelect = (repo: GiteaRepository) => {
    setSelectedRepo(repo);
    setViewMode('detail');
  };

  const handleViewFiles = (repo: GiteaRepository) => {
    setSelectedRepo(repo);
    setViewMode('files');
  };

  const handleRepositoryCreated = (repo: GiteaRepository) => {
    setSelectedRepo(repo);
    setViewMode('detail');
    setActiveTab(0); // Switch to list tab (which will show detail)
  };

  return (
    <Box sx={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Paper sx={{ mb: 2 }}>
        <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
          <Typography variant="h4" gutterBottom>
            Gitea Repository Management
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Manage Gitea repositories with a GitHub-like interface
          </Typography>
        </Box>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs
            value={activeTab}
            onChange={handleTabChange}
            aria-label="gitea module tabs"
            variant="scrollable"
            scrollButtons="auto"
          >
            <Tab
              icon={<ListIcon />}
              iconPosition="start"
              label={selectedRepo && viewMode !== 'list' ? selectedRepo.name : 'Repositories'}
              id="gitea-tab-0"
              aria-controls="gitea-tabpanel-0"
            />
            <Tab
              icon={<AddIcon />}
              iconPosition="start"
              label="New Repository"
              id="gitea-tab-1"
              aria-controls="gitea-tabpanel-1"
            />
          </Tabs>
        </Box>
      </Paper>

      <Box sx={{ flex: 1, overflow: 'auto' }}>
        {viewMode === 'list' && (
          <TabPanel value={activeTab} index={0}>
            <RepositoryList
              onRepositorySelect={handleRepositorySelect}
              onViewFiles={handleViewFiles}
            />
          </TabPanel>
        )}

        {viewMode === 'detail' && selectedRepo && (
          <TabPanel value={activeTab} index={0}>
            <RepositoryDetail
              repository={selectedRepo}
              onBack={() => {
                setViewMode('list');
                setSelectedRepo(null);
              }}
              onViewFiles={() => handleViewFiles(selectedRepo)}
            />
          </TabPanel>
        )}

        {viewMode === 'files' && selectedRepo && (
          <TabPanel value={activeTab} index={0}>
            <FileBrowser
              repository={selectedRepo}
              onBack={() => {
                setViewMode('detail');
              }}
            />
          </TabPanel>
        )}

        <TabPanel value={activeTab} index={1}>
          <CreateRepository onRepositoryCreated={handleRepositoryCreated} />
        </TabPanel>
      </Box>
    </Box>
  );
}

