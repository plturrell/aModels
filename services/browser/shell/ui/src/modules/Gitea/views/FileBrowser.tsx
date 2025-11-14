/**
 * File Browser View
 * 
 * Browse repository files with directory tree navigation
 */

import React, { useState, useEffect, useMemo } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Breadcrumbs,
  Link,
  CircularProgress,
  Alert,
  Button,
  Paper,
  IconButton,
  MenuItem,
  Select,
  FormControl,
  InputLabel,
  TextField,
  InputAdornment,
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import FolderIcon from '@mui/icons-material/Folder';
import InsertDriveFileIcon from '@mui/icons-material/InsertDriveFile';
import HomeIcon from '@mui/icons-material/Home';
import CodeIcon from '@mui/icons-material/Code';
import SearchIcon from '@mui/icons-material/Search';
import { 
  listFiles, 
  getFileContent,
  listBranches,
  type GiteaRepository,
  type GiteaFileInfo,
  type GiteaBranch,
} from '../../../api/gitea';

interface FileBrowserProps {
  repository: GiteaRepository;
  onBack: () => void;
}

export function FileBrowser({ repository, onBack }: FileBrowserProps) {
  const [files, setFiles] = useState<GiteaFileInfo[]>([]);
  const [currentPath, setCurrentPath] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [fileContent, setFileContent] = useState<string | null>(null);
  const [contentLoading, setContentLoading] = useState(false);
  const [branch, setBranch] = useState('main');
  const [branches, setBranches] = useState<GiteaBranch[]>([]);
  const [branchesLoading, setBranchesLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  const [owner, name] = repository.full_name.split('/');
  
  // Filter files based on search query
  const filteredFiles = useMemo(() => {
    if (!searchQuery.trim()) {
      return files;
    }
    const query = searchQuery.toLowerCase();
    return files.filter((file) =>
      file.name.toLowerCase().includes(query) ||
      file.path.toLowerCase().includes(query)
    );
  }, [files, searchQuery]);

  useEffect(() => {
    loadBranches();
  }, []);

  useEffect(() => {
    loadFiles(currentPath);
  }, [currentPath, branch]);

  const loadBranches = async () => {
    setBranchesLoading(true);
    try {
      const branchList = await listBranches(owner, name);
      setBranches(branchList);
      if (branchList.length > 0 && !branchList.find(b => b.name === branch)) {
        setBranch(branchList[0].name);
      }
    } catch (err) {
      console.error('Failed to load branches:', err);
    } finally {
      setBranchesLoading(false);
    }
  };

  const loadFiles = async (path: string) => {
    setLoading(true);
    setError(null);
    try {
      const fileList = await listFiles(owner, name, path || undefined, branch);
      setFiles(fileList);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load files');
    } finally {
      setLoading(false);
    }
  };

  const handleFileClick = async (file: GiteaFileInfo) => {
    if (file.type === 'dir') {
      setCurrentPath(file.path);
      setSelectedFile(null);
      setFileContent(null);
    } else {
      setSelectedFile(file.path);
      setContentLoading(true);
      try {
        const content = await getFileContent(owner, name, file.path, branch);
        setFileContent(content.content);
      } catch (err) {
        setFileContent(`Error loading file: ${err instanceof Error ? err.message : 'Unknown error'}`);
      } finally {
        setContentLoading(false);
      }
    }
  };

  const handleBreadcrumbClick = (path: string) => {
    setCurrentPath(path);
    setSelectedFile(null);
    setFileContent(null);
  };

  const pathParts = currentPath ? currentPath.split('/').filter(Boolean) : [];

  if (loading && files.length === 0) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <Button startIcon={<ArrowBackIcon />} onClick={onBack}>
          Back
        </Button>
        <Box sx={{ flex: 1 }} />
        <FormControl size="small" sx={{ minWidth: 120 }}>
          <InputLabel>Branch</InputLabel>
          <Select
            value={branch}
            label="Branch"
            onChange={(e) => setBranch(e.target.value)}
            disabled={branchesLoading}
          >
            {branches.map((b) => (
              <MenuItem key={b.name} value={b.name}>
                {b.name}
              </MenuItem>
            ))}
            {branches.length === 0 && (
              <MenuItem value="main">main</MenuItem>
            )}
          </Select>
        </FormControl>
      </Box>

      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2, gap: 2 }}>
            <Typography variant="h6" sx={{ flex: 1 }}>
              {repository.name}
            </Typography>
            <TextField
              placeholder="Search files..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              size="small"
              sx={{ width: 300 }}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <SearchIcon />
                  </InputAdornment>
                ),
              }}
            />
          </Box>
          <Breadcrumbs sx={{ mb: 2 }}>
            <Link
              component="button"
              variant="body1"
              onClick={() => handleBreadcrumbClick('')}
              sx={{ display: 'flex', alignItems: 'center' }}
            >
              <HomeIcon sx={{ mr: 0.5, fontSize: 20 }} />
              {repository.name}
            </Link>
            {pathParts.map((part, index) => {
              const path = pathParts.slice(0, index + 1).join('/');
              return (
                <Link
                  key={path}
                  component="button"
                  variant="body1"
                  onClick={() => handleBreadcrumbClick(path)}
                >
                  {part}
                </Link>
              );
            })}
          </Breadcrumbs>

          {error ? (
            <Alert severity="error">{error}</Alert>
          ) : (
            <List>
              {filteredFiles.length === 0 && files.length > 0 ? (
                <Typography variant="body2" color="text.secondary" sx={{ p: 2 }}>
                  No files match your search
                </Typography>
              ) : (
                filteredFiles.map((file) => (
                <ListItem
                  key={file.path}
                  button
                  onClick={() => handleFileClick(file)}
                  sx={{
                    '&:hover': {
                      backgroundColor: 'action.hover',
                    },
                  }}
                >
                  <ListItemIcon>
                    {file.type === 'dir' ? (
                      <FolderIcon color="primary" />
                    ) : (
                      <InsertDriveFileIcon />
                    )}
                  </ListItemIcon>
                  <ListItemText
                    primary={file.name}
                    secondary={file.type === 'file' ? `${(file.size / 1024).toFixed(2)} KB` : 'Directory'}
                  />
                </ListItem>
                ))
              )}
              {files.length === 0 && (
                <Typography variant="body2" color="text.secondary" sx={{ p: 2 }}>
                  This directory is empty
                </Typography>
              )}
            </List>
          )}
        </CardContent>
      </Card>

      {selectedFile && (
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <CodeIcon sx={{ mr: 1 }} />
              <Typography variant="h6">
                {selectedFile.split('/').pop()}
              </Typography>
              <Box sx={{ flex: 1 }} />
              <IconButton
                size="small"
                onClick={() => {
                  setSelectedFile(null);
                  setFileContent(null);
                }}
              >
                <ArrowBackIcon />
              </IconButton>
            </Box>
            {contentLoading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
                <CircularProgress />
              </Box>
            ) : (
              <Paper
                variant="outlined"
                sx={{
                  p: 2,
                  backgroundColor: 'grey.900',
                  color: 'grey.100',
                  fontFamily: 'monospace',
                  fontSize: '0.875rem',
                  maxHeight: 500,
                  overflow: 'auto',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                }}
              >
                {fileContent || 'No content'}
              </Paper>
            )}
          </CardContent>
        </Card>
      )}
    </Box>
  );
}

