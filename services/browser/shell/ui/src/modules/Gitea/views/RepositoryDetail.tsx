/**
 * Repository Detail View
 * 
 * Displays detailed information about a Gitea repository
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Chip,
  Button,
  Divider,
  List,
  ListItem,
  ListItemText,
  CircularProgress,
  Alert,
  IconButton,
  Tooltip,
  Grid,
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import FolderIcon from '@mui/icons-material/Folder';
import CodeIcon from '@mui/icons-material/Code';
import HistoryIcon from '@mui/icons-material/History';
import DeleteIcon from '@mui/icons-material/Delete';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import { 
  getRepository, 
  listBranches, 
  listCommits, 
  deleteRepository,
  type GiteaRepository,
  type GiteaBranch,
  type GiteaCommit,
} from '../../../api/gitea';

interface RepositoryDetailProps {
  repository: GiteaRepository;
  onBack: () => void;
  onViewFiles: () => void;
}

export function RepositoryDetail({ repository, onBack, onViewFiles }: RepositoryDetailProps) {
  const [repo, setRepo] = useState<GiteaRepository>(repository);
  const [branches, setBranches] = useState<GiteaBranch[]>([]);
  const [commits, setCommits] = useState<GiteaCommit[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    const loadDetails = async () => {
      setLoading(true);
      setError(null);
      try {
        const [owner, name] = repository.full_name.split('/');
        const [repoData, branchesData, commitsData] = await Promise.all([
          getRepository(owner, name),
          listBranches(owner, name).catch(() => []),
          listCommits(owner, name, undefined, 10).catch(() => []),
        ]);
        setRepo(repoData);
        setBranches(branchesData);
        setCommits(commitsData);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load repository details');
      } finally {
        setLoading(false);
      }
    };

    loadDetails();
  }, [repository]);

  const handleDelete = async () => {
    if (!window.confirm(`Are you sure you want to delete ${repo.full_name}? This action cannot be undone.`)) {
      return;
    }

    setDeleting(true);
    try {
      const [owner, name] = repo.full_name.split('/');
      await deleteRepository(owner, name);
      onBack();
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to delete repository');
    } finally {
      setDeleting(false);
    }
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box>
        <Button startIcon={<ArrowBackIcon />} onClick={onBack} sx={{ mb: 2 }}>
          Back to List
        </Button>
        <Alert severity="error">{error}</Alert>
      </Box>
    );
  }

  const [owner, name] = repo.full_name.split('/');

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <Button startIcon={<ArrowBackIcon />} onClick={onBack}>
          Back
        </Button>
        <Box sx={{ flex: 1 }} />
        <Button
          variant="outlined"
          startIcon={<FolderIcon />}
          onClick={onViewFiles}
          sx={{ mr: 1 }}
        >
          Browse Files
        </Button>
        <Button
          variant="outlined"
          href={repo.html_url}
          target="_blank"
          rel="noopener noreferrer"
          startIcon={<OpenInNewIcon />}
          sx={{ mr: 1 }}
        >
          View on Gitea
        </Button>
        <Tooltip title="Delete Repository">
          <IconButton
            color="error"
            onClick={handleDelete}
            disabled={deleting}
          >
            {deleting ? <CircularProgress size={24} /> : <DeleteIcon />}
          </IconButton>
        </Tooltip>
      </Box>

      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'start', mb: 2 }}>
            <CodeIcon sx={{ mr: 1, mt: 0.5, color: 'text.secondary' }} />
            <Box sx={{ flex: 1 }}>
              <Typography variant="h4" gutterBottom>
                {repo.name}
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                {repo.full_name}
              </Typography>
              {repo.description && (
                <Typography variant="body1" sx={{ mt: 1 }}>
                  {repo.description}
                </Typography>
              )}
            </Box>
            <Chip
              label={repo.private ? 'Private' : 'Public'}
              color={repo.private ? 'default' : 'primary'}
              variant="outlined"
            />
          </Box>

          <Divider sx={{ my: 2 }} />

          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Clone URLs
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              <Box>
                <Typography variant="caption" color="text.secondary">
                  HTTPS:
                </Typography>
                <Typography variant="body2" sx={{ fontFamily: 'monospace', ml: 1 }}>
                  {repo.clone_url}
                </Typography>
              </Box>
              <Box>
                <Typography variant="caption" color="text.secondary">
                  SSH:
                </Typography>
                <Typography variant="body2" sx={{ fontFamily: 'monospace', ml: 1 }}>
                  {repo.ssh_url}
                </Typography>
              </Box>
            </Box>
          </Box>
        </CardContent>
      </Card>

      <Grid container spacing={2}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Branches ({branches.length})
              </Typography>
              {branches.length === 0 ? (
                <Typography variant="body2" color="text.secondary">
                  No branches found
                </Typography>
              ) : (
                <List dense>
                  {branches.slice(0, 5).map((branch) => (
                    <ListItem key={branch.name}>
                      <ListItemText
                        primary={branch.name}
                        secondary={branch.commit.message || branch.commit.id.substring(0, 7)}
                      />
                    </ListItem>
                  ))}
                  {branches.length > 5 && (
                    <Typography variant="caption" color="text.secondary" sx={{ pl: 2 }}>
                      ... and {branches.length - 5} more
                    </Typography>
                  )}
                </List>
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <HistoryIcon sx={{ mr: 1 }} />
                <Typography variant="h6">
                  Recent Commits
                </Typography>
              </Box>
              {commits.length === 0 ? (
                <Typography variant="body2" color="text.secondary">
                  No commits found
                </Typography>
              ) : (
                <List dense>
                  {commits.map((commit) => (
                    <ListItem key={commit.id}>
                      <ListItemText
                        primary={
                          <Typography variant="body2" noWrap>
                            {commit.message.split('\n')[0]}
                          </Typography>
                        }
                        secondary={
                          <Typography variant="caption" color="text.secondary">
                            {commit.author.name} • {new Date(commit.author.date).toLocaleDateString()}
                          </Typography>
                        }
                      />
                    </ListItem>
                  ))}
                </List>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

