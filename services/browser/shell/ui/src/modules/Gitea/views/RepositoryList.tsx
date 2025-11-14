/**
 * Repository List View
 * 
 * Displays a list of Gitea repositories with GitHub-like UI
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  InputAdornment,
  Grid,
  Chip,
  IconButton,
  CircularProgress,
  Alert,
  Button,
  Pagination,
  Stack,
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import VisibilityIcon from '@mui/icons-material/Visibility';
import FolderIcon from '@mui/icons-material/Folder';
import DeleteIcon from '@mui/icons-material/Delete';
import RefreshIcon from '@mui/icons-material/Refresh';
import { listRepositories, deleteRepository, type GiteaRepository } from '../../../api/gitea';

interface RepositoryListProps {
  onRepositorySelect: (repo: GiteaRepository) => void;
  onViewFiles: (repo: GiteaRepository) => void;
}

export function RepositoryList({ onRepositorySelect, onViewFiles }: RepositoryListProps) {
  const [repositories, setRepositories] = useState<GiteaRepository[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [ownerFilter, setOwnerFilter] = useState('');
  const [deleting, setDeleting] = useState<number | null>(null);
  const [page, setPage] = useState(1);
  const [pageSize] = useState(12);
  const [hasMore, setHasMore] = useState(false);
  const [totalPages, setTotalPages] = useState(1);

  const loadRepositories = async (pageNum: number = page) => {
    setLoading(true);
    setError(null);
    try {
      const repos = await listRepositories(ownerFilter || undefined);
      setRepositories(repos);
      
      // Calculate pagination
      const total = repos.length;
      const pages = Math.ceil(total / pageSize);
      setTotalPages(pages);
      setHasMore(pageNum < pages);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load repositories');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadRepositories();
  }, [ownerFilter]);

  const handleDelete = async (repo: GiteaRepository, event: React.MouseEvent) => {
    event.stopPropagation();
    if (!window.confirm(`Are you sure you want to delete ${repo.full_name}?`)) {
      return;
    }

    setDeleting(repo.id);
    try {
      const [owner, name] = repo.full_name.split('/');
      await deleteRepository(owner, name);
      await loadRepositories();
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to delete repository');
    } finally {
      setDeleting(null);
    }
  };

  const filteredRepositories = repositories.filter((repo) => {
    const matchesSearch = searchQuery === '' || 
      repo.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      repo.description?.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesSearch;
  });

  // Apply pagination to filtered results
  const paginatedRepositories = filteredRepositories.slice(
    (page - 1) * pageSize,
    page * pageSize
  );
  
  const filteredTotalPages = Math.ceil(filteredRepositories.length / pageSize);

  const handlePageChange = (_event: React.ChangeEvent<unknown>, value: number) => {
    setPage(value);
    window.scrollTo({ top: 0, behavior: 'smooth' });
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
      <Box sx={{ p: 2 }}>
        <Alert severity="error" action={
          <Button color="inherit" size="small" onClick={loadRepositories}>
            Retry
          </Button>
        }>
          {error}
        </Alert>
      </Box>
    );
  }

  return (
    <Box>
      <Box sx={{ mb: 3, display: 'flex', gap: 2, alignItems: 'center' }}>
        <TextField
          placeholder="Search repositories..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          size="small"
          sx={{ flex: 1 }}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon />
              </InputAdornment>
            ),
          }}
        />
        <TextField
          placeholder="Filter by owner..."
          value={ownerFilter}
          onChange={(e) => setOwnerFilter(e.target.value)}
          size="small"
          sx={{ width: 200 }}
        />
        <IconButton onClick={loadRepositories} title="Refresh">
          <RefreshIcon />
        </IconButton>
      </Box>

      {filteredRepositories.length === 0 ? (
        <Box sx={{ textAlign: 'center', p: 4 }}>
          <Typography variant="h6" color="text.secondary">
            {repositories.length === 0 ? 'No repositories found' : 'No repositories match your search'}
          </Typography>
        </Box>
      ) : (
        <>
          <Grid container spacing={2}>
            {paginatedRepositories.map((repo) => (
            <Grid item xs={12} sm={6} md={4} key={repo.id}>
              <Card
                sx={{
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  cursor: 'pointer',
                  '&:hover': {
                    boxShadow: 4,
                  },
                }}
                onClick={() => onRepositorySelect(repo)}
              >
                <CardContent sx={{ flex: 1 }}>
                  <Box sx={{ display: 'flex', alignItems: 'start', mb: 1 }}>
                    <FolderIcon sx={{ mr: 1, color: 'text.secondary' }} />
                    <Box sx={{ flex: 1, minWidth: 0 }}>
                      <Typography variant="h6" noWrap sx={{ fontWeight: 600 }}>
                        {repo.name}
                      </Typography>
                      <Typography variant="caption" color="text.secondary" noWrap>
                        {repo.full_name}
                      </Typography>
                    </Box>
                  </Box>

                  {repo.description && (
                    <Typography
                      variant="body2"
                      color="text.secondary"
                      sx={{
                        mb: 1,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        display: '-webkit-box',
                        WebkitLineClamp: 2,
                        WebkitBoxOrient: 'vertical',
                      }}
                    >
                      {repo.description}
                    </Typography>
                  )}

                  <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', mt: 2 }}>
                    <Chip
                      label={repo.private ? 'Private' : 'Public'}
                      size="small"
                      color={repo.private ? 'default' : 'primary'}
                      variant="outlined"
                    />
                    <Box sx={{ flex: 1 }} />
                    <IconButton
                      size="small"
                      onClick={(e) => {
                        e.stopPropagation();
                        onViewFiles(repo);
                      }}
                      title="Browse files"
                    >
                      <VisibilityIcon fontSize="small" />
                    </IconButton>
                    <IconButton
                      size="small"
                      onClick={(e) => handleDelete(repo, e)}
                      disabled={deleting === repo.id}
                      title="Delete repository"
                      color="error"
                    >
                      {deleting === repo.id ? (
                        <CircularProgress size={16} />
                      ) : (
                        <DeleteIcon fontSize="small" />
                      )}
                    </IconButton>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
          </Grid>
          
          {filteredTotalPages > 1 && (
            <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
              <Stack spacing={2}>
                <Pagination
                  count={filteredTotalPages}
                  page={page}
                  onChange={handlePageChange}
                  color="primary"
                  size="large"
                  showFirstButton
                  showLastButton
                />
                <Typography variant="body2" color="text.secondary" textAlign="center">
                  Showing {((page - 1) * pageSize) + 1}-{Math.min(page * pageSize, filteredRepositories.length)} of {filteredRepositories.length} repositories
                </Typography>
              </Stack>
            </Box>
          )}
        </>
      )}
    </Box>
  );
}

