/**
 * Create Repository View
 * 
 * Form to create a new Gitea repository
 */

import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  TextField,
  Button,
  FormControlLabel,
  Switch,
  Typography,
  Alert,
  CircularProgress,
} from '@mui/material';
import { createRepository, type CreateRepositoryRequest, type GiteaRepository } from '../../../api/gitea';

interface CreateRepositoryProps {
  onRepositoryCreated: (repo: GiteaRepository) => void;
}

export function CreateRepository({ onRepositoryCreated }: CreateRepositoryProps) {
  const [formData, setFormData] = useState<CreateRepositoryRequest>({
    name: '',
    description: '',
    private: false,
    auto_init: true,
    readme: '',
  });
  const [owner, setOwner] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<GiteaRepository | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setSuccess(null);

    if (!formData.name) {
      setError('Repository name is required');
      setLoading(false);
      return;
    }

    try {
      const repo = await createRepository({
        ...formData,
        owner: owner || undefined,
      });
      setSuccess(repo);
      setTimeout(() => {
        onRepositoryCreated(repo);
      }, 1500);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create repository');
    } finally {
      setLoading(false);
    }
  };

  if (success) {
    return (
      <Box>
        <Alert severity="success" sx={{ mb: 2 }}>
          Repository created successfully!
        </Alert>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              {success.name}
            </Typography>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              {success.full_name}
            </Typography>
            {success.description && (
              <Typography variant="body2" sx={{ mt: 1 }}>
                {success.description}
              </Typography>
            )}
            <Box sx={{ mt: 2 }}>
              <Button
                variant="outlined"
                href={success.html_url}
                target="_blank"
                rel="noopener noreferrer"
              >
                View on Gitea
              </Button>
            </Box>
          </CardContent>
        </Card>
      </Box>
    );
  }

  return (
    <Box sx={{ maxWidth: 600, mx: 'auto' }}>
      <Card>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            Create New Repository
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            Create a new Gitea repository with a GitHub-like experience
          </Typography>

          <form onSubmit={handleSubmit}>
            <TextField
              fullWidth
              label="Owner (optional, leave empty for user repos)"
              value={owner}
              onChange={(e) => setOwner(e.target.value)}
              margin="normal"
              helperText="Leave empty to create under your user account"
            />

            <TextField
              fullWidth
              required
              label="Repository Name"
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              margin="normal"
              helperText="Repository name must be unique"
            />

            <TextField
              fullWidth
              label="Description"
              value={formData.description}
              onChange={(e) => setFormData({ ...formData, description: e.target.value })}
              margin="normal"
              multiline
              rows={3}
              helperText="Brief description of the repository"
            />

            <FormControlLabel
              control={
                <Switch
                  checked={formData.private}
                  onChange={(e) => setFormData({ ...formData, private: e.target.checked })}
                />
              }
              label="Private Repository"
              sx={{ mt: 2, mb: 1 }}
            />

            <FormControlLabel
              control={
                <Switch
                  checked={formData.auto_init}
                  onChange={(e) => setFormData({ ...formData, auto_init: e.target.checked })}
                />
              }
              label="Initialize with README"
              sx={{ mb: 2 }}
            />

            {formData.auto_init && (
              <TextField
                fullWidth
                label="README Content"
                value={formData.readme}
                onChange={(e) => setFormData({ ...formData, readme: e.target.value })}
                margin="normal"
                multiline
                rows={5}
                helperText="Initial README.md content (optional)"
              />
            )}

            {error && (
              <Alert severity="error" sx={{ mt: 2 }}>
                {error}
              </Alert>
            )}

            <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
              <Button
                type="submit"
                variant="contained"
                disabled={loading}
                startIcon={loading ? <CircularProgress size={20} /> : null}
              >
                {loading ? 'Creating...' : 'Create Repository'}
              </Button>
              <Button
                variant="outlined"
                onClick={() => {
                  setFormData({
                    name: '',
                    description: '',
                    private: false,
                    auto_init: true,
                    readme: '',
                  });
                  setOwner('');
                  setError(null);
                }}
              >
                Reset
              </Button>
            </Box>
          </form>
        </CardContent>
      </Card>
    </Box>
  );
}

