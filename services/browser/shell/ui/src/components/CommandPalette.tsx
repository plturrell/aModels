import React, { useState, useEffect } from 'react';
import { Box, TextField, Typography, List, ListItem, ListItemButton, ListItemText, Paper, Modal } from '@mui/material';
import { unifiedSearch } from '../api/search';

interface CommandPaletteProps {
  onCommand: (command: string, data: any) => void;
}

export function CommandPalette({ onCommand }: CommandPaletteProps) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.metaKey && event.key === 'k') {
        event.preventDefault();
        setOpen((prev) => !prev);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const handleSearch = async () => {
    if (!query.trim()) return;

    const results = await unifiedSearch({ query });
    onCommand('search', results);
    setOpen(false);
  };

  return (
    <Modal open={open} onClose={() => setOpen(false)} sx={{ display: 'flex', justifyContent: 'center', alignItems: 'flex-start', pt: '20vh' }}>
      <Paper sx={{ width: '600px', p: 2 }}>
        <TextField 
          fullWidth 
          placeholder="Type a command or search..." 
          autoFocus 
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
        />
      </Paper>
    </Modal>
  );
}
