/**
 * Keyboard Shortcuts Help Dialog
 * 
 * Shows all available keyboard shortcuts to the user
 */

import {
  Dialog,
  DialogTitle,
  DialogContent,
  Table,
  TableBody,
  TableRow,
  TableCell,
  Chip,
  Typography,
  IconButton,
  Box,
  Divider,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import { GLOBAL_SHORTCUTS, formatShortcut } from '../../hooks/useGlobalShortcuts';

interface ShortcutsDialogProps {
  open: boolean;
  onClose: () => void;
}

export function ShortcutsDialog({ open, onClose }: ShortcutsDialogProps) {
  const shortcutGroups = {
    'Navigation': [
      { key: 'Mod+1', description: 'Navigate to Home' },
      { key: 'Mod+2', description: 'Navigate to Graph' },
      { key: 'Mod+3', description: 'Navigate to Extract' },
      { key: 'Mod+4', description: 'Navigate to Training' },
      { key: 'Mod+5', description: 'Navigate to Postgres' },
      { key: 'Mod+6', description: 'Navigate to LocalAI' },
      { key: 'Mod+7', description: 'Navigate to DMS' },
      { key: 'Mod+8', description: 'Navigate to SAP' },
    ],
    'Actions': [
      { key: 'Mod+K', description: 'Open Command Palette' },
      { key: 'Escape', description: 'Close Modal/Dialog' },
      { key: '?', description: 'Show this help dialog' },
    ],
  };

  return (
    <Dialog 
      open={open} 
      onClose={onClose}
      maxWidth="sm"
      fullWidth
      PaperProps={{
        sx: { borderRadius: 2 }
      }}
    >
      <DialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Typography variant="h6" component="div">
            Keyboard Shortcuts
          </Typography>
          <IconButton
            aria-label="close"
            onClick={onClose}
            sx={{ color: 'text.secondary' }}
          >
            <CloseIcon />
          </IconButton>
        </Box>
      </DialogTitle>
      <DialogContent>
        {Object.entries(shortcutGroups).map(([groupName, shortcuts], groupIndex) => (
          <Box key={groupName} sx={{ mb: groupIndex < Object.keys(shortcutGroups).length - 1 ? 3 : 0 }}>
            <Typography 
              variant="subtitle2" 
              color="text.secondary" 
              sx={{ mb: 1.5, textTransform: 'uppercase', letterSpacing: 0.5 }}
            >
              {groupName}
            </Typography>
            <Table size="small">
              <TableBody>
                {shortcuts.map(({ key, description }) => (
                  <TableRow key={key} sx={{ '&:last-child td': { border: 0 } }}>
                    <TableCell sx={{ border: 0, py: 1 }}>
                      <Chip 
                        label={formatShortcut(key)} 
                        size="small" 
                        sx={{ 
                          fontFamily: 'monospace',
                          fontWeight: 600,
                          minWidth: 60,
                        }} 
                      />
                    </TableCell>
                    <TableCell sx={{ border: 0, py: 1 }}>
                      <Typography variant="body2">
                        {description}
                      </Typography>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Box>
        ))}
        
        <Divider sx={{ my: 2 }} />
        
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', textAlign: 'center' }}>
          Press <strong>?</strong> anytime to show this dialog
        </Typography>
      </DialogContent>
    </Dialog>
  );
}
