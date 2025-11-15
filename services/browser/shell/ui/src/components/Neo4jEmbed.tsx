import React, { useState } from 'react';
import { 
  Dialog, 
  DialogTitle, 
  DialogContent, 
  IconButton,
  Box,
  Alert
} from '@mui/material';
import { Close as CloseIcon } from '@mui/icons-material';

interface Neo4jEmbedProps {
  open: boolean;
  onClose: () => void;
}

const Neo4jEmbed: React.FC<Neo4jEmbedProps> = ({ open, onClose }) => {
  const [iframeError, setIframeError] = useState(false);
  
  return (
    <Dialog 
      open={open} 
      onClose={onClose}
      maxWidth="xl"
      fullWidth
      PaperProps={{
        sx: { height: '90vh' }
      }}
    >
      <DialogTitle sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        Neo4j Browser
        <IconButton onClick={onClose} size="small">
          <CloseIcon />
        </IconButton>
      </DialogTitle>
      
      <DialogContent sx={{ p: 0 }}>
        {iframeError ? (
          <Alert severity="warning" sx={{ m: 2 }}>
            Unable to load Neo4j Browser. Please ensure Neo4j is running on port 7474.
            <br />
            <a href="http://localhost:7474" target="_blank" rel="noopener noreferrer">
              Open in new tab →
            </a>
          </Alert>
        ) : (
          <Box sx={{ width: '100%', height: '100%' }}>
            <iframe
              src="http://localhost:7474/browser"
              style={{
                width: '100%',
                height: '100%',
                border: 'none'
              }}
              title="Neo4j Browser"
              onError={() => setIframeError(true)}
            />
          </Box>
        )}
      </DialogContent>
    </Dialog>
  );
};

export default Neo4jEmbed;
