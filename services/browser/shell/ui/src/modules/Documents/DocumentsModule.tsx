/**
 * Documents Module - Ruthlessly Simplified
 * One view: Document grid
 * One action: Upload
 */

import React, { useState } from "react";
import {
  Box,
  Typography,
  Button,
  Card,
  CardContent,
  CardActions
} from "@mui/material";
import UploadIcon from "@mui/icons-material/Upload";
import DescriptionIcon from "@mui/icons-material/Description";

interface Document {
  id: string;
  title: string;
  size: string;
  date: string;
}

export function DocumentsModule() {
  const [documents] = useState<Document[]>([
    { id: "1", title: "Project Proposal.pdf", size: "2.4 MB", date: "2 hours ago" },
    { id: "2", title: "Meeting Notes.docx", size: "156 KB", date: "Yesterday" },
    { id: "3", title: "Quarterly Report.xlsx", size: "890 KB", date: "3 days ago" },
    { id: "4", title: "Design Mockups.fig", size: "5.2 MB", date: "1 week ago" }
  ]);

  return (
    <Box sx={{ p: 6, maxWidth: 1200, margin: "0 auto" }}>
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 4 }}>
        <Typography variant="h4" sx={{ fontWeight: 600 }}>
          Documents
        </Typography>
        <Button
          variant="contained"
          startIcon={<UploadIcon />}
          sx={{ borderRadius: 3, textTransform: "none" }}
        >
          Upload
        </Button>
      </Box>

      <Box sx={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 2 }}>
        {documents.map((doc) => (
          <Card key={doc.id} sx={{ borderRadius: 3, cursor: "pointer", "&:hover": { boxShadow: 4 } }}>
            <CardContent>
              <Box sx={{ display: "flex", alignItems: "start", gap: 2, mb: 2 }}>
                <DescriptionIcon sx={{ fontSize: 40, color: "primary.main" }} />
                <Box sx={{ flex: 1, minWidth: 0 }}>
                  <Typography variant="h6" noWrap sx={{ fontSize: 16 }}>
                    {doc.title}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {doc.size}
                  </Typography>
                </Box>
              </Box>
              <Typography variant="caption" color="text.secondary">
                {doc.date}
              </Typography>
            </CardContent>
            <CardActions>
              <Button size="small" sx={{ textTransform: "none" }}>
                Open
              </Button>
            </CardActions>
          </Card>
        ))}
      </Box>

      {documents.length === 0 && (
        <Box sx={{ textAlign: "center", py: 8, color: "text.secondary" }}>
          <Typography variant="h6" gutterBottom>
            No documents yet
          </Typography>
          <Typography variant="body2">
            Upload your first document to get started
          </Typography>
        </Box>
      )}
    </Box>
  );
}
