/**
 * Flows Module - Ruthlessly Simplified
 * One view: List of flows
 * One action: Create/Run
 */

import React, { useState } from "react";
import {
  Box,
  Typography,
  Button,
  Card,
  CardContent,
  CardActions,
  Chip
} from "@mui/material";
import AddIcon from "@mui/icons-material/Add";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";

interface Flow {
  id: string;
  name: string;
  description: string;
  status: "active" | "idle";
}

export function FlowsModule() {
  const [flows] = useState<Flow[]>([
    { id: "1", name: "Data Processing Flow", description: "Extract and process documents", status: "active" },
    { id: "2", name: "AI Analysis Flow", description: "Analyze content with AI models", status: "idle" },
    { id: "3", name: "Search Indexing Flow", description: "Index content for search", status: "idle" }
  ]);

  return (
    <Box sx={{ p: 6, maxWidth: 1200, margin: "0 auto" }}>
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 4 }}>
        <Typography variant="h4" sx={{ fontWeight: 600 }}>
          Flows
        </Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          sx={{ borderRadius: 3, textTransform: "none" }}
        >
          Create Flow
        </Button>
      </Box>

      <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
        {flows.map((flow) => (
          <Card key={flow.id} sx={{ borderRadius: 3 }}>
            <CardContent>
              <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "start", mb: 2 }}>
                <Typography variant="h6" gutterBottom>
                  {flow.name}
                </Typography>
                <Chip
                  label={flow.status}
                  color={flow.status === "active" ? "success" : "default"}
                  size="small"
                  sx={{ borderRadius: 1.5 }}
                />
              </Box>
              <Typography variant="body2" color="text.secondary">
                {flow.description}
              </Typography>
            </CardContent>
            <CardActions>
              <Button size="small" startIcon={<PlayArrowIcon />} sx={{ textTransform: "none" }}>
                Run Flow
              </Button>
            </CardActions>
          </Card>
        ))}
      </Box>

      {flows.length === 0 && (
        <Box sx={{ textAlign: "center", py: 8, color: "text.secondary" }}>
          <Typography variant="h6" gutterBottom>
            No flows yet
          </Typography>
          <Typography variant="body2">
            Create your first flow to get started
          </Typography>
        </Box>
      )}
    </Box>
  );
}
