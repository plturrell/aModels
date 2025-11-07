import React, { useMemo } from "react";
import { Box, List, ListItem, ListItemButton, ListItemIcon, ListItemText, Typography, Divider } from '@mui/material';
import HomeIcon from '@mui/icons-material/Home';
import ChatIcon from '@mui/icons-material/Chat';
import DescriptionIcon from '@mui/icons-material/Description';
import AccountTreeIcon from '@mui/icons-material/AccountTree';
import InsightsIcon from '@mui/icons-material/Insights';
import SearchIcon from '@mui/icons-material/Search';
import DashboardIcon from '@mui/icons-material/Dashboard';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';

import { useShellStore, type ShellModuleId } from "../state/useShellStore";

const NAV_ITEMS: Array<{ id: ShellModuleId; label: string; description: string; icon: React.ElementType }> = [
  {
    id: "localai",
    label: "LocalAI",
    description: "Chat on top of vendored models",
    icon: ChatIcon
  },
  {
    id: "dms",
    label: "Documents",
    description: "Curated library with relationships",
    icon: DescriptionIcon
  },
  {
    id: "flows",
    label: "Flows",
    description: "AgentFlow / LangFlow orchestration",
    icon: AccountTreeIcon
  },
  {
    id: "telemetry",
    label: "Telemetry",
    description: "Latency and usage instrumentation",
    icon: InsightsIcon
  },
  {
    id: "search",
    label: "Search",
    description: "Semantic search across content",
    icon: SearchIcon
  },
  {
    id: "perplexity",
    label: "Perplexity",
    description: "Processing results & analytics",
    icon: DashboardIcon
  },
  {
    id: "home",
    label: "Home",
    description: "Overview and quick links",
    icon: HomeIcon
  }
];

export function NavPanel() {
  const activeModule = useShellStore((state) => state.activeModule);
  const setActiveModule = useShellStore((state) => state.setActiveModule);
  const theme = useShellStore((state) => state.theme);
  const toggleTheme = useShellStore((state) => state.toggleTheme);

  const orderedItems = useMemo(() => {
    const primary = NAV_ITEMS.filter((item) => item.id !== "home");
    const home = NAV_ITEMS.find((item) => item.id === "home");
    return home ? [...primary, home] : primary;
  }, []);

  return (
    <Box sx={{ overflow: 'auto', height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ p: 2, textAlign: 'center' }}>
        <Typography variant="h6" component="div">
          aModels Shell
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Chromium host for documents, flows, and telemetry
        </Typography>
      </Box>
      <Divider />
      <List>
        {orderedItems.map((item) => (
          <ListItem key={item.id} disablePadding>
            <ListItemButton
              selected={activeModule === item.id}
              onClick={() => setActiveModule(item.id)}
            >
              <ListItemIcon>
                <item.icon />
              </ListItemIcon>
              <ListItemText
                primary={item.label}
                secondary={item.description}
              />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
      <Box sx={{ mt: 'auto', p: 2 }}>
        <ListItem disablePadding>
          <ListItemButton onClick={toggleTheme}>
            <ListItemIcon>
              {theme === "dark" ? <Brightness7Icon /> : <Brightness4Icon />} 
            </ListItemIcon>
            <ListItemText primary={`Theme: ${theme === "dark" ? "Dark" : "Light"}`} />
          </ListItemButton>
        </ListItem>
      </Box>
    </Box>
  );
}