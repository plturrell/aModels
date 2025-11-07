import { Box, Typography, List, ListItem, ListItemButton, ListItemText, Paper } from '@mui/material';
import { useShellStore } from "../../state/useShellStore";
import { Panel } from "../../components/Panel";

const quickLinks = [
  {
    label: "SGMI Control-M overview",
    description: "Inspect the latest waits, triggers, and dummy jobs",
    targetModule: "localai" as const
  },
  {
    label: "Document library",
    description: "Review the latest uploads and relationships",
    targetModule: "dms" as const
  },
  {
    label: "Flow orchestrator",
    description: "Sync and run AgentFlow / LangFlow pipelines",
    targetModule: "flows" as const
  },
  {
    label: "Telemetry dashboard",
    description: "Track response latency and token usage",
    targetModule: "telemetry" as const
  }
];

export function HomeModule() {
  const setActiveModule = useShellStore((state) => state.setActiveModule);

  return (
    <Box>
      <Panel title="Welcome" subtitle="Your aModels Chromium workspace">
        <Typography variant="body1" paragraph>
          Use the navigation to explore SGMI Control-M data, inspect LocalAI models, and monitor
          telemetry captured from live interactions. Everything you see is sourced directly from the
          repository—no mock data.
        </Typography>
      </Panel>

      <Panel title="Quick Links" subtitle="Jump into the modules">
        <List>
          {quickLinks.map((link) => (
            <ListItem key={link.label} disablePadding>
              <ListItemButton onClick={() => setActiveModule(link.targetModule)}>
                <ListItemText
                  primary={link.label}
                  secondary={link.description}
                  primaryTypographyProps={{
                    variant: "body1",
                    fontWeight: 500
                  }}
                  secondaryTypographyProps={{
                    variant: "body2",
                    color: "text.secondary"
                  }}
                />
              </ListItemButton>
            </ListItem>
          ))}
        </List>
      </Panel>
    </Box>
  );
}
