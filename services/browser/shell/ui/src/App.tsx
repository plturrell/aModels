import { useState } from 'react';
import { ThemeProvider, CssBaseline, Typography } from "@mui/material";
import { WorkbenchLayout } from "./components/WorkbenchLayout";
import { SessionsPanel } from "./components/SessionsPanel";
import { AgentLogPanel } from "./components/AgentLogPanel";
import { Canvas } from "./components/Canvas";
import { CommandPalette } from "./components/CommandPalette";
import theme from "./theme";

function App() {
  const [sessions, setSessions] = useState<any[]>([]);
  const [activeSession, setActiveSession] = useState<any>(null);

  const handleCommand = (command: string, data: any) => {
    const newSession = { id: Date.now(), command, data };
    setSessions([...sessions, newSession]);
    setActiveSession(newSession);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <CommandPalette onCommand={handleCommand} />
      <WorkbenchLayout
        sessions={<SessionsPanel sessions={sessions} onSelect={setActiveSession} />}
        canvas={<Canvas session={activeSession} />}
        agentLog={<AgentLogPanel session={activeSession} />}
      />
    </ThemeProvider>
  );
}

export default App;
