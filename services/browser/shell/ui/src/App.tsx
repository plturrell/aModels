import { useState } from 'react';
import { ThemeProvider, CssBaseline, Typography } from "@mui/material";
import { WorkbenchLayout } from "./components/WorkbenchLayout";
import { SessionsPanel } from "./components/SessionsPanel";
import { AgentLogPanel } from "./components/AgentLogPanel";
import { Canvas } from "./components/Canvas";
import { CommandPalette } from "./components/CommandPalette";
import { ErrorBoundary } from "./components/ErrorBoundary";
import theme from "./theme-sap";

function App() {
  const [sessions, setSessions] = useState<any[]>([]);
  const [activeSession, setActiveSession] = useState<any>(null);

  const handleCommand = (command: string, data: any) => {
    const newSession = { id: Date.now(), command, data };
    setSessions([...sessions, newSession]);
    setActiveSession(newSession);
  };

  const handleError = (error: Error) => {
    console.error('Application error:', error);
    // TODO: Send to error tracking service
  };

  return (
    <ErrorBoundary onError={handleError}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <CommandPalette onCommand={handleCommand} />
        <ErrorBoundary>
          <WorkbenchLayout
            sessions={<SessionsPanel sessions={sessions} onSelect={setActiveSession} />}
            canvas={
              <ErrorBoundary>
                <Canvas session={activeSession} />
              </ErrorBoundary>
            }
            agentLog={
              <ErrorBoundary>
                <AgentLogPanel session={activeSession} />
              </ErrorBoundary>
            }
          />
        </ErrorBoundary>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;
