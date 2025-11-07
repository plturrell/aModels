import { useState, useEffect } from "react";
import {
  Box,
  Typography,
  Tabs,
  Tab,
  TextField,
  Button,
  Alert,
  CircularProgress,
  Grid,
  Card,
  CardContent,
  Chip,
  Stack
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import DashboardIcon from '@mui/icons-material/Dashboard';
import BarChartIcon from '@mui/icons-material/BarChart';
import TimelineIcon from '@mui/icons-material/Timeline';
import SearchIcon from '@mui/icons-material/Search';

import { Panel } from "../../components/Panel";
import { 
  getProcessingStatus, 
  getProcessingResults, 
  getIntelligence,
  getRequestHistory,
  processDocuments,
  type ProcessingRequest
} from "../../api/perplexity";

import { ProcessingView } from "./views/ProcessingView";
import { ResultsView } from "./views/ResultsView";
import { AnalyticsView } from "./views/AnalyticsView";

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`perplexity-tabpanel-${index}`}
      aria-labelledby={`perplexity-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ pt: 3 }}>{children}</Box>}
    </div>
  );
}

/**
 * Perplexity Dashboard Module
 * 
 * Native React integration - fully integrated with Browser Shell.
 * Uses Observable Plot for visualizations and Material-UI for consistency.
 * Designed with the Jobs & Ive lens: Simplicity, Beauty, Intuition, Delight.
 */
export function PerplexityModule() {
  const [activeTab, setActiveTab] = useState(0);
  const [requestId, setRequestId] = useState<string>("");
  const [newQuery, setNewQuery] = useState<string>("");
  const [processing, setProcessing] = useState(false);
  const [processError, setProcessError] = useState<string | null>(null);

  // Fetch data based on active tab
  // Note: Using direct API calls instead of useApiData to support custom base URL
  const [statusData, setStatusData] = useState<ProcessingRequest | null>(null);
  const [statusLoading, setStatusLoading] = useState(false);
  const [statusError, setStatusError] = useState<Error | null>(null);

  const [resultsData, setResultsData] = useState<any>(null);
  const [resultsLoading, setResultsLoading] = useState(false);
  const [resultsError, setResultsError] = useState<Error | null>(null);

  const [intelligenceData, setIntelligenceData] = useState<any>(null);
  const [intelligenceLoading, setIntelligenceLoading] = useState(false);
  const [intelligenceError, setIntelligenceError] = useState<Error | null>(null);

  const [historyData, setHistoryData] = useState<any>(null);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState<Error | null>(null);

  // Fetch status
  useEffect(() => {
    if (!requestId) {
      setStatusData(null);
      return;
    }
    setStatusLoading(true);
    setStatusError(null);
    getProcessingStatus(requestId)
      .then(setStatusData)
      .catch(err => setStatusError(err instanceof Error ? err : new Error(String(err))))
      .finally(() => setStatusLoading(false));
  }, [requestId]);

  // Fetch results
  useEffect(() => {
    if (!requestId) {
      setResultsData(null);
      return;
    }
    setResultsLoading(true);
    setResultsError(null);
    getProcessingResults(requestId)
      .then(setResultsData)
      .catch(err => setResultsError(err instanceof Error ? err : new Error(String(err))))
      .finally(() => setResultsLoading(false));
  }, [requestId]);

  // Fetch intelligence
  useEffect(() => {
    if (!requestId) {
      setIntelligenceData(null);
      return;
    }
    setIntelligenceLoading(true);
    setIntelligenceError(null);
    getIntelligence(requestId)
      .then(setIntelligenceData)
      .catch(err => setIntelligenceError(err instanceof Error ? err : new Error(String(err))))
      .finally(() => setIntelligenceLoading(false));
  }, [requestId]);

  // Fetch history
  useEffect(() => {
    setHistoryLoading(true);
    setHistoryError(null);
    getRequestHistory({ limit: 50 })
      .then(setHistoryData)
      .catch(err => setHistoryError(err instanceof Error ? err : new Error(String(err))))
      .finally(() => setHistoryLoading(false));
  }, []);

  const handleProcess = async () => {
    if (!newQuery.trim()) return;
    
    setProcessing(true);
    setProcessError(null);
    
    try {
      const result = await processDocuments({
        query: newQuery.trim(),
        limit: 10,
        async: true
      });
      
      setRequestId(result.request_id);
      setNewQuery("");
      setActiveTab(0); // Switch to Processing tab
    } catch (err) {
      setProcessError(err instanceof Error ? err.message : "Failed to process documents");
    } finally {
      setProcessing(false);
    }
  };

  return (
    <Box sx={{ width: '100%', height: '100%', overflow: 'auto' }}>
      <Panel 
        title="Perplexity Dashboard" 
        subtitle="Beautiful, interactive visualization of processing results"
      >
        <Typography variant="body2" paragraph sx={{ color: 'text.secondary', mb: 2 }}>
          Explore processed documents, visualize relationships, analyze trends, and discover patterns 
          in your Perplexity data. Fully integrated with Browser Shell for a seamless experience.
        </Typography>

        {/* New Query Input */}
        <Stack direction="row" spacing={2} sx={{ mb: 3 }}>
          <TextField
            fullWidth
            label="New Query"
            placeholder="Enter a query to process..."
            value={newQuery}
            onChange={(e) => setNewQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                void handleProcess();
              }
            }}
            disabled={processing}
            size="small"
          />
          <Button
            variant="contained"
            startIcon={processing ? <CircularProgress size={16} /> : <SendIcon />}
            onClick={handleProcess}
            disabled={processing || !newQuery.trim()}
            sx={{ minWidth: 120 }}
          >
            Process
          </Button>
        </Stack>

        {processError && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setProcessError(null)}>
            {processError}
          </Alert>
        )}

        {/* Request ID Input */}
        <Stack direction="row" spacing={2} sx={{ mb: 3 }}>
          <TextField
            fullWidth
            label="Request ID"
            placeholder="Enter request ID to view..."
            value={requestId}
            onChange={(e) => setRequestId(e.target.value)}
            size="small"
            helperText="Enter a request ID to view processing status and results"
          />
        </Stack>

        {/* Tabs */}
        <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
          <Tabs 
            value={activeTab} 
            onChange={(_, newValue) => setActiveTab(newValue)}
            aria-label="Perplexity dashboard tabs"
          >
            <Tab 
              icon={<TimelineIcon />} 
              iconPosition="start"
              label="Processing" 
              id="perplexity-tab-0"
            />
            <Tab 
              icon={<DashboardIcon />} 
              iconPosition="start"
              label="Results" 
              id="perplexity-tab-1"
            />
            <Tab 
              icon={<BarChartIcon />} 
              iconPosition="start"
              label="Analytics" 
              id="perplexity-tab-2"
            />
            <Tab 
              icon={<SearchIcon />} 
              iconPosition="start"
              label="Search" 
              id="perplexity-tab-3"
            />
          </Tabs>
        </Box>

        {/* Tab Panels */}
        <TabPanel value={activeTab} index={0}>
          <ProcessingView 
            requestId={requestId}
            data={statusData}
            loading={statusLoading}
            error={statusError}
          />
        </TabPanel>

        <TabPanel value={activeTab} index={1}>
          <ResultsView
            requestId={requestId}
            resultsData={resultsData}
            intelligenceData={intelligenceData}
            loading={resultsLoading || intelligenceLoading}
            error={resultsError || intelligenceError}
          />
        </TabPanel>

        <TabPanel value={activeTab} index={2}>
          <AnalyticsView
            historyData={historyData}
            loading={historyLoading}
            error={historyError}
          />
        </TabPanel>

        <TabPanel value={activeTab} index={3}>
          <Box>
            <Typography variant="h6" gutterBottom>
              Search Documents
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              Search functionality coming soon...
            </Typography>
          </Box>
        </TabPanel>
      </Panel>
    </Box>
  );
}
