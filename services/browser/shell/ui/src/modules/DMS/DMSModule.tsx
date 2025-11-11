/**
 * DMS Module - Ruthlessly Simplified
 * One action: Process documents
 * One view: Status + Results
 */

import React, { useState, useEffect } from "react";
import {
  Box,
  Typography,
  TextField,
  Button,
  CircularProgress,
  Card,
  CardContent,
  Grid
} from "@mui/material";
import SendIcon from "@mui/icons-material/Send";
import { Panel } from "../../components/Panel";
import {
  getDMSResults,
  processDMSDocuments,
  type DMSProcessedDocument
} from "../../api/dms";

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
      id={`dms-tabpanel-${index}`}
      aria-labelledby={`dms-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

export function DMSModule() {
  const [activeTab, setActiveTab] = useState(0);
  const [requestId, setRequestId] = useState("");
  const [documentId, setDocumentId] = useState("");
  const [processing, setProcessing] = useState(false);
  const [processError, setProcessError] = useState<string | null>(null);

  // Processing view state
  const [statusData, setStatusData] = useState<DMSProcessingRequest | null>(null);
  const [statusLoading, setStatusLoading] = useState(false);
  const [statusError, setStatusError] = useState<string | null>(null);

  // Results view state
  const [resultsDocuments, setResultsDocuments] = useState<DMSProcessedDocument[] | null>(null);
  const [resultsIntelligence, setResultsIntelligence] = useState<DMSRequestIntelligence | null>(null);
  const [resultsLoading, setResultsLoading] = useState(false);
  const [resultsError, setResultsError] = useState<string | null>(null);

  // Analytics view state
  const [historyData, setHistoryData] = useState<DMSRequestHistory | null>(null);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState<string | null>(null);

  // Documents view state
  const [documentsData, setDocumentsData] = useState<DMSDocument[] | null>(null);
  const [documentsLoading, setDocumentsLoading] = useState(false);
  const [documentsError, setDocumentsError] = useState<string | null>(null);

  // Load history on mount
  useEffect(() => {
    loadHistory();
  }, []);

  // Load status when requestId changes
  useEffect(() => {
    if (requestId && activeTab === 0) {
      loadStatus(requestId);
    }
  }, [requestId, activeTab]);

  // Load results when requestId changes
  useEffect(() => {
    if (requestId && activeTab === 1) {
      loadResults(requestId);
    }
  }, [requestId, activeTab]);

  // Load documents on mount or when tab changes
  useEffect(() => {
    if (activeTab === 3) {
      loadDocuments();
    }
  }, [activeTab]);

  const loadStatus = async (id: string) => {
    setStatusLoading(true);
    setStatusError(null);
    try {
      const data = await getDMSStatus(id);
      setStatusData(data);
    } catch (err) {
      setStatusError(err instanceof Error ? err.message : "Failed to load status");
    } finally {
      setStatusLoading(false);
    }
  };

  const loadResults = async (id: string) => {
    setResultsLoading(true);
    setResultsError(null);
    try {
      const [results, intelligence] = await Promise.all([
        getDMSResults(id),
        getDMSIntelligence(id).catch(() => null)
      ]);
      setResultsDocuments(results.documents || []);
      setResultsIntelligence(intelligence?.intelligence || null);
    } catch (err) {
      setResultsError(err instanceof Error ? err.message : "Failed to load results");
    } finally {
      setResultsLoading(false);
    }
  };

  const loadHistory = async () => {
    setHistoryLoading(true);
    setHistoryError(null);
    try {
      const data = await getDMSHistory({ limit: 50 });
      setHistoryData(data);
    } catch (err) {
      setHistoryError(err instanceof Error ? err.message : "Failed to load history");
    } finally {
      setHistoryLoading(false);
    }
  };

  const loadDocuments = async () => {
    setDocumentsLoading(true);
    setDocumentsError(null);
    try {
      const data = await listDMSDocuments({ limit: 50 });
      setDocumentsData(data);
    } catch (err) {
      setDocumentsError(err instanceof Error ? err.message : "Failed to load documents");
    } finally {
      setDocumentsLoading(false);
    }
  };

  const handleProcess = async () => {
    if (!documentId.trim()) {
      setProcessError("Please enter a document ID");
      return;
    }

    setProcessing(true);
    setProcessError(null);

    try {
      const result = await processDMSDocuments({
        document_id: documentId.trim(),
        async: true
      });
      setRequestId(result.request_id);
      setActiveTab(0); // Switch to processing view
      // Load status after a short delay
      setTimeout(() => loadStatus(result.request_id), 1000);
    } catch (err) {
      setProcessError(err instanceof Error ? err.message : "Failed to process document");
    } finally {
      setProcessing(false);
    }
  };

  return (
    <Box sx={{ height: "100%", display: "flex", flexDirection: "column" }}>
      <Panel title="DMS Document Processing" dense>
        <Box sx={{ mb: 2 }}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Process documents through the full pipeline: OCR → Catalog → Training → LocalAI → Search
          </Typography>
        </Box>

        <Box sx={{ display: "flex", gap: 2, mb: 2 }}>
          <TextField
            label="Document ID"
            value={documentId}
            onChange={(e) => setDocumentId(e.target.value)}
            placeholder="Enter document ID from DMS"
            size="small"
            sx={{ flex: 1 }}
            onKeyPress={(e) => {
              if (e.key === "Enter") {
                handleProcess();
              }
            }}
          />
          <Button
            variant="contained"
            onClick={handleProcess}
            disabled={processing || !documentId.trim()}
            startIcon={processing ? <CircularProgress size={16} /> : <SendIcon />}
          >
            Process
          </Button>
        </Box>

        {processError && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setProcessError(null)}>
            {processError}
          </Alert>
        )}

        {requestId && (
          <Box sx={{ mb: 2 }}>
            <TextField
              label="Request ID"
              value={requestId}
              onChange={(e) => setRequestId(e.target.value)}
              size="small"
              sx={{ flex: 1 }}
              fullWidth
            />
          </Box>
        )}
      </Panel>

      <Box sx={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
        <Tabs
          value={activeTab}
          onChange={(_, newValue) => setActiveTab(newValue)}
          sx={{ borderBottom: 1, borderColor: "divider" }}
        >
          <Tab label="Processing" icon={<TimelineIcon />} iconPosition="start" />
          <Tab label="Results" icon={<DashboardIcon />} iconPosition="start" />
          <Tab label="Analytics" icon={<BarChartIcon />} iconPosition="start" />
          <Tab label="Documents" icon={<DescriptionIcon />} iconPosition="start" />
        </Tabs>

        <Box sx={{ flex: 1, overflow: "auto" }}>
          <TabPanel value={activeTab} index={0}>
            <ProcessingView
              data={statusData}
              loading={statusLoading}
              error={statusError}
            />
          </TabPanel>

          <TabPanel value={activeTab} index={1}>
            <ResultsView
              documents={resultsDocuments}
              intelligence={resultsIntelligence}
              loading={resultsLoading}
              error={resultsError}
            />
          </TabPanel>

          <TabPanel value={activeTab} index={2}>
            <AnalyticsView
              history={historyData}
              loading={historyLoading}
              error={historyError}
            />
          </TabPanel>

          <TabPanel value={activeTab} index={3}>
            <DocumentsView
              documents={documentsData}
              loading={documentsLoading}
              error={documentsError}
            />
          </TabPanel>
        </Box>
      </Box>
    </Box>
  );
}

