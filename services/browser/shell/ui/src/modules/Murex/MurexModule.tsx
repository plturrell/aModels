/**
 * Murex Module
 * Main component for Murex trade processing and ETL to SAP
 */

import React, { useState, useEffect } from "react";
import {
  Box,
  Typography,
  Tabs,
  Tab,
  TextField,
  Button,
  Alert,
  CircularProgress
} from "@mui/material";
import SendIcon from "@mui/icons-material/Send";
import DashboardIcon from "@mui/icons-material/Dashboard";
import BarChartIcon from "@mui/icons-material/BarChart";
import TimelineIcon from "@mui/icons-material/Timeline";
import SwapHorizIcon from "@mui/icons-material/SwapHoriz";
import { Panel } from "../../components/Panel";
import {
  getMurexProcessingStatus,
  getMurexProcessingResults,
  getMurexIntelligence,
  getMurexRequestHistory,
  processMurexTrades,
  type MurexProcessingRequest,
  type MurexTrade,
  type MurexRequestIntelligence,
  type MurexRequestHistory
} from "../../api/murex";
import { ProcessingView } from "./views/ProcessingView";
import { ResultsView } from "./views/ResultsView";
import { AnalyticsView } from "./views/AnalyticsView";
import { ETLView } from "./views/ETLView";

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
      id={`murex-tabpanel-${index}`}
      aria-labelledby={`murex-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

export function MurexModule() {
  const [activeTab, setActiveTab] = useState(0);
  const [requestId, setRequestId] = useState("");
  const [processing, setProcessing] = useState(false);
  const [processError, setProcessError] = useState<string | null>(null);

  // Processing view state
  const [statusData, setStatusData] = useState<MurexProcessingRequest | null>(null);
  const [statusLoading, setStatusLoading] = useState(false);
  const [statusError, setStatusError] = useState<string | null>(null);

  // Results view state
  const [resultsTrades, setResultsTrades] = useState<MurexTrade[] | null>(null);
  const [resultsIntelligence, setResultsIntelligence] = useState<MurexRequestIntelligence | null>(null);
  const [resultsLoading, setResultsLoading] = useState(false);
  const [resultsError, setResultsError] = useState<string | null>(null);

  // Analytics view state
  const [historyData, setHistoryData] = useState<MurexRequestHistory | null>(null);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState<string | null>(null);

  // Process form state
  const [tableName, setTableName] = useState("trades");
  const [filters, setFilters] = useState("");

  // Load status when requestId changes
  useEffect(() => {
    if (requestId && activeTab === 0) {
      loadStatus();
    }
  }, [requestId, activeTab]);

  // Load results when requestId changes
  useEffect(() => {
    if (requestId && activeTab === 1) {
      loadResults();
    }
  }, [requestId, activeTab]);

  // Load history for analytics
  useEffect(() => {
    if (activeTab === 2) {
      loadHistory();
    }
  }, [activeTab]);

  const loadStatus = async () => {
    if (!requestId) return;
    setStatusLoading(true);
    setStatusError(null);
    try {
      const status = await getMurexProcessingStatus(requestId);
      setStatusData(status);
    } catch (error) {
      setStatusError(error instanceof Error ? error.message : "Failed to load status");
    } finally {
      setStatusLoading(false);
    }
  };

  const loadResults = async () => {
    if (!requestId) return;
    setResultsLoading(true);
    setResultsError(null);
    try {
      const results = await getMurexProcessingResults(requestId);
      setResultsTrades(results.documents || []);
      setResultsIntelligence(results.intelligence || null);
    } catch (error) {
      setResultsError(error instanceof Error ? error.message : "Failed to load results");
    } finally {
      setResultsLoading(false);
    }
  };

  const loadHistory = async () => {
    setHistoryLoading(true);
    setHistoryError(null);
    try {
      const history = await getMurexRequestHistory({ limit: 50 });
      setHistoryData(history);
    } catch (error) {
      setHistoryError(error instanceof Error ? error.message : "Failed to load history");
    } finally {
      setHistoryLoading(false);
    }
  };

  const handleProcessTrades = async () => {
    if (!tableName) {
      setProcessError("Table name is required");
      return;
    }

    setProcessing(true);
    setProcessError(null);

    try {
      const params: any = {
        table: tableName,
        async: true
      };

      // Parse filters if provided
      if (filters.trim()) {
        try {
          params.filters = JSON.parse(filters);
        } catch {
          setProcessError("Invalid JSON in filters");
          setProcessing(false);
          return;
        }
      }

      const result = await processMurexTrades(params);
      setRequestId(result.request_id);
      setActiveTab(0); // Switch to processing view
    } catch (error) {
      setProcessError(error instanceof Error ? error.message : "Failed to process trades");
    } finally {
      setProcessing(false);
    }
  };

  return (
    <Panel title="Murex Trade Processing">
      <Box sx={{ width: "100%" }}>
        <Typography variant="h4" gutterBottom>
          Murex Trade Processing & ETL to SAP
        </Typography>

        <Box sx={{ borderBottom: 1, borderColor: "divider", mb: 2 }}>
          <Tabs value={activeTab} onChange={(_, newValue) => setActiveTab(newValue)}>
            <Tab icon={<TimelineIcon />} iconPosition="start" label="Processing" />
            <Tab icon={<DashboardIcon />} iconPosition="start" label="Results" />
            <Tab icon={<BarChartIcon />} iconPosition="start" label="Analytics" />
            <Tab icon={<SwapHorizIcon />} iconPosition="start" label="ETL to SAP" />
          </Tabs>
        </Box>

        <TabPanel value={activeTab} index={0}>
          <ProcessingView
            requestId={requestId}
            statusData={statusData}
            loading={statusLoading}
            error={statusError}
            onRequestIdChange={setRequestId}
            onRefresh={loadStatus}
          />
        </TabPanel>

        <TabPanel value={activeTab} index={1}>
          <ResultsView
            requestId={requestId}
            trades={resultsTrades}
            intelligence={resultsIntelligence}
            loading={resultsLoading}
            error={resultsError}
            onRequestIdChange={setRequestId}
            onRefresh={loadResults}
          />
        </TabPanel>

        <TabPanel value={activeTab} index={2}>
          <AnalyticsView
            history={historyData}
            loading={historyLoading}
            error={historyError}
            onRefresh={loadHistory}
          />
        </TabPanel>

        <TabPanel value={activeTab} index={3}>
          <ETLView />
        </TabPanel>

        {/* Process Form */}
        <Box sx={{ mt: 4, p: 2, bgcolor: "background.paper", borderRadius: 1 }}>
          <Typography variant="h6" gutterBottom>
            Process Murex Trades
          </Typography>
          <Box sx={{ display: "flex", flexDirection: "column", gap: 2, mt: 2 }}>
            <TextField
              label="Table Name"
              value={tableName}
              onChange={(e) => setTableName(e.target.value)}
              placeholder="trades"
              required
            />
            <TextField
              label="Filters (JSON, optional)"
              value={filters}
              onChange={(e) => setFilters(e.target.value)}
              placeholder='{"status": "executed", "trade_date_from": "2024-01-01"}'
              multiline
              rows={3}
            />
            {processError && <Alert severity="error">{processError}</Alert>}
            <Button
              variant="contained"
              startIcon={processing ? <CircularProgress size={20} /> : <SendIcon />}
              onClick={handleProcessTrades}
              disabled={processing || !tableName}
            >
              {processing ? "Processing..." : "Process Trades"}
            </Button>
          </Box>
        </Box>
      </Box>
    </Panel>
  );
}

