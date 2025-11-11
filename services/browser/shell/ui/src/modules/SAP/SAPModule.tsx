/**
 * SAP Extract Module
 * 
 * Comprehensive SAP Business Data Cloud extraction and visualization module
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Tabs,
  Tab,
  Alert,
  CircularProgress,
} from '@mui/material';
import CloudIcon from '@mui/icons-material/Cloud';
import StorageIcon from '@mui/icons-material/Storage';
import SchemaIcon from '@mui/icons-material/Schema';
import AccountTreeIcon from '@mui/icons-material/AccountTree';
import { Panel } from '../../components/Panel';
import { ConnectionConfigView } from './views/ConnectionConfigView';
import { DataProductsView } from './views/DataProductsView';
import { SchemaExtractionView } from './views/SchemaExtractionView';
import { GraphIntegrationView } from './views/GraphIntegrationView';
import { useSAPConnection } from './hooks/useSAPConnection';
import type { SAPBDCConnectionConfig } from '../../api/sap';

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
      id={`sap-tabpanel-${index}`}
      aria-labelledby={`sap-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ pt: 3 }}>{children}</Box>}
    </div>
  );
}

export function SAPModule() {
  const [activeTab, setActiveTab] = useState(0);
  const { connection, setConnection, isConnected, testConnection } = useSAPConnection();

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  return (
    <Box sx={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Panel
        title="SAP Business Data Cloud"
        subtitle="Extract data products, schemas, and intelligent applications from SAP BDC"
        dense
      >
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={activeTab} onChange={handleTabChange} aria-label="SAP module tabs">
            <Tab
              icon={<CloudIcon />}
              iconPosition="start"
              label="Connection"
              id="sap-tab-0"
              aria-controls="sap-tabpanel-0"
            />
            <Tab
              icon={<StorageIcon />}
              iconPosition="start"
              label="Data Products"
              id="sap-tab-1"
              aria-controls="sap-tabpanel-1"
              disabled={!isConnected}
            />
            <Tab
              icon={<SchemaIcon />}
              iconPosition="start"
              label="Schema Extraction"
              id="sap-tab-2"
              aria-controls="sap-tabpanel-2"
              disabled={!isConnected}
            />
            <Tab
              icon={<AccountTreeIcon />}
              iconPosition="start"
              label="Graph Visualization"
              id="sap-tab-3"
              aria-controls="sap-tabpanel-3"
              disabled={!isConnected}
            />
          </Tabs>
        </Box>
      </Panel>

      <Box sx={{ flex: 1, overflow: 'auto', p: 2 }}>
        <TabPanel value={activeTab} index={0}>
          <ConnectionConfigView
            connection={connection}
            onConnectionChange={setConnection}
            onTestConnection={testConnection}
            isConnected={isConnected}
          />
        </TabPanel>

        <TabPanel value={activeTab} index={1}>
          <DataProductsView connection={connection} />
        </TabPanel>

        <TabPanel value={activeTab} index={2}>
          <SchemaExtractionView connection={connection} />
        </TabPanel>

        <TabPanel value={activeTab} index={3}>
          <GraphIntegrationView connection={connection} />
        </TabPanel>
      </Box>
    </Box>
  );
}


