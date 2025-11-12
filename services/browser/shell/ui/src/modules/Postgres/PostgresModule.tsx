/**
 * Postgres Module
 * 
 * Database administration interface for PostgreSQL
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  TextField,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Alert,
  CircularProgress,
  Chip,
  Stack,
  Grid,
  Card,
  CardContent,
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import InfoIcon from '@mui/icons-material/Info';
import { fetchJSON } from '../../api/client';

const API_BASE = import.meta.env.VITE_GATEWAY_URL || 'http://localhost:8000';

interface TableInfo {
  table_schema: string;
  table_name: string;
}

interface ColumnInfo {
  column_name: string;
  data_type: string;
  is_nullable: string;
  column_default: string | null;
}

interface QueryResult {
  columns: string[];
  rows: Record<string, any>[];
  row_count: number;
  truncated: boolean;
}

interface StatusResponse {
  enabled: boolean;
  allow_mutations: boolean;
  default_limit: number;
}

export function PostgresModule() {
  const [tables, setTables] = useState<TableInfo[]>([]);
  const [filteredTables, setFilteredTables] = useState<TableInfo[]>([]);
  const [selectedTable, setSelectedTable] = useState<{ schema: string; table: string } | null>(null);
  const [columns, setColumns] = useState<ColumnInfo[]>([]);
  const [query, setQuery] = useState('');
  const [queryResult, setQueryResult] = useState<QueryResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [tableSearch, setTableSearch] = useState('');

  useEffect(() => {
    fetchStatus();
    fetchTables();
  }, []);

  useEffect(() => {
    if (selectedTable) {
      fetchColumns(selectedTable.schema, selectedTable.table);
      generateQuery();
    }
  }, [selectedTable]);

  useEffect(() => {
    if (tableSearch) {
      const filtered = tables.filter(
        (t) =>
          `${t.table_schema}.${t.table_name}`.toLowerCase().includes(tableSearch.toLowerCase())
      );
      setFilteredTables(filtered);
    } else {
      setFilteredTables(tables);
    }
  }, [tableSearch, tables]);

  const fetchStatus = async () => {
    try {
      const data = await fetchJSON<StatusResponse>(`${API_BASE}/db/status`);
      setStatus(data);
    } catch (err) {
      console.error('Failed to fetch status:', err);
    }
  };

  const fetchTables = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await fetchJSON<{ tables: TableInfo[] }>(`${API_BASE}/db/tables`);
      setTables(data.tables || []);
      setFilteredTables(data.tables || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch tables');
    } finally {
      setLoading(false);
    }
  };

  const fetchColumns = async (schema: string, table: string) => {
    try {
      setLoading(true);
      setError(null);
      const data = await fetchJSON<ColumnInfo[]>(
        `${API_BASE}/db/table/${encodeURIComponent(schema)}/${encodeURIComponent(table)}`
      );
      setColumns(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch columns');
    } finally {
      setLoading(false);
    }
  };

  const generateQuery = () => {
    if (selectedTable) {
      const limit = status?.default_limit || 200;
      setQuery(`SELECT * FROM ${selectedTable.schema}.${selectedTable.table} LIMIT ${limit};`);
    }
  };

  const executeQuery = async () => {
    if (!query.trim()) {
      setError('Query is empty');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setQueryResult(null);

      const limit = status?.default_limit || 200;
      const result = await fetchJSON<QueryResult>(`${API_BASE}/db/query`, {
        method: 'POST',
        body: JSON.stringify({
          sql: query.trim(),
          limit: limit,
        }),
      });

      setQueryResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Query execution failed');
      setQueryResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Paper sx={{ p: 2 }}>
        <Typography variant="h4" gutterBottom>
          PostgreSQL Admin
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Database administration interface for querying and exploring PostgreSQL databases
        </Typography>
        {status && (
          <Stack direction="row" spacing={1} sx={{ mt: 2 }}>
            <Chip
              label={status.enabled ? 'Enabled' : 'Disabled'}
              color={status.enabled ? 'success' : 'default'}
              size="small"
            />
            <Chip
              label={status.allow_mutations ? 'Mutations Allowed' : 'Read-Only'}
              color={status.allow_mutations ? 'warning' : 'info'}
              size="small"
            />
            <Chip label={`Default Limit: ${status.default_limit}`} size="small" />
          </Stack>
        )}
      </Paper>

      <Grid container spacing={2} sx={{ flex: 1, overflow: 'hidden' }}>
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2, height: '100%', overflow: 'auto' }}>
            <Typography variant="h6" gutterBottom>
              Tables
            </Typography>
            <TextField
              fullWidth
              size="small"
              placeholder="Search tables..."
              value={tableSearch}
              onChange={(e) => setTableSearch(e.target.value)}
              sx={{ mb: 2 }}
            />
            {loading && tables.length === 0 ? (
              <CircularProgress size={24} />
            ) : filteredTables.length === 0 ? (
              <Typography variant="body2" color="text.secondary">
                No tables found
              </Typography>
            ) : (
              <Box sx={{ maxHeight: '400px', overflow: 'auto' }}>
                {filteredTables.map((table) => {
                  const isSelected =
                    selectedTable?.schema === table.table_schema &&
                    selectedTable?.table === table.table_name;
                  return (
                    <Box
                      key={`${table.table_schema}.${table.table_name}`}
                      onClick={() =>
                        setSelectedTable({ schema: table.table_schema, table: table.table_name })
                      }
                      sx={{
                        p: 1,
                        cursor: 'pointer',
                        borderRadius: 1,
                        bgcolor: isSelected ? 'action.selected' : 'transparent',
                        '&:hover': { bgcolor: 'action.hover' },
                        mb: 0.5,
                      }}
                    >
                      <Typography variant="body2">
                        {table.table_schema}.{table.table_name}
                      </Typography>
                    </Box>
                  );
                })}
              </Box>
            )}
          </Paper>
        </Grid>

        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column' }}>
            {selectedTable && (
              <Box sx={{ mb: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Columns: {selectedTable.schema}.{selectedTable.table}
                </Typography>
                {columns.length > 0 ? (
                  <TableContainer>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Column</TableCell>
                          <TableCell>Type</TableCell>
                          <TableCell>Nullable</TableCell>
                          <TableCell>Default</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {columns.map((col) => (
                          <TableRow key={col.column_name}>
                            <TableCell>{col.column_name}</TableCell>
                            <TableCell>{col.data_type}</TableCell>
                            <TableCell>{col.is_nullable}</TableCell>
                            <TableCell>{col.column_default || '-'}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    No columns available
                  </Typography>
                )}
              </Box>
            )}

            <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Typography variant="h6">SQL Query</Typography>
              <TextField
                fullWidth
                multiline
                rows={6}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Enter SQL query..."
                sx={{ fontFamily: 'monospace' }}
              />
              <Stack direction="row" spacing={2}>
                <Button
                  variant="contained"
                  startIcon={<PlayArrowIcon />}
                  onClick={executeQuery}
                  disabled={loading || !query.trim()}
                >
                  {loading ? 'Executing...' : 'Execute Query'}
                </Button>
                {selectedTable && (
                  <Button variant="outlined" onClick={generateQuery}>
                    Generate SELECT
                  </Button>
                )}
              </Stack>

              {error && (
                <Alert severity="error" onClose={() => setError(null)}>
                  {error}
                </Alert>
              )}

              {queryResult && (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    Results ({queryResult.row_count} row{queryResult.row_count !== 1 ? 's' : ''}
                    {queryResult.truncated ? ' - truncated' : ''})
                  </Typography>
                  <TableContainer sx={{ maxHeight: '400px', overflow: 'auto' }}>
                    <Table size="small" stickyHeader>
                      <TableHead>
                        <TableRow>
                          {queryResult.columns.map((col) => (
                            <TableCell key={col}>{col}</TableCell>
                          ))}
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {queryResult.rows.map((row, idx) => (
                          <TableRow key={idx}>
                            {queryResult.columns.map((col) => (
                              <TableCell key={col}>
                                {row[col] === null || row[col] === undefined
                                  ? 'NULL'
                                  : String(row[col])}
                              </TableCell>
                            ))}
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Box>
              )}

              {!selectedTable && (
                <Alert severity="info" icon={<InfoIcon />}>
                  Select a table from the left panel to view columns and generate queries
                </Alert>
              )}
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}

