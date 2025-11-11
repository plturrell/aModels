/**
 * Relational Module - Ruthlessly Simplified
 * One action: Run SQL query
 * One view: Results table
 */

import React, { useState } from "react";
import {
  Box,
  Typography,
  TextField,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  CircularProgress
} from "@mui/material";
import {
  processRelationalTables,
  getRelationalProcessingResults,
  type RelationalTable
} from "../../api/relational";

export function RelationalModule() {
  const [query, setQuery] = useState("");
  const [processing, setProcessing] = useState(false);
  const [results, setResults] = useState<RelationalTable[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleExecute = async () => {
    if (!query.trim()) return;

    setProcessing(true);
    setError(null);
    setResults(null);

    try {
      const result = await processRelationalTables({
        table: "query_result",
        schema: "public",
        database_url: "postgresql://localhost:5432/amodels",
        database_type: "postgres",
        async: true
      });

      // Poll for results
      setTimeout(async () => {
        try {
          const data = await getRelationalProcessingResults(result.request_id);
          setResults(data.documents || []);
        } catch (err) {
          setError("Failed to fetch results");
        } finally {
          setProcessing(false);
        }
      }, 2000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Query failed");
      setProcessing(false);
    }
  };

  return (
    <Box sx={{ p: 6, maxWidth: 1200, margin: "0 auto" }}>
      {/* Query Input */}
      <Box sx={{ mb: 6 }}>
        <Typography variant="h4" sx={{ mb: 2, fontWeight: 600 }}>
          SQL Query
        </Typography>

        <TextField
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter SQL query..."
          multiline
          rows={6}
          fullWidth
          variant="outlined"
          sx={{
            mb: 2,
            "& .MuiOutlinedInput-root": {
              fontFamily: "monospace",
              fontSize: 14,
              borderRadius: 3
            }
          }}
        />

        <Button
          variant="contained"
          onClick={handleExecute}
          disabled={processing || !query.trim()}
          sx={{
            borderRadius: 3,
            textTransform: "none",
            fontSize: 16,
            px: 4,
            py: 1.5
          }}
        >
          {processing ? <CircularProgress size={24} /> : "Execute"}
        </Button>

        {error && (
          <Typography color="error" sx={{ mt: 2 }}>
            {error}
          </Typography>
        )}
      </Box>

      {/* Processing State */}
      {processing && (
        <Box sx={{ textAlign: "center", py: 8 }}>
          <CircularProgress size={48} sx={{ mb: 2 }} />
          <Typography variant="h6" color="text.secondary">
            Executing query...
          </Typography>
        </Box>
      )}

      {/* Results Table */}
      {results && results.length > 0 && (
        <Box>
          <Typography variant="h5" sx={{ mb: 3, fontWeight: 600 }}>
            Results ({results.length} tables)
          </Typography>

          {results.map((table, tableIndex) => (
            <Box key={tableIndex} sx={{ mb: 4 }}>
              <Typography variant="h6" sx={{ mb: 2 }}>
                {table.title || table.id}
              </Typography>

              {table.metadata && Object.keys(table.metadata).length > 0 && (
                <TableContainer component={Paper} sx={{ borderRadius: 3 }}>
                  <Table>
                    <TableHead>
                      <TableRow sx={{ bgcolor: "grey.100" }}>
                        <TableCell sx={{ fontWeight: 600 }}>Key</TableCell>
                        <TableCell sx={{ fontWeight: 600 }}>Value</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {Object.entries(table.metadata || {}).map(([key, value], rowIndex) => (
                        <TableRow key={rowIndex}>
                          <TableCell>{key}</TableCell>
                          <TableCell>{String(value)}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </Box>
          ))}
        </Box>
      )}

      {/* Empty State */}
      {!processing && !results && !error && (
        <Box sx={{ textAlign: "center", py: 8, color: "text.secondary" }}>
          <Typography variant="h6" gutterBottom>
            Enter a SQL query to begin
          </Typography>
          <Typography variant="body2">
            SELECT * FROM your_table LIMIT 10
          </Typography>
        </Box>
      )}
    </Box>
  );
}
