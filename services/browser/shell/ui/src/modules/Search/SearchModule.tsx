import { useState } from "react";
import {
  Box,
  Typography,
  TextField,
  Button,
  Alert,
  Paper,
  List,
  ListItem,
  ListItemText,
  CircularProgress,
  Chip,
  Stack,
  Divider,
  FormControlLabel,
  Checkbox,
  Tabs,
  Tab
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import ClearIcon from '@mui/icons-material/Clear';

import { Panel } from "../../components/Panel";
import { unifiedSearch, type UnifiedSearchResult } from "../../api/search";

const formatSimilarity = (similarity: number) => {
  return `${(similarity * 100).toFixed(1)}%`;
};

const truncateContent = (content: string, maxLength: number = 200) => {
  if (content.length <= maxLength) return content;
  return `${content.slice(0, maxLength)}…`;
};

export function SearchModule() {
  const [query, setQuery] = useState<string>("");
  const [results, setResults] = useState<UnifiedSearchResult[]>([]);
  const [sources, setSources] = useState<Record<string, unknown>>({});
  const [searchResponse, setSearchResponse] = useState<UnifiedSearchResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<Error | null>(null);
  const [hasSearched, setHasSearched] = useState<boolean>(false);
  const [usePerplexity, setUsePerplexity] = useState<boolean>(false);
  const [enableFramework, setEnableFramework] = useState<boolean>(false);
  const [enablePlot, setEnablePlot] = useState<boolean>(false);
  const [selectedTab, setSelectedTab] = useState<number>(0);

  const handleSearch = async () => {
    const trimmedQuery = query.trim();
    if (!trimmedQuery) return;

    setLoading(true);
    setError(null);
    setHasSearched(true);

    try {
      const response = await unifiedSearch({
        query: trimmedQuery,
        top_k: 20,
        sources: ["inference", "knowledge_graph", "catalog"],
        use_perplexity: usePerplexity,
        enable_framework: enableFramework,
        enable_plot: enablePlot,
        enable_stdlib: true
      });
      setResults(response.combined_results || []);
      setSources(response.sources || {});
      setSearchResponse(response);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
      setResults([]);
      setSources({});
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setQuery("");
    setResults([]);
    setError(null);
    setHasSearched(false);
  };

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void handleSearch();
    }
  };

  return (
    <Box>
      <Panel
        title="Semantic Search"
        subtitle="Search across documents, knowledge graphs, and indexed content"
        actions={
          <Button
            variant="outlined"
            size="small"
            startIcon={<ClearIcon />}
            onClick={handleClear}
            disabled={!query && !hasSearched}
          >
            Clear
          </Button>
        }
      >
        <Stack spacing={2}>
          <Stack spacing={2}>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <TextField
                fullWidth
                placeholder="Enter your search query..."
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                onKeyDown={handleKeyDown}
                disabled={loading}
                variant="outlined"
                InputProps={{
                  startAdornment: <SearchIcon sx={{ mr: 1, color: 'text.secondary' }} />
                }}
              />
              <Button
                variant="contained"
                startIcon={loading ? <CircularProgress size={16} /> : <SearchIcon />}
                onClick={handleSearch}
                disabled={!query.trim() || loading}
                sx={{ minWidth: 120 }}
              >
                {loading ? "Searching…" : "Search"}
              </Button>
            </Box>
            <Stack spacing={1}>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={usePerplexity}
                    onChange={(e) => setUsePerplexity(e.target.checked)}
                    disabled={loading}
                  />
                }
                label="Include Perplexity web search (requires API key)"
              />
              <FormControlLabel
                control={
                  <Checkbox
                    checked={enableFramework}
                    onChange={(e) => setEnableFramework(e.target.checked)}
                    disabled={loading}
                  />
                }
                label="Enable AI enrichment (query understanding & result summarization)"
              />
              <FormControlLabel
                control={
                  <Checkbox
                    checked={enablePlot}
                    onChange={(e) => setEnablePlot(e.target.checked)}
                    disabled={loading}
                  />
                }
                label="Generate visualization data"
              />
            </Stack>
          </Stack>

          {error ? (
            <Alert severity="error">
              <Typography variant="body2" fontWeight={500} gutterBottom>
                Search failed
              </Typography>
              <Typography variant="body2" component="pre" sx={{ 
                fontSize: '0.75rem', 
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
                mb: 0
              }}>
                {error.message}
              </Typography>
            </Alert>
          ) : null}

          {!hasSearched ? (
            <Paper variant="outlined" sx={{ p: 4, textAlign: 'center' }}>
              <SearchIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
              <Typography variant="h6" gutterBottom>
                Start searching
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Enter a query to search across indexed documents, knowledge graphs, and content.
                Results are ranked by semantic similarity.
              </Typography>
            </Paper>
          ) : null}
        </Stack>
      </Panel>

      {hasSearched && !loading && results.length > 0 && (
        <Panel title="Search Results" subtitle={`Found ${results.length} result${results.length !== 1 ? 's' : ''} across multiple sources`}>
          <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
            <Tabs value={selectedTab} onChange={(_, newValue) => setSelectedTab(newValue)}>
              <Tab label="All Results" />
              <Tab label="By Source" />
              {searchResponse?.visualization && <Tab label="Visualization" />}
              {searchResponse?.result_enrichment && <Tab label="AI Insights" />}
            </Tabs>
          </Box>
          
          {selectedTab === 0 && (
            <List>
              {results.map((result, index) => (
                <Box key={`${result.source}-${result.id}-${index}`}>
                  <ListItem alignItems="flex-start" sx={{ py: 2 }}>
                    <ListItemText
                      primary={
                        <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }} flexWrap="wrap">
                          <Typography variant="body2" fontWeight={500}>
                            Result #{index + 1}
                          </Typography>
                          <Chip
                            label={result.source}
                            size="small"
                            color="secondary"
                            variant="outlined"
                          />
                          <Chip
                            label={formatSimilarity(result.similarity)}
                            size="small"
                            color="primary"
                            variant="outlined"
                          />
                          <Typography variant="caption" color="text.secondary">
                            ID: {result.id}
                          </Typography>
                        </Stack>
                      }
                      secondary={
                        <>
                          <Typography
                            variant="body2"
                            sx={{
                              mt: 1,
                              whiteSpace: 'pre-wrap',
                              wordBreak: 'break-word'
                            }}
                          >
                            {truncateContent(result.content)}
                          </Typography>
                          {result.citations && result.citations.length > 0 && (
                            <Box sx={{ mt: 1 }}>
                              <Typography variant="caption" color="text.secondary">
                                Citations: {result.citations.join(", ")}
                              </Typography>
                            </Box>
                          )}
                        </>
                      }
                    />
                  </ListItem>
                  {index < results.length - 1 && <Divider />}
                </Box>
              ))}
            </List>
          )}
          
          {selectedTab === 1 && (
            <Stack spacing={2}>
              {Object.entries(sources).map(([sourceName, sourceData]) => (
                <Paper key={sourceName} variant="outlined" sx={{ p: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    {sourceName.charAt(0).toUpperCase() + sourceName.slice(1).replace('_', ' ')}
                  </Typography>
                  {sourceData && typeof sourceData === 'object' && 'error' in sourceData ? (
                    <Alert severity="error">{(sourceData as {error: string}).error}</Alert>
                  ) : (
                    <Typography variant="body2" color="text.secondary">
                      {Array.isArray(sourceData) 
                        ? `${sourceData.length} results` 
                        : typeof sourceData === 'object' && sourceData !== null && 'content' in sourceData
                        ? "Web search result available"
                        : "No results"}
                    </Typography>
                  )}
                </Paper>
              ))}
            </Stack>
          )}
          
          {selectedTab === 2 && searchResponse?.visualization && (
            <Stack spacing={2}>
              <Paper variant="outlined" sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Source Distribution
                </Typography>
                <Stack spacing={1}>
                  {Object.entries(searchResponse.visualization.source_distribution).map(([source, count]) => (
                    <Box key={source} sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography variant="body2">{source}</Typography>
                      <Chip label={count} size="small" />
                    </Box>
                  ))}
                </Stack>
              </Paper>
              <Paper variant="outlined" sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Score Statistics
                </Typography>
                <Typography variant="body2">
                  Average: {(searchResponse.visualization.score_statistics.average * 100).toFixed(1)}%
                </Typography>
                <Typography variant="body2">
                  Range: {(searchResponse.visualization.score_statistics.min * 100).toFixed(1)}% - {(searchResponse.visualization.score_statistics.max * 100).toFixed(1)}%
                </Typography>
                <Typography variant="body2">
                  Total Results: {searchResponse.visualization.score_statistics.count}
                </Typography>
              </Paper>
            </Stack>
          )}
          
          {selectedTab === 3 && searchResponse?.result_enrichment && (
            <Paper variant="outlined" sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                AI-Generated Summary
              </Typography>
              <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                {searchResponse.result_enrichment.summary || "No summary available"}
              </Typography>
            </Paper>
          )}
        </Panel>
      )}

      {hasSearched && !loading && results.length === 0 && !error && (
        <Panel title="No Results" subtitle="Try a different search query">
          <Paper variant="outlined" sx={{ p: 3, textAlign: 'center' }}>
            <Typography variant="body2" color="text.secondary">
              No results found for "{query}". Try adjusting your search terms or check if documents are indexed.
            </Typography>
          </Paper>
        </Panel>
      )}
    </Box>
  );
}

