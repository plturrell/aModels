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
  Divider
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import ClearIcon from '@mui/icons-material/Clear';

import { Panel } from "../../components/Panel";
import { searchDocuments, type SearchResult } from "../../api/search";

const formatSimilarity = (similarity: number) => {
  return `${(similarity * 100).toFixed(1)}%`;
};

const truncateContent = (content: string, maxLength: number = 200) => {
  if (content.length <= maxLength) return content;
  return `${content.slice(0, maxLength)}…`;
};

export function SearchModule() {
  const [query, setQuery] = useState<string>("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<Error | null>(null);
  const [hasSearched, setHasSearched] = useState<boolean>(false);

  const handleSearch = async () => {
    const trimmedQuery = query.trim();
    if (!trimmedQuery) return;

    setLoading(true);
    setError(null);
    setHasSearched(true);

    try {
      const response = await searchDocuments({
        query: trimmedQuery,
        top_k: 20
      });
      setResults(response.results || []);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
      setResults([]);
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
        <Panel title="Search Results" subtitle={`Found ${results.length} result${results.length !== 1 ? 's' : ''}`}>
          <List>
            {results.map((result, index) => (
              <Box key={result.id}>
                <ListItem alignItems="flex-start" sx={{ py: 2 }}>
                  <ListItemText
                    primary={
                      <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
                        <Typography variant="body2" fontWeight={500}>
                          Result #{index + 1}
                        </Typography>
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
                    }
                  />
                </ListItem>
                {index < results.length - 1 && <Divider />}
              </Box>
            ))}
          </List>
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

