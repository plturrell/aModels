import { useEffect, useMemo, useState, type ReactNode } from "react";
import { parse } from "yaml";
import composeRaw from "#repo/services/localai/LocalAI/docker-compose.yaml?raw";
import {
  Box,
  Typography,
  TextField,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Slider,
  Chip,
  Link,
  Alert,
  CircularProgress,
  Stack,
  Paper,
  List,
  ListItem,
  ListItemText,
  Divider
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import ClearIcon from '@mui/icons-material/Clear';

import { Panel } from "../../components/Panel";
import { useLocalAIInventory } from "../../api/hooks";
import {
  useLocalAIChatStore,
  type ChatFollowUp,
  type ChatMessage,
  type NormalisedCitation
} from "../../state/useLocalAIChatStore";

interface ComposeService {
  ports?: Array<string | number>;
  environment?: Array<string> | Record<string, string>;
  command?: string[] | string;
  image?: string;
}

interface ComposeFile {
  services?: Record<string, ComposeService>;
}

const compose = parse(composeRaw) as ComposeFile;
const apiService = compose.services?.api;

const exposedPorts = (() => {
  if (!apiService?.ports) return [] as string[];
  return apiService.ports.map((entry) => (typeof entry === "number" ? entry.toString() : entry));
})();

const getBaseUrl = () => {
  if (!exposedPorts.length) return "http://localhost:8080";
  const [first] = exposedPorts;
  const [hostPort] = first.split(":");
  return `http://localhost:${hostPort}`;
};

const seedPrompts = [
  "Summarise SGMI Control-M triggers versus waits-for dependencies.",
  "Which LocalAI models are missing documentation?",
  "What telemetry improvements should we ship next?"
];

function useStreamedText(text: string, streaming?: boolean) {
  const [displayed, setDisplayed] = useState(streaming ? "" : text);

  useEffect(() => {
    if (!streaming) {
      setDisplayed(text);
      return;
    }

    setDisplayed("");
    let active = true;
    let index = 0;
    let timeoutId = window.setTimeout(step, 60);

    function step() {
      if (!active) return;
      const increment = Math.max(1, Math.round(text.length / 72));
      index = Math.min(text.length, index + increment);
      setDisplayed(text.slice(0, index));
      if (index < text.length) {
        timeoutId = window.setTimeout(step, 22);
      }
    }

    return () => {
      active = false;
      window.clearTimeout(timeoutId);
    };
  }, [text, streaming]);

  return streaming ? displayed : text;
}

function renderInlineWithCitations(
  text: string,
  citations: NormalisedCitation[],
  keyPrefix: string
): ReactNode[] {
  const nodes: ReactNode[] = [];
  const pattern = /\[\[(\d+)\]\]/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(
        <span key={`${keyPrefix}-text-${lastIndex}`}>{text.slice(lastIndex, match.index)}</span>
      );
    }

    const citationIndex = Number.parseInt(match[1], 10) - 1;
    const citation = citations[citationIndex];
    const label = `[[${match[1]}]]`;
    const key = `${keyPrefix}-cite-${match.index}`;

    if (citation?.url) {
      nodes.push(
        <Link
          key={key}
          href={citation.url}
          target="_blank"
          rel="noreferrer"
          sx={{ 
            color: 'primary.main',
            textDecoration: 'underline',
            '&:hover': { textDecoration: 'none' }
          }}
        >
          {label}
        </Link>
      );
    } else {
      nodes.push(
        <Chip
          key={key}
          label={label}
          size="small"
          variant="outlined"
          sx={{ 
            height: 'auto',
            fontSize: '0.75rem',
            ml: 0.5,
            mr: 0.5
          }}
        />
      );
    }

    lastIndex = pattern.lastIndex;
  }

  if (lastIndex < text.length) {
    nodes.push(<span key={`${keyPrefix}-tail`}>{text.slice(lastIndex)}</span>);
  }

  if (!nodes.length) {
    nodes.push(text);
  }
  return nodes;
}

function CitationList({ citations }: { citations: NormalisedCitation[] }) {
  return (
    <Box sx={{ mt: 2, pt: 2, borderTop: 1, borderColor: 'divider' }}>
      <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
        Citations
      </Typography>
      <List dense>
        {citations.map((citation, index) => (
          <ListItem key={citation.id ?? index} sx={{ py: 0.5, px: 0 }}>
            <Chip
              label={`[[${index + 1}]]`}
              size="small"
              sx={{ mr: 1, minWidth: 40 }}
            />
            <ListItemText
              primary={
                citation.url ? (
                  <Link href={citation.url} target="_blank" rel="noreferrer">
                    {citation.label}
                  </Link>
                ) : (
                  <Typography variant="body2">{citation.label}</Typography>
                )
              }
              secondary={citation.snippet ? (
                <Typography variant="caption" color="text.secondary">
                  {citation.snippet}
                </Typography>
              ) : undefined}
            />
          </ListItem>
        ))}
      </List>
    </Box>
  );
}

function ChatBubble({ message }: { message: ChatMessage }) {
  const text = useStreamedText(message.content, message.streaming);
  const citations = message.citations ?? [];

  const paragraphs = useMemo(
    () => text.split(/\n{2,}/).filter((block) => block.trim().length > 0),
    [text]
  );

  const timestamp = new Date(message.createdAt);
  const timeLabel =
    Number.isNaN(timestamp.getTime()) === false
      ? timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
      : "";
  const roleLabel = message.role === "assistant" ? "LocalAI" : "You";

  const isAssistant = message.role === "assistant";
  const bgColor = isAssistant ? 'primary.light' : 'grey.200';
  const textColor = isAssistant ? 'primary.contrastText' : 'text.primary';

  return (
    <Paper
      elevation={1}
      sx={{
        p: 2,
        mb: 2,
        bgcolor: isAssistant ? 'primary.light' : 'grey.100',
        color: isAssistant ? 'primary.contrastText' : 'text.primary',
        opacity: message.error ? 0.7 : 1,
        borderLeft: isAssistant ? 4 : 0,
        borderColor: isAssistant ? 'primary.dark' : 'transparent',
        position: 'relative'
      }}
    >
      <Box sx={{ mb: 1 }}>
        {paragraphs.length
          ? paragraphs.map((paragraph, paragraphIndex) => {
              const lines = paragraph.split("\n");
              return (
                <Typography
                  key={`${message.id}-${paragraphIndex}`}
                  variant="body1"
                  paragraph
                  sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}
                >
                  {lines.map((line, lineIndex) => (
                    <span key={`${message.id}-${paragraphIndex}-${lineIndex}`}>
                      {renderInlineWithCitations(
                        line,
                        citations,
                        `${message.id}-${paragraphIndex}-${lineIndex}`
                      )}
                      {lineIndex < lines.length - 1 ? <br /> : null}
                    </span>
                  ))}
                </Typography>
              );
            })
          : (
            <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
              {renderInlineWithCitations(text, citations, `${message.id}-inline`)}
            </Typography>
          )}
        {message.streaming ? (
          <Box
            component="span"
            sx={{
              display: 'inline-block',
              width: 8,
              height: 16,
              bgcolor: 'currentColor',
              ml: 0.5,
              animation: 'blink 1s infinite',
              '@keyframes blink': {
                '0%, 100%': { opacity: 1 },
                '50%': { opacity: 0 }
              }
            }}
            aria-hidden="true"
          />
        ) : null}
      </Box>
      
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 1 }}>
        <Typography variant="caption" color="text.secondary">
          {roleLabel}
        </Typography>
        {timeLabel ? (
          <Typography variant="caption" color="text.secondary">
            {timeLabel}
          </Typography>
        ) : null}
      </Box>

      {message.error ? (
        <Alert severity="error" sx={{ mt: 1 }}>
          Message failed to send. Try again.
        </Alert>
      ) : null}
      
      {citations.length ? <CitationList citations={citations} /> : null}
    </Paper>
  );
}

export function LocalAIModule() {
  const { data: inventory, loading, error, refresh } = useLocalAIInventory();

  const model = useLocalAIChatStore((state) => state.model);
  const setModel = useLocalAIChatStore((state) => state.setModel);
  const temperature = useLocalAIChatStore((state) => state.temperature);
  const setTemperature = useLocalAIChatStore((state) => state.setTemperature);
  const messages = useLocalAIChatStore((state) => state.messages);
  const followUps = useLocalAIChatStore((state) => state.followUps);
  const pending = useLocalAIChatStore((state) => state.pending);
  const chatError = useLocalAIChatStore((state) => state.error);
  const sendMessage = useLocalAIChatStore((state) => state.sendMessage);
  const applyFollowUp = useLocalAIChatStore((state) => state.applyFollowUp);
  const resetChat = useLocalAIChatStore((state) => state.reset);

  const [draft, setDraft] = useState("");

  const modelOptions = useMemo(() => inventory?.models ?? [], [inventory]);
  const documentedModels = useMemo(
    () => modelOptions.filter((entry) => entry.readme),
    [modelOptions]
  );

  useEffect(() => {
    if (!modelOptions.length) return;
    if (!modelOptions.some((entry) => entry.id === model)) {
      setModel(modelOptions[0].id);
    }
  }, [modelOptions, model, setModel]);

  if (!inventory) {
    return (
      <Box>
        <Panel title="LocalAI Inventory" subtitle="Loading model catalog">
          {loading ? (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <CircularProgress size={20} />
              <Typography>Loading LocalAI models…</Typography>
            </Box>
          ) : (
            <Typography>Unable to load LocalAI inventory.</Typography>
          )}
          {error ? (
            <Alert severity="error" sx={{ mt: 2 }}>
              Error: {error.message}
            </Alert>
          ) : null}
          <Button
            variant="outlined"
            onClick={refresh}
            disabled={loading}
            sx={{ mt: 2 }}
          >
            Reload
          </Button>
        </Panel>
      </Box>
    );
  }

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const value = draft.trim();
    if (!value) return;
    void sendMessage(value);
    setDraft("");
  };

  const handleFollowUp = (followUp: ChatFollowUp) => {
    setDraft("");
    void applyFollowUp(followUp.prompt);
  };

  const handleSeedPrompt = (prompt: string) => {
    setDraft(prompt);
  };

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      const value = draft.trim();
      if (!value || pending) return;
      void sendMessage(value);
      setDraft("");
    }
  };

  const modelActions = (
    <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap">
      <FormControl size="small" sx={{ minWidth: 120 }}>
        <InputLabel>Model</InputLabel>
        <Select
          value={model}
          label="Model"
          onChange={(event) => setModel(event.target.value)}
        >
          {modelOptions.map((entry) => (
            <MenuItem key={entry.id} value={entry.id}>
              {entry.id}
            </MenuItem>
          ))}
        </Select>
      </FormControl>
      
      <Box sx={{ minWidth: 150, display: 'flex', alignItems: 'center', gap: 1 }}>
        <Typography variant="caption" sx={{ minWidth: 60 }}>
          Creativity
        </Typography>
        <Slider
          value={temperature}
          onChange={(_, value) => setTemperature(value as number)}
          min={0}
          max={1}
          step={0.1}
          size="small"
          sx={{ flex: 1 }}
        />
        <Typography variant="caption" sx={{ minWidth: 30, textAlign: 'right' }}>
          {temperature.toFixed(1)}
        </Typography>
      </Box>
      
      <Button
        variant="outlined"
        size="small"
        startIcon={<ClearIcon />}
        onClick={() => {
          resetChat();
          setDraft("");
        }}
        disabled={!messages.length && !chatError && !draft.trim().length}
      >
        Clear Thread
      </Button>
    </Stack>
  );

  return (
    <Box>
      <Panel title="Live Answer" subtitle="LocalAI interprets SGMI context" actions={modelActions}>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, minHeight: 420 }}>
          <Box
            sx={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              gap: 2,
              overflowY: 'auto',
              maxHeight: 500,
              pr: 1
            }}
            aria-live="polite"
          >
            {messages.length ? (
              messages.map((message) => <ChatBubble key={message.id} message={message} />)
            ) : (
              <Paper
                variant="outlined"
                sx={{
                  p: 3,
                  bgcolor: 'grey.50',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 2
                }}
              >
                <Typography variant="body2" color="text.secondary">
                  Ask something grounded in SGMI—jobs, telemetry, or training drift.
                </Typography>
                <Stack direction="row" spacing={1} flexWrap="wrap">
                  {seedPrompts.map((prompt) => (
                    <Button
                      key={prompt}
                      variant="outlined"
                      size="small"
                      onClick={() => handleSeedPrompt(prompt)}
                      disabled={pending}
                      sx={{ textTransform: 'none' }}
                    >
                      {prompt}
                    </Button>
                  ))}
                </Stack>
              </Paper>
            )}
            {pending ? (
              <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', py: 2 }}>
                <CircularProgress size={16} />
                <Typography variant="body2" color="text.secondary">
                  Thinking...
                </Typography>
              </Box>
            ) : null}
          </Box>

          {chatError ? (
            <Alert severity="error">Error: {chatError}</Alert>
          ) : null}

          {followUps.length ? (
            <Box>
              <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
                Suggested follow-ups
              </Typography>
              <Stack direction="row" spacing={1} flexWrap="wrap">
                {followUps.map((item) => (
                  <Button
                    key={item.id}
                    variant="outlined"
                    size="small"
                    onClick={() => handleFollowUp(item)}
                    disabled={pending}
                    sx={{ textTransform: 'none' }}
                  >
                    {item.label}
                  </Button>
                ))}
              </Stack>
            </Box>
          ) : null}

          <Box component="form" onSubmit={handleSubmit} sx={{ display: 'flex', gap: 1 }}>
            <TextField
              fullWidth
              multiline
              rows={3}
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about waits, telemetry, or training coverage…"
              disabled={pending}
              variant="outlined"
            />
            <Button
              type="submit"
              variant="contained"
              startIcon={<SendIcon />}
              disabled={pending || !draft.trim()}
              sx={{ alignSelf: 'flex-end' }}
            >
              {pending ? "Sending…" : "Send"}
            </Button>
          </Box>
        </Box>
      </Panel>

      <Stack direction={{ xs: 'column', md: 'row' }} spacing={2}>
        <Box sx={{ flex: 1 }}>
          <Panel title="Runtime Snapshot" subtitle="services/localai/LocalAI/docker-compose.yaml" dense>
            <Stack spacing={2}>
              <Box>
                <Typography variant="caption" color="text.secondary">Image</Typography>
                <Typography variant="body2">
                  {apiService?.image ?? "quay.io/go-skynet/local-ai:master"}
                </Typography>
              </Box>
              <Box>
                <Typography variant="caption" color="text.secondary">Ports</Typography>
                <Typography variant="body2">
                  {exposedPorts.join(", ") || "8080:8080"}
                </Typography>
              </Box>
              <Box>
                <Typography variant="caption" color="text.secondary">Command</Typography>
                <Typography
                  variant="body2"
                  component="code"
                  sx={{
                    display: 'block',
                    fontFamily: 'monospace',
                    fontSize: '0.875rem',
                    bgcolor: 'grey.100',
                    p: 0.5,
                    borderRadius: 1
                  }}
                >
                  {Array.isArray(apiService?.command)
                    ? apiService.command.join(" ")
                    : apiService?.command ?? "phi-2"}
                </Typography>
              </Box>
              <Box>
                <Typography variant="caption" color="text.secondary">Base URL</Typography>
                <Link href={getBaseUrl()} target="_blank" rel="noreferrer">
                  {getBaseUrl()}
                </Link>
              </Box>
            </Stack>
          </Panel>
        </Box>

        <Box sx={{ flex: 1 }}>
          <Panel
            title="Model Inventory"
            subtitle={`Documented ${documentedModels.length} of ${modelOptions.length}`}
            dense
          >
            <List dense>
              {modelOptions.slice(0, 6).map((entry) => (
                <ListItem key={entry.id} sx={{ py: 0.5, px: 0 }}>
                  <ListItemText
                    primary={entry.id}
                    secondary={entry.readme ? "Doc complete" : "Needs README"}
                  />
                  {entry.readme ? (
                    <Chip label="Ready" size="small" color="success" />
                  ) : (
                    <Chip label="Todo" size="small" color="warning" />
                  )}
                </ListItem>
              ))}
              {modelOptions.length > 6 ? (
                <ListItem>
                  <ListItemText
                    primary={`+${modelOptions.length - 6} more models`}
                    primaryTypographyProps={{ variant: 'body2', color: 'text.secondary' }}
                  />
                </ListItem>
              ) : null}
            </List>
          </Panel>
        </Box>
      </Stack>
    </Box>
  );
}
