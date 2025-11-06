import { useEffect, useMemo, useState, type ReactNode } from "react";
import { parse } from "yaml";
import composeRaw from "#repo/services/localai/LocalAI/docker-compose.yaml?raw";

import { Panel } from "../../components/Panel";
import { useLocalAIInventory } from "../../api/hooks";
import {
  useLocalAIChatStore,
  type ChatFollowUp,
  type ChatMessage,
  type NormalisedCitation
} from "../../state/useLocalAIChatStore";

import styles from "./LocalAIModule.module.css";

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
        <a
          key={key}
          href={citation.url}
          target="_blank"
          rel="noreferrer"
          className={styles.footnote}
        >
          {label}
        </a>
      );
    } else {
      nodes.push(
        <span key={key} className={styles.footnote}>
          {label}
        </span>
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
    <ul className={styles.citationList}>
      {citations.map((citation, index) => (
        <li key={citation.id ?? index} className={styles.citationItem}>
          <span className={styles.citationBadge}>[[{index + 1}]]</span>
          <div className={styles.citationContent}>
            {citation.url ? (
              <a href={citation.url} target="_blank" rel="noreferrer">
                {citation.label}
              </a>
            ) : (
              <span>{citation.label}</span>
            )}
            {citation.snippet ? (
              <p className={styles.citationSnippet}>{citation.snippet}</p>
            ) : null}
          </div>
        </li>
      ))}
    </ul>
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

  return (
    <article
      className={[
        styles.chatMessage,
        message.role === "assistant" ? styles.assistantMessage : styles.userMessage,
        message.streaming ? styles.streaming : "",
        message.error ? styles.errored : ""
      ].join(" ")}
    >
      <div className={styles.messageBody}>
        {paragraphs.length
          ? paragraphs.map((paragraph, paragraphIndex) => {
              const lines = paragraph.split("\n");
              return (
                <p key={`${message.id}-${paragraphIndex}`} className={styles.paragraph}>
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
                </p>
              );
            })
          : renderInlineWithCitations(text, citations, `${message.id}-inline`)}
        {message.streaming ? <span className={styles.cursor} aria-hidden="true" /> : null}
      </div>
      <footer className={styles.messageMeta}>
        <span>{roleLabel}</span>
        {timeLabel ? <span>{timeLabel}</span> : null}
      </footer>
      {message.error ? (
        <p className={styles.messageError}>Message failed to send. Try again.</p>
      ) : null}
      {citations.length ? <CitationList citations={citations} /> : null}
    </article>
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
      <div className={styles.placeholder}>
        <Panel title="LocalAI Inventory" subtitle="Loading model catalog">
          {loading ? <p>Loading LocalAI models…</p> : <p>Unable to load LocalAI inventory.</p>}
          {error ? <p className={styles.error}>Error: {error.message}</p> : null}
          <button type="button" onClick={refresh} disabled={loading}>
            Reload
          </button>
        </Panel>
      </div>
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

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      const value = draft.trim();
      if (!value || pending) return;
      void sendMessage(value);
      setDraft("");
    }
  };

  const modelActions = (
    <div className={styles.metaControls}>
      <div className={styles.pillGroup}>
        <label className={styles.selectLabel}>
          <span>Model</span>
          <select
            className={styles.modelSelect}
            value={model}
            onChange={(event) => setModel(event.target.value)}
          >
            {modelOptions.map((entry) => (
              <option key={entry.id} value={entry.id}>
                {entry.id}
              </option>
            ))}
          </select>
        </label>
        <label className={styles.selectLabel}>
          <span>Creativity</span>
          <input
            className={styles.temperatureSlider}
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={temperature}
            onChange={(event) => setTemperature(Number(event.target.value))}
          />
          <small>{temperature.toFixed(1)}</small>
        </label>
      </div>
      <button
        type="button"
        className={styles.resetButton}
        onClick={() => {
          resetChat();
          setDraft("");
        }}
        disabled={!messages.length && !chatError && !draft.trim().length}
      >
        Clear Thread
      </button>
    </div>
  );

  return (
    <div className={styles.localai}>
      <Panel title="Live Answer" subtitle="LocalAI interprets SGMI context" actions={modelActions}>
        <div className={styles.hero}>
          <div className={styles.chatScroll} aria-live="polite">
            {messages.length ? (
              messages.map((message) => <ChatBubble key={message.id} message={message} />)
            ) : (
              <div className={styles.chatPlaceholder}>
                <p>Ask something grounded in SGMI—jobs, telemetry, or training drift.</p>
                <div className={styles.seedRow}>
                  {seedPrompts.map((prompt) => (
                    <button
                      key={prompt}
                      type="button"
                      className={styles.seedButton}
                      onClick={() => handleSeedPrompt(prompt)}
                      disabled={pending}
                    >
                      {prompt}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {pending ? (
              <div className={styles.typingIndicator} role="status" aria-live="assertive">
                <span />
                <span />
                <span />
              </div>
            ) : null}
          </div>

          {chatError ? <div className={styles.errorBanner}>Error: {chatError}</div> : null}

          {followUps.length ? (
            <div className={styles.followUps}>
              <span className={styles.followUpsLabel}>Suggested follow-ups</span>
              <div className={styles.followUpRow}>
                {followUps.map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    className={styles.followUpButton}
                    onClick={() => handleFollowUp(item)}
                    disabled={pending}
                  >
                    {item.label}
                  </button>
                ))}
              </div>
            </div>
          ) : null}

          <form className={styles.inputBar} onSubmit={handleSubmit}>
            <textarea
              className={styles.promptInput}
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about waits, telemetry, or training coverage…"
              rows={3}
              disabled={pending}
            />
            <button type="submit" className={styles.sendButton} disabled={pending || !draft.trim()}>
              {pending ? "Sending…" : "Send"}
            </button>
          </form>
        </div>
      </Panel>

      <div className={styles.supportingRow}>
        <Panel title="Runtime Snapshot" subtitle="services/localai/LocalAI/docker-compose.yaml" dense>
          <dl className={styles.runtimeGrid}>
            <div>
              <dt>Image</dt>
              <dd>{apiService?.image ?? "quay.io/go-skynet/local-ai:master"}</dd>
            </div>
            <div>
              <dt>Ports</dt>
              <dd>{exposedPorts.join(", ") || "8080:8080"}</dd>
            </div>
            <div>
              <dt>Command</dt>
              <dd>
                <code>
                  {Array.isArray(apiService?.command)
                    ? apiService.command.join(" ")
                    : apiService?.command ?? "phi-2"}
                </code>
              </dd>
            </div>
            <div>
              <dt>Base URL</dt>
              <dd>
                <a href={getBaseUrl()} target="_blank" rel="noreferrer">
                  {getBaseUrl()}
                </a>
              </dd>
            </div>
          </dl>
        </Panel>

        <Panel title="Model Inventory" subtitle={`Documented ${documentedModels.length} of ${modelOptions.length}`} dense>
          <ul className={styles.modelList}>
            {modelOptions.slice(0, 6).map((entry) => (
              <li key={entry.id} className={entry.readme ? styles.modelReady : styles.modelTodo}>
                <span>{entry.id}</span>
                <small>{entry.readme ? "Doc complete" : "Needs README"}</small>
              </li>
            ))}
            {modelOptions.length > 6 ? (
              <li className={styles.modelMore}>+{modelOptions.length - 6} more models</li>
            ) : null}
          </ul>
        </Panel>
      </div>
    </div>
  );
}
