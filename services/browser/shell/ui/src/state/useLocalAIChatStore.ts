import { create } from "zustand";

import {
  sendLocalAIChat,
  type ChatCitation,
  type ChatRequest,
  type ChatResponse
} from "../api/localai";
import { streamLocalAIChat } from "../api/localai-streaming";
import { useTelemetryStore } from "./useTelemetryStore";

export type ChatRole = "user" | "assistant";

export interface NormalisedCitation {
  id: string;
  label: string;
  url?: string;
  snippet?: string;
}

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  createdAt: number;
  streaming?: boolean;
  citations?: NormalisedCitation[];
  error?: boolean;
}

export interface ChatFollowUp {
  id: string;
  label: string;
  prompt: string;
}

interface LocalAIChatState {
  model: string;
  messages: ChatMessage[];
  followUps: ChatFollowUp[];
  pending: boolean;
  error: string | null;
  temperature: number;
  streaming: boolean;
  setModel: (model: string) => void;
  setTemperature: (value: number) => void;
  reset: () => void;
  sendMessage: (prompt: string, opts?: { model?: string; stream?: boolean }) => Promise<void>;
  applyFollowUp: (prompt: string) => Promise<void>;
}

const SYSTEM_PROMPT =
  "You are the aModels in-browser research assistant. Focus on SGMI datasets, training metrics, and Control-M topology. Reference sources using [[n]] markers that map to citations.";

const DEFAULT_MODEL = "sap-rpt-1-oss-main";

const createId = () => `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

function normaliseCitations(entries: ChatCitation[] | undefined): NormalisedCitation[] {
  if (!entries?.length) return [];
  return entries.map((entry, index) => {
      const label =
        entry.label ||
        entry.title ||
        entry.id ||
        (typeof entry.source === "string" ? entry.source : undefined) ||
        `Source ${index + 1}`;

      return {
        id: entry.id ?? `${index}`,
        label,
        url: entry.url || (typeof entry.source === "string" ? entry.source : undefined),
        snippet: entry.snippet || entry.text
      } satisfies NormalisedCitation;
    });
}

function gatherCitations(response: ChatResponse): NormalisedCitation[] {
  const fromChoice = response.choices?.[0]?.message?.citations;
  const fromMetadata =
    (response.choices?.[0]?.message?.metadata as { citations?: ChatCitation[] } | undefined)
      ?.citations;
  const topLevel = response.citations;
  const combined = [...(fromChoice ?? []), ...(fromMetadata ?? []), ...(topLevel ?? [])];
  return normaliseCitations(combined);
}

function deriveFollowUps(content: string, citations: NormalisedCitation[]): ChatFollowUp[] {
  const suggestions: ChatFollowUp[] = [];
  const lower = content.toLowerCase();

  if (citations.length) {
    const primary = citations[0];
    suggestions.push({
      id: createId(),
      label: `Open more on ${primary.label}`,
      prompt:
        `Drill into the findings from ${primary.label}. Summarise the key data dependencies and provide actionable next steps.`
    });
  }

  if (lower.includes("dependency") || lower.includes("wait")) {
    suggestions.push({
      id: createId(),
      label: "Map downstream waits",
      prompt:
        "Highlight the downstream Control-M jobs affected by the current waits and recommend scheduling optimisations."
    });
  }

  if (lower.includes("training") || lower.includes("dataset")) {
    suggestions.push({
      id: createId(),
      label: "Compare training snapshots",
      prompt:
        "Compare the latest training dataset to the previous snapshot and surface any schema or metric drifts."
    });
  }

  suggestions.push({
    id: createId(),
    label: "Generate action plan",
    prompt:
      "Summarise the conversation so far into a concise SGMI remediation plan with three prioritised actions and required owners."
  });

  const unique = new Map<string, ChatFollowUp>();
  for (const suggestion of suggestions) {
    if (!unique.has(suggestion.prompt)) {
      unique.set(suggestion.prompt, suggestion);
    }
  }

  return Array.from(unique.values()).slice(0, 3);
}

export const useLocalAIChatStore = create<LocalAIChatState>((set, get) => ({
  model: DEFAULT_MODEL,
  messages: [],
  followUps: [],
  pending: false,
  error: null,
  temperature: 0.4,
  streaming: false,
  setModel: (model) => set({ model }),
  setTemperature: (value) => set({ temperature: value }),
  reset: () => set({ messages: [], followUps: [], pending: false, error: null, streaming: false }),
  sendMessage: async (prompt, opts) => {
    const trimmed = prompt.trim();
    if (!trimmed) return;

    const history = get().messages;
    const start = performance.now();
    const currentModel = opts?.model ?? get().model ?? DEFAULT_MODEL;
    const useStreaming = opts?.stream ?? true; // Default to streaming

    const userMessage: ChatMessage = {
      id: createId(),
      role: "user",
      content: trimmed,
      createdAt: Date.now()
    };

    set((state) => ({
      messages: [...state.messages, userMessage],
      pending: true,
      streaming: useStreaming,
      error: null,
      model: currentModel
    }));

    const requestHistory = history
      .filter((message) => message.role === "user" || message.role === "assistant")
      .map((message) => ({ role: message.role, content: message.content }));

    const request: ChatRequest = {
      model: currentModel,
      temperature: get().temperature,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        ...requestHistory,
        { role: "user", content: trimmed }
      ]
    };

    try {
      if (useStreaming) {
        // Use streaming API
        const assistantMessageId = createId();
        let fullContent = "";
        let citations: NormalisedCitation[] = [];

        const assistantMessage: ChatMessage = {
          id: assistantMessageId,
          role: "assistant",
          content: "",
          createdAt: Date.now(),
          streaming: true,
          citations: []
        };

        set((state) => ({
          messages: [...state.messages, assistantMessage]
        }));

        await streamLocalAIChat(request, {
          onContent: (content) => {
            fullContent = content;
            set((state) => ({
              messages: state.messages.map((message) =>
                message.id === assistantMessageId
                  ? { ...message, content: fullContent, streaming: true }
                  : message
              )
            }));
          },
          onComplete: (content) => {
            fullContent = content;
            const citations = gatherCitations({ choices: [{ message: { content } }] } as ChatResponse);
            const followUps = deriveFollowUps(fullContent, citations);

            set((state) => ({
              messages: state.messages.map((message) =>
                message.id === assistantMessageId
                  ? { ...message, content: fullContent, streaming: false, citations }
                  : message
              ),
              followUps,
              pending: false,
              streaming: false,
              error: null
            }));

            const durationMs = Math.round(performance.now() - start);
            useTelemetryStore.getState().recordInteraction({
              id: assistantMessageId,
              model: currentModel,
              durationMs,
              timestamp: Date.now(),
              promptChars: trimmed.length,
              completionChars: fullContent.length,
              citations: citations.length
            });
          },
          onError: (error) => {
            set((state) => ({
              pending: false,
              streaming: false,
              error: error.message,
              messages: state.messages.map((message) =>
                message.id === userMessage.id ? { ...message, error: true } : message
              )
            }));
          }
        });
      } else {
        // Use non-streaming API (fallback)
        const response = await sendLocalAIChat(request);
        const choice = response.choices?.[0];
        const content = choice?.message?.content?.trim();
        if (!content) {
          throw new Error("LocalAI returned an empty response");
        }

        const citations = gatherCitations(response);
        const followUps = deriveFollowUps(content, citations);

        const assistantMessage: ChatMessage = {
          id: createId(),
          role: "assistant",
          content,
          createdAt: Date.now(),
          streaming: false,
          citations
        };

        set((state) => ({
          messages: [...state.messages, assistantMessage],
          followUps,
          pending: false,
          streaming: false,
          error: null
        }));

        const durationMs = Math.round(performance.now() - start);
        const usage = response.usage ?? {};

        useTelemetryStore.getState().recordInteraction({
          id: assistantMessage.id,
          model: response.model ?? currentModel,
          durationMs,
          timestamp: Date.now(),
          promptChars: trimmed.length,
          completionChars: content.length,
          promptTokens: usage.prompt_tokens,
          completionTokens: usage.completion_tokens,
          citations: citations.length
        });
      }
    } catch (error) {
      const err = error instanceof Error ? error.message : String(error);
      set((state) => ({
        pending: false,
        streaming: false,
        error: err,
        messages: state.messages.map((message) =>
          message.id === userMessage.id ? { ...message, error: true } : message
        )
      }));
    }
  },
  applyFollowUp: async (prompt) => {
    await get().sendMessage(prompt);
  }
}));
