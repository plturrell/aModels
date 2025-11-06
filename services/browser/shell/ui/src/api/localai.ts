import { API_BASE } from "./client";

export interface ChatRequestMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface ChatRequest {
  model: string;
  messages: ChatRequestMessage[];
  temperature?: number;
  max_tokens?: number;
}

export interface ChatResponseUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
}

export interface ChatCitation {
  id?: string;
  label?: string;
  title?: string;
  url?: string;
  snippet?: string;
  text?: string;
  source?: string;
}

export interface ChatChoiceMessage {
  role: "assistant";
  content: string;
  citations?: ChatCitation[];
  metadata?: Record<string, unknown>;
  followups?: string[];
}

export interface ChatChoice {
  index: number;
  message: ChatChoiceMessage;
  finish_reason?: string;
}

export interface ChatResponse {
  id?: string;
  model?: string;
  created?: number;
  choices?: ChatChoice[];
  usage?: ChatResponseUsage;
  citations?: ChatCitation[];
}

export async function sendLocalAIChat(request: ChatRequest): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE}/localai/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json"
    },
    body: JSON.stringify(request)
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(`LocalAI request failed (${response.status}): ${message}`);
  }

  return (await response.json()) as ChatResponse;
}
