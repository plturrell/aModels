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
  const url = `${API_BASE}/localai/chat`;
  
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json"
      },
      body: JSON.stringify(request)
    });

    if (!response.ok) {
      let errorMessage = `LocalAI request failed (${response.status})`;
      try {
        const errorText = await response.text();
        if (errorText) {
          try {
            const errorJson = JSON.parse(errorText);
            errorMessage = errorJson.detail || errorJson.message || errorJson.error?.message || errorText;
          } catch {
            errorMessage = errorText;
          }
        }
      } catch {
        errorMessage = `HTTP ${response.status} ${response.statusText}`;
      }
      throw new Error(errorMessage);
    }

    const contentType = response.headers.get("content-type");
    if (contentType && contentType.includes("application/json")) {
      const text = await response.text();
      if (!text.trim()) {
        throw new Error("LocalAI returned an empty response");
      }
      return JSON.parse(text) as ChatResponse;
    }

    return (await response.json()) as ChatResponse;
  } catch (error) {
    if (error instanceof Error) {
      if (error.message.includes("Failed to fetch") || error.message.includes("NetworkError")) {
        throw new Error(`Network error: Unable to reach LocalAI service at ${url}. Check if the service is running and accessible.`);
      }
      throw error;
    }
    throw new Error(`Unexpected error: ${String(error)}`);
  }
}
