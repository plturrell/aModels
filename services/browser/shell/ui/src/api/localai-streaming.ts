/**
 * Streaming support for LocalAI chat
 * Uses Server-Sent Events (SSE) for real-time streaming responses
 */

import { API_BASE } from "./client";
import type { ChatRequest } from "./localai";

export interface StreamingChunk {
  id?: string;
  object?: string;
  created?: number;
  model?: string;
  choices?: Array<{
    index: number;
    delta: {
      role?: string;
      content?: string;
    };
    finish_reason?: string | null;
  }>;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
}

export interface StreamingCallbacks {
  onChunk?: (chunk: StreamingChunk) => void;
  onContent?: (content: string) => void;
  onComplete?: (fullContent: string) => void;
  onError?: (error: Error) => void;
}

/**
 * Stream LocalAI chat response using Server-Sent Events
 */
export async function streamLocalAIChat(
  request: ChatRequest,
  callbacks: StreamingCallbacks = {}
): Promise<void> {
  const url = `${API_BASE}/localai/chat/stream`;
  
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify({
        ...request,
        stream: true,
      }),
    });

    if (!response.ok) {
      let errorMessage = `LocalAI streaming request failed (${response.status})`;
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

    if (!response.body) {
      throw new Error("Response body is null");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let fullContent = "";

    try {
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.trim() === "") continue;
          
          if (line.startsWith("data: ")) {
            const data = line.slice(6);
            
            if (data === "[DONE]") {
              if (callbacks.onComplete) {
                callbacks.onComplete(fullContent);
              }
              return;
            }

            try {
              const chunk: StreamingChunk = JSON.parse(data);
              
              if (callbacks.onChunk) {
                callbacks.onChunk(chunk);
              }

              // Extract content from chunk
              const content = chunk.choices?.[0]?.delta?.content;
              if (content) {
                fullContent += content;
                if (callbacks.onContent) {
                  callbacks.onContent(fullContent);
                }
              }

              // Check if stream is complete
              if (chunk.choices?.[0]?.finish_reason) {
                if (callbacks.onComplete) {
                  callbacks.onComplete(fullContent);
                }
                return;
              }
            } catch (e) {
              // Skip invalid JSON chunks
              console.warn("Failed to parse SSE chunk:", data, e);
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }

    if (fullContent && callbacks.onComplete) {
      callbacks.onComplete(fullContent);
    }
  } catch (error) {
    if (error instanceof Error) {
      if (error.message.includes("Failed to fetch") || error.message.includes("NetworkError")) {
        throw new Error(`Network error: Unable to reach LocalAI service at ${url}. Check if the service is running and accessible.`);
      }
      if (callbacks.onError) {
        callbacks.onError(error);
      }
      throw error;
    }
    const err = new Error(`Unexpected error: ${String(error)}`);
    if (callbacks.onError) {
      callbacks.onError(err);
    }
    throw err;
  }
}

