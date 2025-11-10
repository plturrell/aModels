/**
 * WebSocket client for real-time analytics updates
 */

export interface WebSocketMessage {
  type: string;
  timestamp?: number;
  stats?: any;
  templates?: any;
  [key: string]: any;
}

export interface WebSocketCallbacks {
  onMessage?: (message: WebSocketMessage) => void;
  onError?: (error: Event) => void;
  onOpen?: () => void;
  onClose?: () => void;
}

export class AnalyticsWebSocket {
  private ws: WebSocket | null = null;
  private url: string;
  private callbacks: WebSocketCallbacks;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectDelay = 1000; // Start with 1 second
  private reconnectTimer: number | null = null;
  private isIntentionallyClosed = false;
  private heartbeatInterval: number | null = null;

  constructor(url: string, callbacks: WebSocketCallbacks = {}) {
    this.url = url;
    this.callbacks = callbacks;
  }

  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return; // Already connected
    }

    this.isIntentionallyClosed = false;
    this.reconnectAttempts = 0;

    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        this.reconnectAttempts = 0;
        this.reconnectDelay = 1000;
        this.startHeartbeat();
        if (this.callbacks.onOpen) {
          this.callbacks.onOpen();
        }
        // Subscribe to dashboard updates
        this.send({ type: 'subscribe' });
      };

      this.ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          
          // Handle pong responses
          if (message.type === 'pong') {
            return;
          }

          if (this.callbacks.onMessage) {
            this.callbacks.onMessage(message);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        if (this.callbacks.onError) {
          this.callbacks.onError(error);
        }
      };

      this.ws.onclose = () => {
        this.stopHeartbeat();
        if (this.callbacks.onClose) {
          this.callbacks.onClose();
        }

        // Attempt to reconnect if not intentionally closed
        if (!this.isIntentionallyClosed && this.reconnectAttempts < this.maxReconnectAttempts) {
          this.scheduleReconnect();
        }
      };
    } catch (error) {
      console.error('Error creating WebSocket:', error);
      if (this.callbacks.onError) {
        this.callbacks.onError(error as Event);
      }
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
    }

    this.reconnectAttempts++;
    const delay = Math.min(
      this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1),
      30000 // Max 30 seconds
    );

    this.reconnectTimer = window.setTimeout(() => {
      console.log(`Reconnecting to WebSocket (attempt ${this.reconnectAttempts})...`);
      this.connect();
    }, delay);
  }

  private startHeartbeat(): void {
    this.stopHeartbeat();
    this.heartbeatInterval = window.setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.send({ type: 'ping' });
      }
    }, 30000); // Send ping every 30 seconds
  }

  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  send(message: WebSocketMessage): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket is not open, cannot send message');
    }
  }

  disconnect(): void {
    this.isIntentionallyClosed = true;
    this.stopHeartbeat();
    
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  getReadyState(): number | undefined {
    return this.ws?.readyState;
  }
}

/**
 * React hook for WebSocket connection
 */
import { useEffect, useRef, useState } from 'react';

export function useAnalyticsWebSocket(
  url: string | null,
  callbacks: WebSocketCallbacks = {}
): { ws: AnalyticsWebSocket | null; connected: boolean; error: Event | null } {
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<Event | null>(null);
  const wsRef = useRef<AnalyticsWebSocket | null>(null);

  useEffect(() => {
    if (!url) {
      return;
    }

    const ws = new AnalyticsWebSocket(url, {
      ...callbacks,
      onOpen: () => {
        setConnected(true);
        setError(null);
        if (callbacks.onOpen) {
          callbacks.onOpen();
        }
      },
      onClose: () => {
        setConnected(false);
        if (callbacks.onClose) {
          callbacks.onClose();
        }
      },
      onError: (err) => {
        setError(err);
        setConnected(false);
        if (callbacks.onError) {
          callbacks.onError(err);
        }
      },
      onMessage: callbacks.onMessage,
    });

    wsRef.current = ws;
    ws.connect();

    return () => {
      ws.disconnect();
      wsRef.current = null;
    };
  }, [url]);

  return {
    ws: wsRef.current,
    connected,
    error,
  };
}

