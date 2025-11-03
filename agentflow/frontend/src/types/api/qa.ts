// API Types for Q&A functionality

export interface QARequest {
  question: string;
  include_news?: boolean;
  include_calculations?: boolean;
  model?: string;
}

export interface QAResponse {
  run_id: string;
  status: string;
}

export interface RunStatus {
  run_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  result?: {
    answer: string;
    sources?: {
      hana_data?: any[];
      search_results?: any[];
      news_results?: any[];
      calculations?: any;
    };
  };
  error?: string;
  created_at?: string;
  updated_at?: string;
}

export interface Conversation {
  id: string;
  title: string;
  createdAt: Date;
  updatedAt: Date;
  messageCount: number;
  messages: Message[];
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  sources?: {
    hana_data?: any[];
    search_results?: any[];
    news_results?: any[];
    calculations?: any;
  };
  runId?: string;
}

export interface AskQuestionOptions {
  includeNews?: boolean;
  includeCalculations?: boolean;
  model?: string;
}
