import { QARequest, QAResponse, RunStatus } from '../../types/api/qa';

const API_BASE_URL = process.env.REACT_APP_ORCHESTRATION_URL || 'http://localhost:8081';
const LANGGRAPH_URL = process.env.REACT_APP_LANGGRAPH_URL || 'http://localhost:8080';
const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8080';

export interface AskQuestionOptions {
  includeNews?: boolean;
  includeCalculations?: boolean;
  model?: string;
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

class QAAPI {
  private async makeRequest<T>(
    url: string,
    options: RequestInit = {}
  ): Promise<T> {
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': process.env.REACT_APP_API_KEY || 'supersecret',
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(
        errorData.detail || `HTTP error! status: ${response.status}`
      );
    }

    return response.json();
  }

  async askQuestion(
    question: string,
    options: AskQuestionOptions = {}
  ): Promise<QAResponse> {
    const request: QARequest = {
      question,
      include_news: options.includeNews ?? true,
      include_calculations: options.includeCalculations ?? false,
      model: options.model || 'phi-2',
    };

    return this.makeRequest<QAResponse>(`${API_BASE_URL}/qa/ask`, {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async pollRunStatus(runId: string): Promise<RunStatus> {
    return this.makeRequest<RunStatus>(`${LANGGRAPH_URL}/runs/${runId}`);
  }

  async getRunStatus(runId: string): Promise<RunStatus> {
    return this.makeRequest<RunStatus>(`${LANGGRAPH_URL}/runs/${runId}`);
  }

  async getConversationHistory(): Promise<Conversation[]> {
    // TODO: Implement when backend supports conversation history
    // For now, return mock data
    return this.getMockConversations();
  }

  async deleteConversation(conversationId: string): Promise<void> {
    // TODO: Implement when backend supports conversation deletion
    console.log('Delete conversation:', conversationId);
  }

  async exportConversation(conversationId: string): Promise<Blob> {
    // TODO: Implement when backend supports conversation export
    const conversation = await this.getConversationById(conversationId);
    const content = this.formatConversationForExport(conversation);
    return new Blob([content], { type: 'text/markdown' });
  }

  private async getConversationById(conversationId: string): Promise<Conversation> {
    // TODO: Implement when backend supports individual conversation retrieval
    const conversations = await this.getConversationHistory();
    const conversation = conversations.find(c => c.id === conversationId);
    if (!conversation) {
      throw new Error('Conversation not found');
    }
    return conversation;
  }

  private formatConversationForExport(conversation: Conversation): string {
    let content = `# ${conversation.title}\n\n`;
    content += `Created: ${conversation.createdAt.toLocaleString()}\n`;
    content += `Updated: ${conversation.updatedAt.toLocaleString()}\n\n`;
    content += `---\n\n`;

    conversation.messages.forEach((message, index) => {
      content += `## ${message.role === 'user' ? 'Question' : 'Answer'} ${index + 1}\n\n`;
      content += `${message.content}\n\n`;
      
      if (message.sources) {
        content += `### Sources\n\n`;
        
        if (message.sources.hana_data?.length) {
          content += `#### HANA Data (${message.sources.hana_data.length} records)\n\n`;
        }
        
        if (message.sources.search_results?.length) {
          content += `#### Regulations & Standards (${message.sources.search_results.length} documents)\n\n`;
        }
        
        if (message.sources.news_results?.length) {
          content += `#### News Articles (${message.sources.news_results.length} articles)\n\n`;
        }
        
        if (message.sources.calculations) {
          content += `#### Calculations\n\n`;
          Object.entries(message.sources.calculations).forEach(([key, value]) => {
            content += `- **${key}**: ${value}\n`;
          });
          content += '\n';
        }
      }
      
      content += `---\n\n`;
    });

    return content;
  }

  private getMockConversations(): Conversation[] {
    return [
      {
        id: '1',
        title: 'Data Privacy Regulations',
        createdAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000), // 2 days ago
        updatedAt: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000), // 1 day ago
        messageCount: 4,
        messages: [
          {
            id: '1-1',
            role: 'user',
            content: 'What are the main data privacy regulations I need to comply with?',
            timestamp: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000),
          },
          {
            id: '1-2',
            role: 'assistant',
            content: 'The main data privacy regulations you need to comply with include:\n\n1. **GDPR (General Data Protection Regulation)** - EU regulation\n2. **CCPA (California Consumer Privacy Act)** - California state law\n3. **PIPEDA (Personal Information Protection and Electronic Documents Act)** - Canada\n4. **LGPD (Lei Geral de Proteção de Dados)** - Brazil\n\nEach has specific requirements for data collection, processing, and user rights.',
            timestamp: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000 + 30000),
            sources: {
              search_results: [
                {
                  id: 'gdpr-doc',
                  title: 'GDPR Compliance Guide',
                  content: 'General Data Protection Regulation requirements...',
                  metadata: { source: 'EU Commission' }
                }
              ]
            }
          }
        ]
      },
      {
        id: '2',
        title: 'HANA Database Performance',
        createdAt: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000), // 5 days ago
        updatedAt: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000), // 3 days ago
        messageCount: 6,
        messages: [
          {
            id: '2-1',
            role: 'user',
            content: 'How can I optimize my HANA database performance?',
            timestamp: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000),
          },
          {
            id: '2-2',
            role: 'assistant',
            content: 'Here are key strategies for optimizing HANA database performance:\n\n1. **Memory Management** - Ensure adequate RAM allocation\n2. **Index Optimization** - Use appropriate indexes\n3. **Query Optimization** - Optimize SQL queries\n4. **Partitioning** - Implement table partitioning\n5. **Compression** - Use column store compression',
            timestamp: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000 + 45000),
            sources: {
              hana_data: [
                {
                  table_name: 'performance_metrics',
                  query_time: '2.3s',
                  memory_usage: '85%'
                }
              ]
            }
          }
        ]
      }
    ];
  }

  // WebSocket connection for real-time updates
  createWebSocketConnection(runId: string): WebSocket {
    const wsUrl = `${WS_URL}/ws/runs/${runId}`;
    return new WebSocket(wsUrl);
  }
}

export const qaAPI = new QAAPI();
export default qaAPI;
