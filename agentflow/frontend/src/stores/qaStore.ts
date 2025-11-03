import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import qaAPI, { Conversation, Message, AskQuestionOptions } from '../API/qaAPI';

interface QAState {
  // State
  conversations: Conversation[];
  currentConversation: Conversation | null;
  isLoading: boolean;
  error: string | null;
  
  // Actions
  askQuestion: (question: string, options?: AskQuestionOptions) => Promise<void>;
  selectConversation: (conversation: Conversation) => void;
  createNewConversation: () => void;
  loadConversationHistory: () => Promise<void>;
  deleteConversation: (conversationId: string) => Promise<void>;
  exportConversation: (conversationId: string) => Promise<Blob>;
  clearError: () => void;
  
  // Internal actions
  addMessageToCurrentConversation: (message: Message) => void;
  updateCurrentConversationTitle: (title: string) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

const generateId = () => Math.random().toString(36).substr(2, 9);

const generateConversationTitle = (question: string): string => {
  // Generate a title from the first question
  const words = question.trim().split(' ');
  if (words.length <= 4) {
    return question.trim();
  }
  return words.slice(0, 4).join(' ') + '...';
};

export const useQAStore = create<QAState>()(
  persist(
    (set, get) => ({
      // Initial state
      conversations: [],
      currentConversation: null,
      isLoading: false,
      error: null,

      // Actions
      askQuestion: async (question: string, options: AskQuestionOptions = {}) => {
        const state = get();
        
        try {
          state.setLoading(true);
          state.clearError();

          // Create new conversation if none exists
          if (!state.currentConversation) {
            state.createNewConversation();
          }

          // Add user message
          const userMessage: Message = {
            id: generateId(),
            role: 'user',
            content: question,
            timestamp: new Date(),
          };
          state.addMessageToCurrentConversation(userMessage);

          // Update conversation title if it's the first message
          if (state.currentConversation?.messages.length === 1) {
            const title = generateConversationTitle(question);
            state.updateCurrentConversationTitle(title);
          }

          // Call API
          const response = await qaAPI.askQuestion(question, options);
          
          // Poll for completion
          const pollForCompletion = async (runId: string) => {
            const maxAttempts = 60; // 5 minutes max
            let attempts = 0;
            
            const poll = async (): Promise<void> => {
              try {
                const status = await qaAPI.pollRunStatus(runId);
                
                if (status.status === 'completed') {
                  // Add assistant message
                  const assistantMessage: Message = {
                    id: generateId(),
                    role: 'assistant',
                    content: status.result?.answer || 'No answer generated',
                    timestamp: new Date(),
                    sources: status.result?.sources,
                    runId: runId,
                  };
                  state.addMessageToCurrentConversation(assistantMessage);
                  state.setLoading(false);
                  return;
                }
                
                if (status.status === 'failed') {
                  state.setError(status.error || 'Request failed');
                  state.setLoading(false);
                  return;
                }
                
                // Continue polling
                attempts++;
                if (attempts < maxAttempts) {
                  setTimeout(poll, 5000); // Poll every 5 seconds
                } else {
                  state.setError('Request timed out');
                  state.setLoading(false);
                }
              } catch (error) {
                state.setError(`Polling failed: ${error}`);
                state.setLoading(false);
              }
            };
            
            poll();
          };

          // Start polling
          if (response.run_id) {
            pollForCompletion(response.run_id);
          } else {
            state.setError('No run ID received');
            state.setLoading(false);
          }

        } catch (error) {
          state.setError(`Failed to ask question: ${error}`);
          state.setLoading(false);
        }
      },

      selectConversation: (conversation: Conversation) => {
        set({ currentConversation: conversation });
      },

      createNewConversation: () => {
        const newConversation: Conversation = {
          id: generateId(),
          title: 'New Conversation',
          createdAt: new Date(),
          updatedAt: new Date(),
          messageCount: 0,
          messages: [],
        };

        set((state) => ({
          conversations: [newConversation, ...state.conversations],
          currentConversation: newConversation,
        }));
      },

      loadConversationHistory: async () => {
        try {
          const conversations = await qaAPI.getConversationHistory();
          set({ conversations });
        } catch (error) {
          console.error('Failed to load conversation history:', error);
        }
      },

      deleteConversation: async (conversationId: string) => {
        try {
          await qaAPI.deleteConversation(conversationId);
          set((state) => ({
            conversations: state.conversations.filter(c => c.id !== conversationId),
            currentConversation: state.currentConversation?.id === conversationId 
              ? null 
              : state.currentConversation,
          }));
        } catch (error) {
          console.error('Failed to delete conversation:', error);
        }
      },

      exportConversation: async (conversationId: string) => {
        try {
          return await qaAPI.exportConversation(conversationId);
        } catch (error) {
          console.error('Failed to export conversation:', error);
          throw error;
        }
      },

      clearError: () => {
        set({ error: null });
      },

      // Internal actions
      addMessageToCurrentConversation: (message: Message) => {
        set((state) => {
          if (!state.currentConversation) return state;

          const updatedConversation = {
            ...state.currentConversation,
            messages: [...state.currentConversation.messages, message],
            messageCount: state.currentConversation.messages.length + 1,
            updatedAt: new Date(),
          };

          return {
            currentConversation: updatedConversation,
            conversations: state.conversations.map(c =>
              c.id === state.currentConversation!.id ? updatedConversation : c
            ),
          };
        });
      },

      updateCurrentConversationTitle: (title: string) => {
        set((state) => {
          if (!state.currentConversation) return state;

          const updatedConversation = {
            ...state.currentConversation,
            title,
            updatedAt: new Date(),
          };

          return {
            currentConversation: updatedConversation,
            conversations: state.conversations.map(c =>
              c.id === state.currentConversation!.id ? updatedConversation : c
            ),
          };
        });
      },

      setLoading: (loading: boolean) => {
        set({ isLoading: loading });
      },

      setError: (error: string | null) => {
        set({ error });
      },
    }),
    {
      name: 'qa-store',
      partialize: (state) => ({
        conversations: state.conversations,
        currentConversation: state.currentConversation,
      }),
    }
  )
);
