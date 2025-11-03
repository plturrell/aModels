import React, { useState, useEffect, useRef } from 'react';
import { Send, Loader2, MessageSquare, Settings } from 'lucide-react';
import QuestionInput from './QuestionInput';
import AnswerDisplay from './AnswerDisplay';
import SourceCitations from './SourceCitations';
import ConversationHistory from './ConversationHistory';
import { useQAStore } from '../../stores/qaStore';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  sources?: any;
  runId?: string;
}

const QAPage: React.FC = () => {
  const [currentMessage, setCurrentMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  const {
    conversations,
    currentConversation,
    isLoading: storeLoading,
    error,
    askQuestion,
    selectConversation,
    createNewConversation
  } = useQAStore();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [currentConversation?.messages]);

  const handleSubmit = async (question: string, options: any) => {
    if (!question.trim() || isLoading) return;

    setIsLoading(true);
    try {
      await askQuestion(question, options);
      setCurrentMessage('');
    } catch (err) {
      console.error('Failed to ask question:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleNewConversation = () => {
    createNewConversation();
  };

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar - Conversation History */}
      <div className="w-80 bg-white border-r border-gray-200 flex flex-col">
        <div className="p-4 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-gray-900">Q&A Assistant</h2>
            <button
              onClick={handleNewConversation}
              className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
              title="New Conversation"
            >
              <MessageSquare className="w-5 h-5" />
            </button>
          </div>
        </div>
        
        <ConversationHistory 
          conversations={conversations}
          currentConversation={currentConversation}
          onSelectConversation={selectConversation}
        />
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="bg-white border-b border-gray-200 p-4">
          <h1 className="text-xl font-semibold text-gray-900">
            {currentConversation?.title || 'New Conversation'}
          </h1>
          {currentConversation?.createdAt && (
            <p className="text-sm text-gray-500">
              Created {currentConversation.createdAt.toLocaleDateString()}
            </p>
          )}
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {currentConversation?.messages.map((message) => (
            <div key={message.id} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-3xl ${message.role === 'user' ? 'bg-blue-500 text-white' : 'bg-white border border-gray-200'} rounded-lg p-4 shadow-sm`}>
                <div className="prose prose-sm max-w-none">
                  <AnswerDisplay 
                    content={message.content}
                    isLoading={false}
                    error={null}
                  />
                </div>
                
                {message.sources && message.role === 'assistant' && (
                  <div className="mt-4">
                    <SourceCitations sources={message.sources} />
                  </div>
                )}
                
                <div className="text-xs opacity-70 mt-2">
                  {message.timestamp.toLocaleTimeString()}
                </div>
              </div>
            </div>
          ))}
          
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
                <div className="flex items-center space-x-2">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span className="text-sm text-gray-500">Thinking...</span>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="bg-white border-t border-gray-200 p-4">
          <QuestionInput
            onSubmit={handleSubmit}
            isLoading={isLoading}
            placeholder="Ask a question about regulations, data, or anything else..."
          />
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 border-t border-red-200 p-4">
            <div className="text-sm text-red-600">
              Error: {error}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default QAPage;
