import React from 'react';
import ReactMarkdown from 'react-markdown';
import { Loader2, AlertCircle, CheckCircle } from 'lucide-react';

interface AnswerDisplayProps {
  content: string;
  isLoading: boolean;
  error: string | null;
}

const AnswerDisplay: React.FC<AnswerDisplayProps> = ({
  content,
  isLoading,
  error
}) => {
  if (error) {
    return (
      <div className="flex items-center space-x-2 text-red-600">
        <AlertCircle className="w-4 h-4" />
        <span className="text-sm">{error}</span>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex items-center space-x-2 text-gray-500">
        <Loader2 className="w-4 h-4 animate-spin" />
        <span className="text-sm">Generating answer...</span>
      </div>
    );
  }

  if (!content) {
    return (
      <div className="text-gray-400 text-sm italic">
        No answer available
      </div>
    );
  }

  return (
    <div className="prose prose-sm max-w-none">
      <ReactMarkdown
        components={{
          // Custom styling for markdown elements
          h1: ({ children }) => (
            <h1 className="text-lg font-semibold text-gray-900 mb-2">{children}</h1>
          ),
          h2: ({ children }) => (
            <h2 className="text-base font-semibold text-gray-900 mb-2">{children}</h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-sm font-semibold text-gray-900 mb-1">{children}</h3>
          ),
          p: ({ children }) => (
            <p className="text-sm text-gray-700 mb-2 leading-relaxed">{children}</p>
          ),
          ul: ({ children }) => (
            <ul className="list-disc list-inside text-sm text-gray-700 mb-2 space-y-1">{children}</ul>
          ),
          ol: ({ children }) => (
            <ol className="list-decimal list-inside text-sm text-gray-700 mb-2 space-y-1">{children}</ol>
          ),
          li: ({ children }) => (
            <li className="text-sm text-gray-700">{children}</li>
          ),
          code: ({ children }) => (
            <code className="bg-gray-100 text-gray-800 px-1 py-0.5 rounded text-xs font-mono">{children}</code>
          ),
          pre: ({ children }) => (
            <pre className="bg-gray-100 text-gray-800 p-3 rounded-lg text-xs font-mono overflow-x-auto mb-2">{children}</pre>
          ),
          blockquote: ({ children }) => (
            <blockquote className="border-l-4 border-blue-200 pl-3 text-sm text-gray-600 italic mb-2">{children}</blockquote>
          ),
          strong: ({ children }) => (
            <strong className="font-semibold text-gray-900">{children}</strong>
          ),
          em: ({ children }) => (
            <em className="italic text-gray-700">{children}</em>
          ),
          a: ({ children, href }) => (
            <a 
              href={href} 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-blue-600 hover:text-blue-800 underline"
            >
              {children}
            </a>
          ),
          table: ({ children }) => (
            <div className="overflow-x-auto mb-2">
              <table className="min-w-full border border-gray-200 rounded-lg text-xs">{children}</table>
            </div>
          ),
          th: ({ children }) => (
            <th className="border border-gray-200 bg-gray-50 px-2 py-1 text-left font-semibold text-gray-900">{children}</th>
          ),
          td: ({ children }) => (
            <td className="border border-gray-200 px-2 py-1 text-gray-700">{children}</td>
          )
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};

export default AnswerDisplay;
