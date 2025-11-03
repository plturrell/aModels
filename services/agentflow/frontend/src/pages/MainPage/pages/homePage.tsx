import React from 'react';
import { Link, useParams } from 'react-router-dom';

interface HomePageProps {
  type: 'flows' | 'components' | 'mcp';
}

const HomePage: React.FC<HomePageProps> = ({ type }) => {
  const { folderId } = useParams<{ folderId?: string }>();

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 p-6">
        <h1 className="text-2xl font-semibold text-gray-900 mb-6">
          {type === 'flows' && 'Flows'}
          {type === 'components' && 'Components'}
          {type === 'mcp' && 'MCP Servers'}
        </h1>
        
        {folderId && (
          <p className="text-sm text-gray-600 mb-4">
            Folder ID: {folderId}
          </p>
        )}
        
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <p className="text-gray-500">
            {type === 'flows' && 'Manage your workflow flows here.'}
            {type === 'components' && 'Browse and manage reusable components.'}
            {type === 'mcp' && 'Configure MCP (Model Context Protocol) servers.'}
          </p>
        </div>

        {type === 'flows' && (
          <div className="mt-6">
            <div className="rounded-lg border border-indigo-200 bg-indigo-50 p-6">
              <h2 className="text-lg font-semibold text-indigo-900">
                AgentFlow Catalog
              </h2>
              <p className="mt-2 text-sm text-indigo-800">
                Import pre-built AgentFlow specifications and translate them into Langflow projects.
              </p>
              <Link
                to="../agentflow/"
                className="mt-4 inline-flex items-center rounded-md border border-indigo-500 bg-white px-4 py-2 text-sm font-medium text-indigo-700 transition hover:bg-indigo-500 hover:text-white"
              >
                Open AgentFlow Catalog
              </Link>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default HomePage;
