import React from 'react';

const KnowledgePage: React.FC = () => {
  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 p-6">
        <h1 className="text-2xl font-semibold text-gray-900 mb-6">
          Knowledge Bases
        </h1>
        
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <p className="text-gray-500">
            Manage your knowledge bases and document collections.
          </p>
        </div>
      </div>
    </div>
  );
};

export default KnowledgePage;
