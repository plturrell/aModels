import React from 'react';

const FilesPage: React.FC = () => {
  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 p-6">
        <h1 className="text-2xl font-semibold text-gray-900 mb-6">
          File Management
        </h1>
        
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <p className="text-gray-500">
            Manage your files and assets here.
          </p>
        </div>
      </div>
    </div>
  );
};

export default FilesPage;
