import React, { useState } from 'react';
import { ChevronDown, ChevronRight, ExternalLink, FileText, Database, Calculator, Newspaper } from 'lucide-react';

interface SourceCitationsProps {
  sources: {
    hana_data?: any[];
    search_results?: any[];
    news_results?: any[];
    calculations?: any;
  };
}

const SourceCitations: React.FC<SourceCitationsProps> = ({ sources }) => {
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    hana: false,
    regulations: false,
    news: false,
    calculations: false
  });

  const toggleSection = (section: string) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  const getSourceIcon = (type: string) => {
    switch (type) {
      case 'hana':
        return <Database className="w-4 h-4 text-blue-600" />;
      case 'regulations':
        return <FileText className="w-4 h-4 text-green-600" />;
      case 'news':
        return <Newspaper className="w-4 h-4 text-orange-600" />;
      case 'calculations':
        return <Calculator className="w-4 h-4 text-purple-600" />;
      default:
        return <FileText className="w-4 h-4 text-gray-600" />;
    }
  };

  const renderHanaData = (data: any[]) => {
    if (!data || data.length === 0) return null;

    return (
      <div className="space-y-2">
        {data.map((item, index) => (
          <div key={index} className="bg-blue-50 border border-blue-200 rounded-lg p-3">
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <h4 className="text-sm font-medium text-blue-900 mb-1">
                  Data Record #{index + 1}
                </h4>
                <div className="text-xs text-blue-700 space-y-1">
                  {Object.entries(item).map(([key, value]) => (
                    <div key={key} className="flex">
                      <span className="font-medium w-20">{key}:</span>
                      <span className="flex-1">{String(value)}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  };

  const renderSearchResults = (results: any[]) => {
    if (!results || results.length === 0) return null;

    return (
      <div className="space-y-2">
        {results.map((result, index) => (
          <div key={index} className="bg-green-50 border border-green-200 rounded-lg p-3">
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <h4 className="text-sm font-medium text-green-900 mb-1">
                  {result.title || result.id || `Document ${index + 1}`}
                </h4>
                <p className="text-xs text-green-700 mb-2 line-clamp-3">
                  {result.content || result.text || 'No preview available'}
                </p>
                {result.metadata && (
                  <div className="text-xs text-green-600">
                    <span className="font-medium">Source:</span> {result.metadata.source || 'Unknown'}
                  </div>
                )}
              </div>
              {result.url && (
                <a
                  href={result.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="ml-2 p-1 text-green-600 hover:text-green-800"
                >
                  <ExternalLink className="w-3 h-3" />
                </a>
              )}
            </div>
          </div>
        ))}
      </div>
    );
  };

  const renderNewsResults = (results: any[]) => {
    if (!results || results.length === 0) return null;

    return (
      <div className="space-y-2">
        {results.map((news, index) => (
          <div key={index} className="bg-orange-50 border border-orange-200 rounded-lg p-3">
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <h4 className="text-sm font-medium text-orange-900 mb-1">
                  {news.title || `News Item ${index + 1}`}
                </h4>
                <p className="text-xs text-orange-700 mb-2 line-clamp-3">
                  {news.content || news.summary || 'No preview available'}
                </p>
                <div className="text-xs text-orange-600">
                  <span className="font-medium">Source:</span> {news.source || 'Perplexity API'}
                </div>
              </div>
              {news.url && (
                <a
                  href={news.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="ml-2 p-1 text-orange-600 hover:text-orange-800"
                >
                  <ExternalLink className="w-3 h-3" />
                </a>
              )}
            </div>
          </div>
        ))}
      </div>
    );
  };

  const renderCalculations = (calculations: any) => {
    if (!calculations) return null;

    return (
      <div className="bg-purple-50 border border-purple-200 rounded-lg p-3">
        <div className="text-xs text-purple-700 space-y-1">
          {Object.entries(calculations).map(([key, value]) => (
            <div key={key} className="flex">
              <span className="font-medium w-24">{key}:</span>
              <span className="flex-1 font-mono">{String(value)}</span>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const hasAnySources = sources.hana_data?.length > 0 || 
                       sources.search_results?.length > 0 || 
                       sources.news_results?.length > 0 || 
                       sources.calculations;

  if (!hasAnySources) {
    return null;
  }

  return (
    <div className="mt-4 border-t border-gray-200 pt-4">
      <h3 className="text-sm font-medium text-gray-900 mb-3">Sources</h3>
      
      <div className="space-y-3">
        {/* HANA Data */}
        {sources.hana_data && sources.hana_data.length > 0 && (
          <div>
            <button
              onClick={() => toggleSection('hana')}
              className="flex items-center space-x-2 text-sm font-medium text-gray-700 hover:text-gray-900 w-full text-left"
            >
              {expandedSections.hana ? (
                <ChevronDown className="w-4 h-4" />
              ) : (
                <ChevronRight className="w-4 h-4" />
              )}
              {getSourceIcon('hana')}
              <span>HANA Data ({sources.hana_data.length})</span>
            </button>
            {expandedSections.hana && (
              <div className="mt-2">
                {renderHanaData(sources.hana_data)}
              </div>
            )}
          </div>
        )}

        {/* Search Results */}
        {sources.search_results && sources.search_results.length > 0 && (
          <div>
            <button
              onClick={() => toggleSection('regulations')}
              className="flex items-center space-x-2 text-sm font-medium text-gray-700 hover:text-gray-900 w-full text-left"
            >
              {expandedSections.regulations ? (
                <ChevronDown className="w-4 h-4" />
              ) : (
                <ChevronRight className="w-4 h-4" />
              )}
              {getSourceIcon('regulations')}
              <span>Regulations & Standards ({sources.search_results.length})</span>
            </button>
            {expandedSections.regulations && (
              <div className="mt-2">
                {renderSearchResults(sources.search_results)}
              </div>
            )}
          </div>
        )}

        {/* News Results */}
        {sources.news_results && sources.news_results.length > 0 && (
          <div>
            <button
              onClick={() => toggleSection('news')}
              className="flex items-center space-x-2 text-sm font-medium text-gray-700 hover:text-gray-900 w-full text-left"
            >
              {expandedSections.news ? (
                <ChevronDown className="w-4 h-4" />
              ) : (
                <ChevronRight className="w-4 h-4" />
              )}
              {getSourceIcon('news')}
              <span>News Articles ({sources.news_results.length})</span>
            </button>
            {expandedSections.news && (
              <div className="mt-2">
                {renderNewsResults(sources.news_results)}
              </div>
            )}
          </div>
        )}

        {/* Calculations */}
        {sources.calculations && (
          <div>
            <button
              onClick={() => toggleSection('calculations')}
              className="flex items-center space-x-2 text-sm font-medium text-gray-700 hover:text-gray-900 w-full text-left"
            >
              {expandedSections.calculations ? (
                <ChevronDown className="w-4 h-4" />
              ) : (
                <ChevronRight className="w-4 h-4" />
              )}
              {getSourceIcon('calculations')}
              <span>Calculations</span>
            </button>
            {expandedSections.calculations && (
              <div className="mt-2">
                {renderCalculations(sources.calculations)}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default SourceCitations;
