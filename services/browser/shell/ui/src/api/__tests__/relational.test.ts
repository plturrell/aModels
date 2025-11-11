import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  getRelationalProcessingStatus,
  getRelationalProcessingResults,
  getRelationalIntelligence,
  getRelationalRequestHistory,
  searchRelationalTables,
  processRelationalTables,
} from '../relational';

// Mock fetch globally
global.fetch = vi.fn();

describe('Relational API', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('getRelationalProcessingStatus', () => {
    it('should fetch processing status successfully', async () => {
      const mockStatus = {
        request_id: 'test-123',
        query: 'test query',
        status: 'completed' as const,
        created_at: '2024-01-01T00:00:00Z',
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockStatus,
      });

      const result = await getRelationalProcessingStatus('test-123');
      expect(result).toEqual(mockStatus);
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/relational/status/test-123'),
        expect.objectContaining({
          headers: { Accept: 'application/json' },
        })
      );
    });

    it('should throw error on failed request', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: false,
        statusText: 'Not Found',
      });

      await expect(
        getRelationalProcessingStatus('invalid-id')
      ).rejects.toThrow('Failed to fetch relational status');
    });
  });

  describe('getRelationalProcessingResults', () => {
    it('should fetch processing results successfully', async () => {
      const mockResults = {
        request_id: 'test-123',
        query: 'test query',
        status: 'completed',
        documents: [],
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResults,
      });

      const result = await getRelationalProcessingResults('test-123');
      expect(result).toEqual(mockResults);
    });
  });

  describe('searchRelationalTables', () => {
    it('should search tables successfully', async () => {
      const mockQuery = {
        query: 'test search',
        top_k: 5,
      };

      const mockResults = {
        query: 'test search',
        results: [
          {
            document_id: 'doc-1',
            title: 'Test Document',
            score: 0.95,
            content: 'Test content',
          },
        ],
        count: 1,
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResults,
      });

      const result = await searchRelationalTables(mockQuery);
      expect(result).toEqual(mockResults);
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/relational/search'),
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
          }),
        })
      );
    });
  });

  describe('processRelationalTables', () => {
    it('should process tables and return request info', async () => {
      const mockParams = {
        table: 'test_table',
        async: true,
      };

      const mockResponse = {
        status: 'processing',
        request_id: 'req-123',
        message: 'Processing started',
        status_url: '/api/relational/status/req-123',
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const result = await processRelationalTables(mockParams);
      expect(result).toEqual(mockResponse);
      expect(result.request_id).toBe('req-123');
    });
  });
});
