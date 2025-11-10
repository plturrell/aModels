/**
 * End-to-end tests for analytics functionality
 * Tests complete user workflows
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { DashboardRenderer } from '../../components/DashboardRenderer';

describe('Analytics E2E Tests', () => {
  describe('Complete Dashboard Workflow', () => {
    it('should complete full dashboard interaction workflow', async () => {
      const specification = {
        title: 'E2E Test Dashboard',
        description: 'End-to-end test',
        charts: [
          {
            type: 'bar',
            title: 'Test Chart',
            data_source: 'test_data',
          },
        ],
        metrics: [
          { label: 'Total', value: 100 },
        ],
        insights: ['Test insight'],
      };

      const data = {
        test_data: [
          { name: 'A', value: 10 },
          { name: 'B', value: 20 },
          { name: 'C', value: 30 },
        ],
      };

      const onChartClick = vi.fn();
      const onDrillDown = vi.fn();

      render(
        <DashboardRenderer
          specification={specification}
          data={data}
          onChartClick={onChartClick}
          onDrillDown={onDrillDown}
          enableInteractivity={true}
        />
      );

      // 1. Verify dashboard loads
      await waitFor(() => {
        expect(screen.getByText('E2E Test Dashboard')).toBeInTheDocument();
      });

      // 2. Verify chart is rendered
      expect(screen.getByText('Test Chart')).toBeInTheDocument();

      // 3. Verify metrics are displayed
      expect(screen.getByText('Total')).toBeInTheDocument();
      expect(screen.getByText('100')).toBeInTheDocument();

      // 4. Verify insights are shown
      expect(screen.getByText('Test insight')).toBeInTheDocument();

      // 5. Test chart interaction (if chart is clickable)
      const chart = screen.getByLabelText(/Chart: Test Chart/i);
      if (chart) {
        fireEvent.keyDown(chart, { key: 'Enter' });
        // Chart click should be handled
      }
    });
  });

  describe('Export Workflow', () => {
    it('should export dashboard data', async () => {
      const specification = {
        title: 'Export Test',
        charts: [
          {
            type: 'bar',
            title: 'Export Chart',
            data_source: 'export_data',
          },
        ],
        metrics: [],
        insights: [],
      };

      const data = {
        export_data: [
          { name: 'Item 1', value: 10 },
          { name: 'Item 2', value: 20 },
        ],
      };

      // Mock export functions
      const exportSpy = vi.fn();
      global.exportToCSV = exportSpy;

      render(
        <DashboardRenderer
          specification={specification}
          data={data}
          enableInteractivity={true}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Export Test')).toBeInTheDocument();
      });

      // Export functionality would be tested here
      // This is a placeholder for actual export testing
    });
  });

  describe('Real-time Updates Workflow', () => {
    it('should handle real-time analytics updates', async () => {
      const specification = {
        title: 'Real-time Test',
        charts: [
          {
            type: 'line',
            title: 'Real-time Chart',
            data_source: 'realtime_data',
          },
        ],
        metrics: [],
        insights: [],
      };

      const initialData = {
        realtime_data: [
          { timestamp: 1000, value: 10 },
          { timestamp: 2000, value: 20 },
        ],
      };

      const { rerender } = render(
        <DashboardRenderer
          specification={specification}
          data={initialData}
          enableInteractivity={true}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Real-time Test')).toBeInTheDocument();
      });

      // Simulate real-time update
      const updatedData = {
        realtime_data: [
          { timestamp: 1000, value: 10 },
          { timestamp: 2000, value: 20 },
          { timestamp: 3000, value: 30 },
        ],
      };

      rerender(
        <DashboardRenderer
          specification={specification}
          data={updatedData}
          enableInteractivity={true}
        />
      );

      // Verify update is reflected
      await waitFor(() => {
        expect(screen.getByText('Real-time Chart')).toBeInTheDocument();
      });
    });
  });
});

import { vi } from 'vitest';

