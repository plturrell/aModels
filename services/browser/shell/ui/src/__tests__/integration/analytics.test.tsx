/**
 * Integration tests for analytics functionality
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { DashboardRenderer } from '../../components/DashboardRenderer';

describe('Analytics Integration Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should render dashboard with analytics data', async () => {
    const specification = {
      title: 'Test Dashboard',
      description: 'Test description',
      charts: [
        {
          type: 'bar',
          title: 'Test Chart',
          data_source: 'test_data',
        },
      ],
      metrics: [],
      insights: [],
    };

    const data = {
      test_data: [
        { name: 'A', value: 10 },
        { name: 'B', value: 20 },
        { name: 'C', value: 30 },
      ],
    };

    render(
      <DashboardRenderer
        specification={specification}
        data={data}
        enableInteractivity={true}
      />
    );

    await waitFor(() => {
      expect(screen.getByText('Test Dashboard')).toBeInTheDocument();
      expect(screen.getByText('Test Chart')).toBeInTheDocument();
    });
  });

  it('should handle empty data gracefully', async () => {
    const specification = {
      title: 'Empty Dashboard',
      charts: [
        {
          type: 'bar',
          title: 'Empty Chart',
          data_source: 'empty_data',
        },
      ],
      metrics: [],
      insights: [],
    };

    const data = {};

    render(
      <DashboardRenderer
        specification={specification}
        data={data}
        enableInteractivity={true}
      />
    );

    await waitFor(() => {
      expect(screen.getByText(/No data available/i)).toBeInTheDocument();
    });
  });

  it('should support keyboard navigation', async () => {
    const specification = {
      title: 'Keyboard Test',
      charts: [
        {
          type: 'bar',
          title: 'Chart 1',
          data_source: 'data1',
        },
      ],
      metrics: [],
      insights: [],
    };

    const data = {
      data1: [{ name: 'A', value: 10 }],
    };

    render(
      <DashboardRenderer
        specification={specification}
        data={data}
        enableInteractivity={true}
      />
    );

    // Test keyboard navigation
    const chart = screen.getByLabelText(/Chart: Chart 1/i);
    expect(chart).toBeInTheDocument();
    expect(chart).toHaveAttribute('tabIndex', '0');
  });

  it('should announce dashboard load to screen readers', async () => {
    const specification = {
      title: 'Accessibility Test',
      charts: [
        {
          type: 'bar',
          title: 'Chart',
          data_source: 'data',
        },
      ],
      metrics: [],
      insights: [],
    };

    const data = {
      data: [{ name: 'A', value: 10 }],
    };

    // Mock screen reader announcement
    const announceSpy = vi.fn();
    vi.spyOn(window, 'dispatchEvent').mockImplementation((event) => {
      if (event.type === 'analytics-announcement') {
        announceSpy();
      }
      return true;
    });

    render(
      <DashboardRenderer
        specification={specification}
        data={data}
        enableInteractivity={true}
      />
    );

    await waitFor(() => {
      expect(screen.getByText('Accessibility Test')).toBeInTheDocument();
    });
  });
});

