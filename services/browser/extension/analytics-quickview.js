/**
 * Analytics Quick View for Browser Extension
 * Provides quick access to analytics data and chart previews
 */

class AnalyticsQuickView {
  constructor() {
    this.cache = new Map();
    this.cacheTTL = 5 * 60 * 1000; // 5 minutes
    this.notifications = [];
    this.ws = null; // WebSocket connection
    this.settings = {
      autoRefresh: true,
      refreshInterval: 30000, // 30 seconds
      showNotifications: true,
      theme: 'auto',
      enableWebSocket: true
    };
    this.init();
  }

  async init() {
    // Load settings from storage
    const stored = await chrome.storage.sync.get(['analyticsSettings']);
    if (stored.analyticsSettings) {
      this.settings = { ...this.settings, ...stored.analyticsSettings };
    }

    // Set up auto-refresh if enabled
    if (this.settings.autoRefresh) {
      this.startAutoRefresh();
    }

    // Start WebSocket for real-time updates if enabled
    if (this.settings.enableWebSocket) {
      this.startWebSocket();
    }

    // Listen for settings changes
    chrome.storage.onChanged.addListener((changes) => {
      if (changes.analyticsSettings) {
        this.settings = { ...this.settings, ...changes.analyticsSettings.newValue };
        if (this.settings.autoRefresh) {
          this.startAutoRefresh();
        } else {
          this.stopAutoRefresh();
        }
      }
    });
  }

  async getBaseUrl() {
    if (window.connectionManager) {
      return window.connectionManager.gatewayUrl;
    }
    const { gatewayBaseUrl } = await chrome.storage.sync.get(['gatewayBaseUrl']);
    return gatewayBaseUrl || 'http://localhost:8000';
  }

  /**
   * Fetch analytics dashboard stats
   */
  async fetchDashboardStats() {
    const cacheKey = 'dashboard_stats';
    const cached = this.cache.get(cacheKey);
    
    if (cached && Date.now() - cached.timestamp < this.cacheTTL) {
      return cached.data;
    }

    try {
      const base = await this.getBaseUrl();
      const response = await fetch(`${base}/runtime/analytics/dashboard`, {
        method: 'GET',
        headers: { 'Accept': 'application/json' }
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();
      this.cache.set(cacheKey, { data, timestamp: Date.now() });
      
      // Dispatch event for real-time updates
      window.dispatchEvent(new CustomEvent('analytics-updated', {
        detail: { data, timestamp: Date.now() }
      }));
      
      return data;
    } catch (error) {
      console.error('Failed to fetch dashboard stats:', error);
      return null;
    }
  }

  /**
   * Start WebSocket connection for real-time updates
   */
  startWebSocket() {
    if (this.ws) {
      return; // Already connected
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/runtime/analytics/ws`;

    try {
      this.ws = new WebSocket(wsUrl);
      
      this.ws.onopen = () => {
        console.log('Analytics WebSocket connected');
        this.ws.send(JSON.stringify({ type: 'subscribe' }));
      };

      this.ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          if (message.type === 'dashboard_update' && message.stats) {
            // Update cache with new data
            const cacheKey = 'dashboard_stats';
            this.cache.set(cacheKey, { 
              data: { stats: message.stats }, 
              timestamp: Date.now() 
            });
            
            // Dispatch update event
            window.dispatchEvent(new CustomEvent('analytics-updated', {
              detail: { data: { stats: message.stats }, timestamp: Date.now() }
            }));
          }
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      this.ws.onerror = (error) => {
        console.error('Analytics WebSocket error:', error);
      };

      this.ws.onclose = () => {
        console.log('Analytics WebSocket closed');
        this.ws = null;
        // Attempt to reconnect after 5 seconds
        setTimeout(() => this.startWebSocket(), 5000);
      };
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
    }
  }

  /**
   * Stop WebSocket connection
   */
  stopWebSocket() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  /**
   * Get chart preview data
   */
  async getChartPreview(chartType, dataSource) {
    const cacheKey = `chart_${chartType}_${dataSource}`;
    const cached = this.cache.get(cacheKey);
    
    if (cached && Date.now() - cached.timestamp < this.cacheTTL) {
      return cached.data;
    }

    try {
      const stats = await this.fetchDashboardStats();
      if (!stats) return null;

      // Transform data based on chart type
      let previewData = null;
      switch (chartType) {
        case 'pie':
          previewData = this.transformForPieChart(stats, dataSource);
          break;
        case 'bar':
          previewData = this.transformForBarChart(stats, dataSource);
          break;
        case 'line':
          previewData = this.transformForLineChart(stats, dataSource);
          break;
        default:
          previewData = stats;
      }

      this.cache.set(cacheKey, { data: previewData, timestamp: Date.now() });
      return previewData;
    } catch (error) {
      console.error('Failed to get chart preview:', error);
      return null;
    }
  }

  /**
   * Transform data for pie chart
   */
  transformForPieChart(stats, dataSource) {
    if (dataSource === 'popular_elements' && stats.stats?.popular_elements) {
      return stats.stats.popular_elements.slice(0, 5).map(el => ({
        name: el.element_name || el.element_id,
        value: el.access_count || 0
      }));
    }
    return [];
  }

  /**
   * Transform data for bar chart
   */
  transformForBarChart(stats, dataSource) {
    if (dataSource === 'usage_statistics' && stats.stats?.usage_statistics) {
      const usage = stats.stats.usage_statistics;
      return {
        labels: Object.keys(usage.access_by_hour || {}),
        values: Object.values(usage.access_by_hour || {})
      };
    }
    return { labels: [], values: [] };
  }

  /**
   * Transform data for line chart
   */
  transformForLineChart(stats, dataSource) {
    if (dataSource === 'quality_trends' && stats.stats?.quality_trends) {
      return stats.stats.quality_trends.map(trend => ({
        date: trend.last_updated,
        value: trend.current_score
      }));
    }
    return [];
  }

  /**
   * Show notification
   */
  showNotification(title, message, type = 'info') {
    if (!this.settings.showNotifications) return;

    const notification = {
      id: Date.now().toString(),
      title,
      message,
      type,
      timestamp: Date.now()
    };

    this.notifications.push(notification);

    // Show browser notification if permission granted
    if (Notification.permission === 'granted') {
      new Notification(title, {
        body: message,
        icon: chrome.runtime.getURL('icon.png')
      });
    } else if (Notification.permission !== 'denied') {
      Notification.requestPermission().then(permission => {
        if (permission === 'granted') {
          new Notification(title, {
            body: message,
            icon: chrome.runtime.getURL('icon.png')
          });
        }
      });
    }

    // Dispatch custom event for popup UI
    window.dispatchEvent(new CustomEvent('analytics-notification', {
      detail: notification
    }));

    // Auto-remove after 5 seconds
    setTimeout(() => {
      this.removeNotification(notification.id);
    }, 5000);
  }

  /**
   * Remove notification
   */
  removeNotification(id) {
    this.notifications = this.notifications.filter(n => n.id !== id);
    window.dispatchEvent(new CustomEvent('analytics-notification-removed', {
      detail: { id }
    }));
  }

  /**
   * Start auto-refresh
   */
  startAutoRefresh() {
    this.stopAutoRefresh();
    this.refreshTimer = setInterval(async () => {
      const stats = await this.fetchDashboardStats();
      if (stats) {
        this.showNotification(
          'Analytics Updated',
          'Dashboard data has been refreshed',
          'info'
        );
      }
    }, this.settings.refreshInterval);
  }

  /**
   * Stop auto-refresh
   */
  stopAutoRefresh() {
    if (this.refreshTimer) {
      clearInterval(this.refreshTimer);
      this.refreshTimer = null;
    }
  }

  /**
   * Clear cache
   */
  clearCache() {
    this.cache.clear();
  }

  /**
   * Save settings
   */
  async saveSettings() {
    await chrome.storage.sync.set({ analyticsSettings: this.settings });
  }

  /**
   * Update setting
   */
  async updateSetting(key, value) {
    this.settings[key] = value;
    await this.saveSettings();
  }
}

// Initialize analytics quick view
if (typeof window !== 'undefined') {
  window.analyticsQuickView = new AnalyticsQuickView();
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
  module.exports = AnalyticsQuickView;
}

