/**
 * Connection Manager - Auto-reconnect with exponential backoff
 * Manages gateway connectivity with intelligent retry logic
 */

class ConnectionManager {
  constructor() {
    this.isConnected = false;
    this.gatewayUrl = 'http://localhost:8000';
    this.checkInterval = null;
    this.retryTimeout = null;
    
    // Exponential backoff configuration
    this.retryAttempts = 0;
    this.maxRetryAttempts = 10;
    this.baseRetryDelay = 1000; // 1 second
    this.maxRetryDelay = 60000; // 60 seconds
    
    // Connection history for reliability scoring
    this.connectionHistory = [];
    this.maxHistorySize = 20;
    
    // Event listeners
    this.listeners = {
      connected: [],
      disconnected: [],
      reconnecting: [],
      error: []
    };
    
    this.loadConfiguration();
    this.startMonitoring();
  }
  
  async loadConfiguration() {
    const { gatewayBaseUrl } = await chrome.storage.sync.get(['gatewayBaseUrl']);
    this.gatewayUrl = gatewayBaseUrl || 'http://localhost:8000';
  }
  
  // Event system
  on(event, callback) {
    if (this.listeners[event]) {
      this.listeners[event].push(callback);
    }
  }
  
  emit(event, data) {
    if (this.listeners[event]) {
      this.listeners[event].forEach(callback => callback(data));
    }
  }
  
  // Start monitoring connection
  startMonitoring() {
    // Initial check
    this.checkConnection();
    
    // Regular checks every 30 seconds when connected
    this.checkInterval = setInterval(() => {
      if (this.retryAttempts === 0) {
        this.checkConnection();
      }
    }, 30000);
  }
  
  stopMonitoring() {
    if (this.checkInterval) {
      clearInterval(this.checkInterval);
      this.checkInterval = null;
    }
    if (this.retryTimeout) {
      clearTimeout(this.retryTimeout);
      this.retryTimeout = null;
    }
  }
  
  async checkConnection() {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3000);
      
      const response = await fetch(`${this.gatewayUrl}/healthz`, {
        signal: controller.signal,
        method: 'GET'
      });
      
      clearTimeout(timeoutId);
      
      if (response.ok) {
        this.handleConnectionSuccess();
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      this.handleConnectionFailure(error);
    }
  }
  
  handleConnectionSuccess() {
    const wasDisconnected = !this.isConnected;
    
    this.isConnected = true;
    this.retryAttempts = 0;
    this.addToHistory(true);
    
    if (wasDisconnected) {
      console.log('✅ Gateway connection restored');
      this.emit('connected', { 
        url: this.gatewayUrl,
        timestamp: Date.now()
      });
    }
    
    // Cancel any pending retry
    if (this.retryTimeout) {
      clearTimeout(this.retryTimeout);
      this.retryTimeout = null;
    }
  }
  
  handleConnectionFailure(error) {
    const wasConnected = this.isConnected;
    
    this.isConnected = false;
    this.addToHistory(false);
    
    if (wasConnected) {
      console.warn('❌ Gateway connection lost:', error.message);
      this.emit('disconnected', { 
        url: this.gatewayUrl,
        error: error.message,
        timestamp: Date.now()
      });
    }
    
    // Start retry logic if not already retrying
    if (this.retryAttempts < this.maxRetryAttempts && !this.retryTimeout) {
      this.scheduleRetry();
    } else if (this.retryAttempts >= this.maxRetryAttempts) {
      console.error('🛑 Max retry attempts reached. Manual intervention required.');
      this.emit('error', { 
        type: 'max_retries',
        message: 'Gateway connection failed after maximum retry attempts'
      });
    }
  }
  
  scheduleRetry() {
    // Calculate delay with exponential backoff
    const delay = Math.min(
      this.baseRetryDelay * Math.pow(2, this.retryAttempts),
      this.maxRetryDelay
    );
    
    // Add jitter to prevent thundering herd
    const jitter = Math.random() * 0.3 * delay;
    const finalDelay = delay + jitter;
    
    this.retryAttempts++;
    
    console.log(`🔄 Retry attempt ${this.retryAttempts}/${this.maxRetryAttempts} in ${Math.round(finalDelay/1000)}s`);
    
    this.emit('reconnecting', {
      attempt: this.retryAttempts,
      maxAttempts: this.maxRetryAttempts,
      delayMs: finalDelay
    });
    
    this.retryTimeout = setTimeout(() => {
      this.retryTimeout = null;
      this.checkConnection();
    }, finalDelay);
  }
  
  // Force immediate reconnect attempt
  forceReconnect() {
    if (this.retryTimeout) {
      clearTimeout(this.retryTimeout);
      this.retryTimeout = null;
    }
    this.retryAttempts = 0;
    this.checkConnection();
  }
  
  // Manual retry (reset retry counter)
  resetAndRetry() {
    this.retryAttempts = 0;
    if (this.retryTimeout) {
      clearTimeout(this.retryTimeout);
      this.retryTimeout = null;
    }
    this.checkConnection();
  }
  
  // Connection history tracking
  addToHistory(success) {
    this.connectionHistory.push({
      success,
      timestamp: Date.now()
    });
    
    // Keep only last N entries
    if (this.connectionHistory.length > this.maxHistorySize) {
      this.connectionHistory.shift();
    }
  }
  
  // Calculate reliability score (0-100)
  getReliabilityScore() {
    if (this.connectionHistory.length === 0) return 100;
    
    const successCount = this.connectionHistory.filter(h => h.success).length;
    return Math.round((successCount / this.connectionHistory.length) * 100);
  }
  
  // Get connection stats
  getStats() {
    const now = Date.now();
    const recentFailures = this.connectionHistory
      .filter(h => !h.success && (now - h.timestamp) < 300000) // Last 5 minutes
      .length;
    
    return {
      isConnected: this.isConnected,
      gatewayUrl: this.gatewayUrl,
      retryAttempts: this.retryAttempts,
      maxRetryAttempts: this.maxRetryAttempts,
      reliabilityScore: this.getReliabilityScore(),
      recentFailures,
      nextRetryIn: this.retryTimeout ? this.getNextRetryDelay() : null
    };
  }
  
  getNextRetryDelay() {
    if (!this.retryTimeout) return null;
    // This is approximate since we don't track exact retry time
    return Math.min(
      this.baseRetryDelay * Math.pow(2, this.retryAttempts - 1),
      this.maxRetryDelay
    );
  }
  
  // Update gateway URL
  updateGatewayUrl(newUrl) {
    if (this.gatewayUrl !== newUrl) {
      this.gatewayUrl = newUrl;
      this.retryAttempts = 0;
      this.connectionHistory = [];
      this.checkConnection();
    }
  }
  
  // Cleanup
  destroy() {
    this.stopMonitoring();
    this.listeners = {
      connected: [],
      disconnected: [],
      reconnecting: [],
      error: []
    };
  }
}

// UI Integration
class ConnectionStatusUI {
  constructor(connectionManager) {
    this.connectionManager = connectionManager;
    this.statusElement = null;
    this.dotElement = null;
    this.textElement = null;
    
    this.init();
  }
  
  init() {
    this.statusElement = document.getElementById('connection-status');
    if (!this.statusElement) {
      console.warn('Connection status element not found');
      return;
    }
    
    this.dotElement = this.statusElement.querySelector('.status-dot');
    this.textElement = this.statusElement.querySelector('span');
    
    // Listen to connection events
    this.connectionManager.on('connected', () => this.updateUI('connected'));
    this.connectionManager.on('disconnected', () => this.updateUI('disconnected'));
    this.connectionManager.on('reconnecting', (data) => this.updateUI('reconnecting', data));
    this.connectionManager.on('error', (data) => this.updateUI('error', data));
    
    // Click to force reconnect
    this.statusElement.addEventListener('click', () => {
      if (!this.connectionManager.isConnected) {
        this.connectionManager.forceReconnect();
      }
    });
    this.statusElement.style.cursor = 'pointer';
    this.statusElement.title = 'Click to retry connection';
  }
  
  updateUI(state, data) {
    if (!this.statusElement) return;
    
    switch (state) {
      case 'connected':
        this.statusElement.className = 'connection-status connected';
        this.dotElement.className = 'status-dot connected';
        this.textElement.textContent = '✓ Connected to gateway';
        this.enableButtons();
        break;
        
      case 'disconnected':
        this.statusElement.className = 'connection-status disconnected';
        this.dotElement.className = 'status-dot disconnected';
        this.textElement.textContent = '✗ Gateway offline - Click to retry';
        this.disableButtons();
        break;
        
      case 'reconnecting':
        this.statusElement.className = 'connection-status checking';
        this.dotElement.className = 'status-dot checking';
        this.textElement.textContent = `🔄 Reconnecting (attempt ${data.attempt}/${data.maxAttempts})...`;
        break;
        
      case 'error':
        this.statusElement.className = 'connection-status disconnected';
        this.dotElement.className = 'status-dot disconnected';
        this.textElement.textContent = '⚠️ Connection failed - Check settings';
        break;
    }
  }
  
  enableButtons() {
    document.querySelectorAll('button:not(#toggle-advanced)').forEach(btn => {
      btn.disabled = false;
    });
  }
  
  disableButtons() {
    document.querySelectorAll('button:not(#toggle-advanced):not(#settings-link):not(#help-link)').forEach(btn => {
      btn.disabled = true;
    });
  }
}

// Initialize
window.connectionManager = new ConnectionManager();

// Initialize UI when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    window.connectionStatusUI = new ConnectionStatusUI(window.connectionManager);
  });
} else {
  window.connectionStatusUI = new ConnectionStatusUI(window.connectionManager);
}
