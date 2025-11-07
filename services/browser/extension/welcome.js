// Welcome - Auto-connect and configure

async function testGatewayConnection() {
  const gatewayUrl = 'http://localhost:8000';
  
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);
    
    const response = await fetch(`${gatewayUrl}/healthz`, {
      signal: controller.signal,
      method: 'GET'
    });
    
    clearTimeout(timeoutId);
    
    if (response.ok) {
      return { success: true, url: gatewayUrl };
    } else {
      throw new Error(`HTTP ${response.status}`);
    }
  } catch (error) {
    return { 
      success: false, 
      error: error.name === 'AbortError' ? 'Connection timeout' : error.message,
      url: gatewayUrl
    };
  }
}

async function initialize() {
  const statusEl = document.getElementById('status');
  const errorEl = document.getElementById('error');
  const actionsEl = document.getElementById('actions');
  
  // Auto-test connection
  const result = await testGatewayConnection();
  
  if (result.success) {
    // Save configuration
    await chrome.storage.sync.set({
      gatewayBaseUrl: result.url,
      setupCompleted: true
    });
    
    // Show success
    statusEl.innerHTML = `
      <div class="status-icon">✓</div>
      <div class="status-text">Connected</div>
      <div class="status-detail">Gateway ready at ${result.url}</div>
    `;
    
    // Show complete button
    actionsEl.style.display = 'block';
  } else {
    // Show error
    statusEl.innerHTML = `
      <div class="status-icon">✗</div>
      <div class="status-text">Connection Failed</div>
      <div class="status-detail">Cannot reach gateway</div>
    `;
    
    errorEl.textContent = `Could not connect to ${result.url}. Please ensure the gateway is running and try reloading this page.`;
    errorEl.style.display = 'block';
  }
}

function complete() {
  // Close welcome tab
  window.close();
}

// Run on load
document.addEventListener('DOMContentLoaded', initialize);
