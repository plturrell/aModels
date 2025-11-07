function showStatus(type, message) {
  const status = document.getElementById('status');
  status.className = `status ${type} visible`;
  
  if (type === 'loading') {
    status.innerHTML = `<div class="spinner"></div><span>${message}</span>`;
  } else {
    status.textContent = message;
  }
  
  // Auto-hide after 5 seconds for success messages
  if (type === 'success') {
    setTimeout(() => {
      status.className = 'status';
    }, 5000);
  }
}

async function testConnection() {
  const input = document.getElementById('gateway-url');
  const testBtn = document.getElementById('test');
  const saveBtn = document.getElementById('save');
  
  const gatewayUrl = input.value.trim() || 'http://localhost:8000';
  
  // Disable buttons during test
  testBtn.disabled = true;
  saveBtn.disabled = true;
  
  showStatus('loading', 'Testing connection...');
  
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);
    
    const response = await fetch(`${gatewayUrl}/healthz`, {
      signal: controller.signal,
      method: 'GET'
    });
    
    clearTimeout(timeoutId);
    
    if (response.ok) {
      const data = await response.json();
      showStatus('success', `✓ Connection successful!\nGateway is online and responding.`);
    } else {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
  } catch (error) {
    if (error.name === 'AbortError') {
      showStatus('error', `✗ Connection timed out\n\nThe gateway didn't respond within 5 seconds. Please check:\n• Is the gateway running?\n• Is the URL correct?\n• Are there network issues?`);
    } else if (error.message === 'Failed to fetch') {
      showStatus('error', `✗ Cannot reach gateway\n\nFailed to connect to: ${gatewayUrl}\n\nPlease verify:\n• The gateway is running\n• The URL is correct\n• Firewall allows the connection`);
    } else {
      showStatus('error', `✗ Connection failed\n\n${error.message}\n\nCheck the gateway logs for more details.`);
    }
  } finally {
    // Re-enable buttons
    testBtn.disabled = false;
    saveBtn.disabled = false;
  }
}

function saveSettings() {
  const input = document.getElementById('gateway-url');
  const browser = document.getElementById('browser-url');
  const saveBtn = document.getElementById('save');
  
  const gatewayUrl = input.value.trim() || 'http://localhost:8000';
  const browserUrl = browser.value.trim() || 'http://localhost:8070';
  
  saveBtn.disabled = true;
  showStatus('loading', 'Saving settings...');
  
  chrome.storage.sync.set(
    { 
      gatewayBaseUrl: gatewayUrl, 
      browserUrl: browserUrl,
      lastUpdated: new Date().toISOString()
    },
    () => {
      saveBtn.disabled = false;
      showStatus('success', '✓ Settings saved successfully!\n\nChanges will take effect immediately.');
    }
  );
}

document.addEventListener('DOMContentLoaded', () => {
  const input = document.getElementById('gateway-url');
  const browser = document.getElementById('browser-url');
  const testBtn = document.getElementById('test');
  const saveBtn = document.getElementById('save');

  // Load saved settings
  chrome.storage.sync.get(['gatewayBaseUrl', 'browserUrl'], (res) => {
    if (res.gatewayBaseUrl) {
      input.value = res.gatewayBaseUrl;
    }
    if (res.browserUrl) {
      browser.value = res.browserUrl;
    }
  });

  // Event listeners
  testBtn.addEventListener('click', testConnection);
  saveBtn.addEventListener('click', saveSettings);
  
  // Allow Enter key to save
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      saveSettings();
    }
  });
  
  browser.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      saveSettings();
    }
  });
});


