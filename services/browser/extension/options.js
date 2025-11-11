function showStatus(type, message) {
  const status = document.getElementById('status');
  // Map to SAP message classes
  const base = 'sap-message ';
  let cls = '';
  switch (type) {
    case 'success':
      cls = 'sap-message-success';
      break;
    case 'error':
      cls = 'sap-message-error';
      break;
    case 'loading':
    default:
      cls = 'sap-message-info';
      break;
  }
  status.className = `${base}${cls} visible`;
  if (type === 'loading') {
    status.innerHTML = `<div class="spinner"></div><span>${message}</span>`;
  } else {
    status.textContent = message;
  }
  if (type === 'success') {
    setTimeout(() => {
      status.className = '';
    }, 3000);
  }
}

async function testConnection() {
  const input = document.getElementById('gateway-url');
  const themeSelect = document.getElementById('sap-theme-select');
  const testBtn = document.getElementById('test');
  const saveBtn = document.getElementById('save');
  
  const url = input.value.trim() || 'http://localhost:8000';
  
  testBtn.disabled = true;
  saveBtn.disabled = true;
  
  showStatus('loading', 'Testing...');
  
  try {
    const controller = new AbortController();
    setTimeout(() => controller.abort(), 5000);
    
    const response = await fetch(`${url}/healthz`, {
      signal: controller.signal
    });
    
    if (response.ok) {
      showStatus('success', '✓ Connected');
    } else {
      throw new Error(`HTTP ${response.status}`);
    }
  } catch (error) {
    showStatus('error', '✗ Cannot connect. Is the gateway running?');
  } finally {
    testBtn.disabled = false;
    saveBtn.disabled = false;
  }
}

function saveSettings() {
  const input = document.getElementById('gateway-url');
  const themeSelect = document.getElementById('sap-theme-select');
  const saveBtn = document.getElementById('save');
  
  const url = input.value.trim() || 'http://localhost:8000';
  const sapTheme = (themeSelect.value || 'auto');
  
  saveBtn.disabled = true;
  showStatus('loading', 'Saving...');
  
  chrome.storage.sync.set(
    { gatewayBaseUrl: url, sapTheme },
    () => {
      saveBtn.disabled = false;
      showStatus('success', '✓ Saved');
    }
  );
}

document.addEventListener('DOMContentLoaded', () => {
  const input = document.getElementById('gateway-url');
  const testBtn = document.getElementById('test');
  const saveBtn = document.getElementById('save');

  // Load saved URL
  chrome.storage.sync.get(['gatewayBaseUrl', 'sapTheme'], (res) => {
    if (res.gatewayBaseUrl) {
      input.value = res.gatewayBaseUrl;
    }
    themeSelect.value = res.sapTheme || 'auto';
  });

  // Event listeners
  testBtn.addEventListener('click', testConnection);
  saveBtn.addEventListener('click', saveSettings);
  
  // Enter key saves
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      saveSettings();
    }
  });
});


