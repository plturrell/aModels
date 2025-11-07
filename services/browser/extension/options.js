function showStatus(type, message) {
  const status = document.getElementById('status');
  status.className = `status ${type} visible`;
  
  if (type === 'loading') {
    status.innerHTML = `<div class="spinner"></div><span>${message}</span>`;
  } else {
    status.textContent = message;
  }
  
  if (type === 'success') {
    setTimeout(() => {
      status.className = 'status';
    }, 3000);
  }
}

async function testConnection() {
  const input = document.getElementById('gateway-url');
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
  const saveBtn = document.getElementById('save');
  
  const url = input.value.trim() || 'http://localhost:8000';
  
  saveBtn.disabled = true;
  showStatus('loading', 'Saving...');
  
  chrome.storage.sync.set(
    { gatewayBaseUrl: url },
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
  chrome.storage.sync.get(['gatewayBaseUrl'], (res) => {
    if (res.gatewayBaseUrl) {
      input.value = res.gatewayBaseUrl;
    }
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


