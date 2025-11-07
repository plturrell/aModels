// ============================================================================
// Connection Management (now handled by ConnectionManager)
// ============================================================================

async function getBase() {
  if (window.connectionManager) {
    return window.connectionManager.gatewayUrl;
  }
  const { gatewayBaseUrl } = await chrome.storage.sync.get(['gatewayBaseUrl']);
  return gatewayBaseUrl || 'http://localhost:8000';
}

// Expose checkConnection for command palette
window.checkConnection = function() {
  if (window.connectionManager) {
    window.connectionManager.forceReconnect();
  }
};

// ============================================================================
// Status Display
// ============================================================================

function showStatus(type, message) {
  const statusEl = document.getElementById('status');
  statusEl.className = `${type} visible`;
  
  if (type === 'loading') {
    statusEl.innerHTML = `<div class="spinner"></div><span>${message}</span>`;
  } else {
    statusEl.textContent = message;
  }
  
  // Auto-hide success messages after 5 seconds
  if (type === 'success') {
    setTimeout(() => {
      statusEl.className = '';
    }, 5000);
  }
}

function hideStatus() {
  const statusEl = document.getElementById('status');
  statusEl.className = '';
}

// ============================================================================
// Enhanced Request Handler
// ============================================================================

async function makeRequest(url, options = {}, actionName = 'Request') {
  showStatus('loading', `${actionName} in progress...`);
  
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000);
    
    const res = await fetch(url, {
      ...options,
      signal: controller.signal
    });
    
    clearTimeout(timeoutId);
    
    if (!res.ok) {
      const errorText = await res.text();
      const error = new Error(`HTTP error! status: ${res.status}, message: ${errorText}`);
      throw error;
    }
    
    const json = await res.json();
    
    // Format success message
    const successMsg = formatSuccessMessage(actionName, json);
    showStatus('success', successMsg);
    
    return json;
  } catch (e) {
    console.error(`${actionName} failed:`, e);
    
    // Use error library for user-friendly messages
    const errorInfo = getUserFriendlyError(e);
    const errorMsg = formatErrorMessage(errorInfo);
    showStatus('error', errorMsg);
    
    return null;
  }
}

// ============================================================================
// Action Handlers
// ============================================================================

// Expose all actions globally for command palette
window.runOcr = async function() {
  const base = await getBase();
  await makeRequest(
    `${base}/extract/ocr`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: 'Sample text for OCR processing' })
    },
    'Text extraction'
  );
};

window.runSql = async function() {
  const base = await getBase();
  await makeRequest(
    `${base}/data/sql`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: 'SELECT 1 as result', args: [] })
    },
    'SQL query'
  );
};

window.runTelemetry = async function() {
  const base = await getBase();
  await makeRequest(
    `${base}/telemetry/recent`,
    {},
    'Telemetry fetch'
  );
};

window.openBrowser = async function() {
  const { browserUrl } = await chrome.storage.sync.get(['browserUrl']);
  const url = browserUrl || 'http://localhost:8070';
  
  try {
    await chrome.tabs.create({ url });
    showStatus('success', '✓ Browser shell opened in new tab');
  } catch (e) {
    // Fallback: try window.open
    try {
      window.open(url, '_blank');
      showStatus('success', '✓ Browser shell opened');
    } catch (err) {
      showStatus('error', `Cannot open browser shell.\nTried URL: ${url}`);
    }
  }
};

window.chatSend = async function() {
  const promptEl = document.getElementById('chat-prompt');
  const modelEl = document.getElementById('chat-model');
  const content = (promptEl.value || '').trim();
  const model = (modelEl.value || 'gpt-3.5-turbo').trim();
  
  if (!content) {
    showStatus('error', 'Please enter a prompt to send to LocalAI');
    promptEl.focus();
    return;
  }
  
  const base = await getBase();
  const data = await makeRequest(
    `${base}/localai/chat`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        messages: [{ role: 'user', content }]
      })
    },
    'LocalAI chat'
  );
  
  if (data && data.choices && data.choices[0]) {
    const response = data.choices[0].message?.content || 'No response content';
    showStatus('success', `💬 LocalAI Response:\n\n${response}`);
    
    // Clear prompt on success
    promptEl.value = '';
  }
};

window.runAgentFlow = async function() {
  const base = await getBase();
  await makeRequest(
    `${base}/agentflow/run`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ flow_id: 'demo', input: { text: 'hello' } })
    },
    'AgentFlow execution'
  );
};

window.runSearch = async function() {
  const base = await getBase();
  await makeRequest(
    `${base}/search/_search`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: { match_all: {} }, size: 1 })
    },
    'OpenSearch query'
  );
};

window.redisSet = async function() {
  const base = await getBase();
  await makeRequest(
    `${base}/redis/set`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ key: 'demo', value: '42', ex: 60 })
    },
    'Redis set'
  );
};

window.redisGet = async function() {
  const base = await getBase();
  await makeRequest(
    `${base}/redis/get?key=demo`,
    {},
    'Redis get'
  );
};

// ============================================================================
// Progressive Disclosure
// ============================================================================

window.toggleAdvanced = function() {
  const section = document.getElementById('advanced-section');
  const button = document.getElementById('toggle-advanced');
  const text = document.getElementById('toggle-text');
  
  const isVisible = section.classList.contains('visible');
  
  if (isVisible) {
    section.classList.remove('visible');
    text.textContent = 'Show Advanced Tools ▼';
    button.setAttribute('aria-expanded', 'false');
  } else {
    section.classList.add('visible');
    text.textContent = 'Hide Advanced Tools ▲';
    button.setAttribute('aria-expanded', 'true');
  }
};

// ============================================================================
// First-Run Detection
// ============================================================================

async function checkFirstRun() {
  const { setupCompleted } = await chrome.storage.sync.get(['setupCompleted']);
  
  if (!setupCompleted) {
    // Open welcome page in new tab
    chrome.tabs.create({ url: chrome.runtime.getURL('welcome.html') });
    window.close();
  }
}

// ============================================================================
// Settings & Help
// ============================================================================

function openSettings() {
  chrome.runtime.openOptionsPage();
}

function openHelp() {
  // Open help documentation or README
  chrome.tabs.create({ 
    url: 'https://github.com/yourusername/aModels/blob/main/services/browser/README.md' 
  });
}

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', async () => {
  // Check if first run
  await checkFirstRun();
  
  // Set up event listeners
  document.getElementById('run-ocr').addEventListener('click', window.runOcr);
  document.getElementById('run-sql').addEventListener('click', window.runSql);
  document.getElementById('run-telemetry').addEventListener('click', window.runTelemetry);
  document.getElementById('run-agentflow').addEventListener('click', window.runAgentFlow);
  document.getElementById('run-search').addEventListener('click', window.runSearch);
  document.getElementById('redis-set').addEventListener('click', window.redisSet);
  document.getElementById('redis-get').addEventListener('click', window.redisGet);
  document.getElementById('chat-send').addEventListener('click', window.chatSend);
  document.getElementById('open-browser').addEventListener('click', window.openBrowser);
  document.getElementById('toggle-advanced').addEventListener('click', window.toggleAdvanced);
  document.getElementById('settings-link').addEventListener('click', (e) => {
    e.preventDefault();
    openSettings();
  });
  document.getElementById('help-link').addEventListener('click', (e) => {
    e.preventDefault();
    openHelp();
  });
  
  // Enable Enter key in chat
  document.getElementById('chat-prompt').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      window.chatSend();
    }
  });
  
  // Show command palette hint
  showCommandPaletteHint();
  
  // Connection is now managed by ConnectionManager (connection-manager.js)
  // No need for manual connection checks here
});

function showCommandPaletteHint() {
  // Show a quick tip about Cmd+K on first 3 opens
  chrome.storage.local.get(['commandPaletteHintShown'], (result) => {
    const shownCount = result.commandPaletteHintShown || 0;
    
    if (shownCount < 3) {
      const hint = document.createElement('div');
      hint.className = 'quick-tip';
      hint.innerHTML = `
        <span class="quick-tip-icon">💡</span>
        <span class="quick-tip-text">
          Press <kbd>Cmd/Ctrl+K</kbd> to open the command palette for quick access to all features!
        </span>
        <button class="quick-tip-dismiss" aria-label="Dismiss hint">✕</button>
      `;
      
      // Insert after h1
      const h1 = document.querySelector('h1');
      h1.after(hint);
      
      // Dismiss handler
      hint.querySelector('.quick-tip-dismiss').addEventListener('click', () => {
        hint.remove();
      });
      
      // Increment counter
      chrome.storage.local.set({ commandPaletteHintShown: shownCount + 1 });
    }
  });
}