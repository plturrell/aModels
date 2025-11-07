async function makeRequest(url, options = {}) {
  const statusEl = document.getElementById('status');
  statusEl.textContent = 'Sending request...';
  try {
    const res = await fetch(url, options);
    if (!res.ok) {
      const errorText = await res.text();
      throw new Error(`HTTP error! status: ${res.status}, message: ${errorText}`);
    }
    const json = await res.json();
    statusEl.textContent = 'Request successful.';
    return json;
  } catch (e) {
    statusEl.textContent = `Error: ${e.message}`;
    console.error(e);
    return null;
  }
}

async function checkHealth() {
  const base = await getBase();
  const data = await makeRequest(`${base}/healthz`);
  if (data) {
    document.getElementById('status').textContent = `OK: ${JSON.stringify(data)}`;
  }
}

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('check').addEventListener('click', checkHealth);
  document.getElementById('run-ocr').addEventListener('click', runOcr);
  document.getElementById('run-sql').addEventListener('click', runSql);
  document.getElementById('run-telemetry').addEventListener('click', runTelemetry);
  document.getElementById('run-agentflow').addEventListener('click', runAgentFlow);
  document.getElementById('run-search').addEventListener('click', runSearch);
  document.getElementById('redis-set').addEventListener('click', redisSet);
  document.getElementById('redis-get').addEventListener('click', redisGet);
  document.getElementById('chat-send').addEventListener('click', chatSend);
  document.getElementById('open-browser').addEventListener('click', openBrowser);
});

async function getBase() {
  const { gatewayBaseUrl } = await chrome.storage.sync.get(['gatewayBaseUrl']);
  return gatewayBaseUrl || 'http://localhost:8000';
}

async function runTelemetry() {
    const base = await getBase();
    const data = await makeRequest(`${base}/telemetry/recent`);
    if (data) {
        document.getElementById('status').textContent = `Telemetry: ${JSON.stringify(data).slice(0, 400)}...`;
    }
}

async function openBrowser() {
  const { browserUrl } = await chrome.storage.sync.get(['browserUrl']);
  const url = browserUrl || 'http://localhost:8070';
  try {
    await chrome.tabs.create({ url });
  } catch (e) {
    // Fallback: try window.open
    window.open(url, '_blank');
  }
}

async function chatSend() {
    const statusEl = document.getElementById('status');
    const promptEl = document.getElementById('chat-prompt');
    const modelEl = document.getElementById('chat-model');
    const content = (promptEl.value || '').trim();
    const model = (modelEl.value || 'gpt-3.5-turbo').trim();
    if (!content) {
        statusEl.textContent = 'Enter a prompt';
        return;
    }
    const base = await getBase();
    const data = await makeRequest(`${base}/localai/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            model,
            messages: [ { role: 'user', content } ]
        })
    });
    if (data) {
        const msg = data && data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content ? data.choices[0].message.content : JSON.stringify(data).slice(0, 400);
        statusEl.textContent = `LocalAI: ${msg}`;
    }
}

async function runAgentFlow() {
    const base = await getBase();
    const data = await makeRequest(`${base}/agentflow/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ flow_id: 'demo', input: { text: 'hello' } })
    });
    if (data) {
        document.getElementById('status').textContent = `AgentFlow: ${JSON.stringify(data).slice(0, 400)}...`;
    }
}

async function runSearch() {
    const base = await getBase();
    const data = await makeRequest(`${base}/search/_search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: { match_all: {} }, size: 1 })
    });
    if (data) {
        document.getElementById('status').textContent = `Search: ${JSON.stringify(data).slice(0, 400)}...`;
    }
}

async function redisSet() {
    const base = await getBase();
    const data = await makeRequest(`${base}/redis/set`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key: 'demo', value: '42', ex: 60 })
    });
    if (data) {
        document.getElementById('status').textContent = `Redis set: ${JSON.stringify(data)}`;
    }
}

async function redisGet() {
    const base = await getBase();
    const data = await makeRequest(`${base}/redis/get?key=demo`);
    if (data) {
        document.getElementById('status').textContent = `Redis get: ${JSON.stringify(data)}`;
    }
}

async function runOcr() {
    const base = await getBase();
    const data = await makeRequest(`${base}/extract/ocr`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: 'hello world' })
    });
    if (data) {
        document.getElementById('status').textContent = `OCR: ${JSON.stringify(data)}`;
    }
}

async function runSql() {
    const base = await getBase();
    const data = await makeRequest(`${base}/data/sql`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: 'SELECT 1 as ok', args: [] })
    });
    if (data) {
        document.getElementById('status').textContent = `SQL: ${JSON.stringify(data)}`;
    }
}