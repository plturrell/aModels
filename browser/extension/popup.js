async function checkHealth() {
  const statusEl = document.getElementById('status');
  statusEl.textContent = 'Checking...';
  const { gatewayBaseUrl } = await chrome.storage.sync.get(['gatewayBaseUrl']);
  const base = gatewayBaseUrl || 'http://localhost:8000';
  try {
    const res = await fetch(`${base}/healthz`);
    const json = await res.json();
    statusEl.textContent = `OK: ${JSON.stringify(json)}`;
  } catch (e) {
    statusEl.textContent = `Error: ${e}`;
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
});

async function getBase() {
  const { gatewayBaseUrl } = await chrome.storage.sync.get(['gatewayBaseUrl']);
  return gatewayBaseUrl || 'http://localhost:8000';
}

async function runTelemetry() {
  const statusEl = document.getElementById('status');
  statusEl.textContent = 'Loading telemetry...';
  const base = await getBase();
  try {
    const res = await fetch(`${base}/telemetry/recent`);
    const json = await res.json();
    statusEl.textContent = `Telemetry: ${JSON.stringify(json).slice(0, 400)}...`;
  } catch (e) {
    statusEl.textContent = `Error: ${e}`;
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
  statusEl.textContent = 'Sending prompt...';
  const base = await getBase();
  try {
    const res = await fetch(`${base}/localai/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        messages: [ { role: 'user', content } ]
      })
    });
    const json = await res.json();
    // OpenAI-compatible response: choices[0].message.content
    const msg = json && json.choices && json.choices[0] && json.choices[0].message && json.choices[0].message.content ? json.choices[0].message.content : JSON.stringify(json).slice(0, 400);
    statusEl.textContent = `LocalAI: ${msg}`;
  } catch (e) {
    statusEl.textContent = `Error: ${e}`;
  }
}

async function runAgentFlow() {
  const statusEl = document.getElementById('status');
  statusEl.textContent = 'Running agentflow...';
  const base = await getBase();
  try {
    const res = await fetch(`${base}/agentflow/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ flow_id: 'demo', input: { text: 'hello' } })
    });
    const json = await res.json();
    statusEl.textContent = `AgentFlow: ${JSON.stringify(json).slice(0, 400)}...`;
  } catch (e) {
    statusEl.textContent = `Error: ${e}`;
  }
}

async function runSearch() {
  const statusEl = document.getElementById('status');
  statusEl.textContent = 'Searching...';
  const base = await getBase();
  try {
    const res = await fetch(`${base}/search/_search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: { match_all: {} }, size: 1 })
    });
    const json = await res.json();
    statusEl.textContent = `Search: ${JSON.stringify(json).slice(0, 400)}...`;
  } catch (e) {
    statusEl.textContent = `Error: ${e}`;
  }
}

async function redisSet() {
  const statusEl = document.getElementById('status');
  statusEl.textContent = 'Redis set...';
  const base = await getBase();
  try {
    const res = await fetch(`${base}/redis/set`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ key: 'demo', value: '42', ex: 60 })
    });
    const json = await res.json();
    statusEl.textContent = `Redis set: ${JSON.stringify(json)}`;
  } catch (e) {
    statusEl.textContent = `Error: ${e}`;
  }
}

async function redisGet() {
  const statusEl = document.getElementById('status');
  statusEl.textContent = 'Redis get...';
  const base = await getBase();
  try {
    const res = await fetch(`${base}/redis/get?key=demo`);
    const json = await res.json();
    statusEl.textContent = `Redis get: ${JSON.stringify(json)}`;
  } catch (e) {
    statusEl.textContent = `Error: ${e}`;
  }
}

async function runOcr() {
  const statusEl = document.getElementById('status');
  statusEl.textContent = 'Running OCR...';
  const base = await getBase();
  try {
    const res = await fetch(`${base}/extract/ocr`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: 'hello world' })
    });
    const json = await res.json();
    statusEl.textContent = `OCR: ${JSON.stringify(json)}`;
  } catch (e) {
    statusEl.textContent = `Error: ${e}`;
  }
}

async function runSql() {
  const statusEl = document.getElementById('status');
  statusEl.textContent = 'Running SQL...';
  const base = await getBase();
  try {
    const res = await fetch(`${base}/data/sql`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: 'SELECT 1 as ok', args: [] })
    });
    const json = await res.json();
    statusEl.textContent = `SQL: ${JSON.stringify(json)}`;
  } catch (e) {
    statusEl.textContent = `Error: ${e}`;
  }
}


