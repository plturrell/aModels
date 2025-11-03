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
});

async function getBase() {
  const { gatewayBaseUrl } = await chrome.storage.sync.get(['gatewayBaseUrl']);
  return gatewayBaseUrl || 'http://localhost:8000';
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


