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
});


