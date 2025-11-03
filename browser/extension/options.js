document.addEventListener('DOMContentLoaded', () => {
  const input = document.getElementById('gateway-url');
  const status = document.getElementById('status');
  const browser = document.getElementById('browser-url');
  const save = document.getElementById('save');

  chrome.storage.sync.get(['gatewayBaseUrl', 'browserUrl'], (res) => {
    if (res.gatewayBaseUrl) input.value = res.gatewayBaseUrl;
    if (res.browserUrl) browser.value = res.browserUrl;
  });

  save.addEventListener('click', () => {
    const url = input.value.trim() || 'http://localhost:8000';
    chrome.storage.sync.set({ gatewayBaseUrl: url, browserUrl: (browser.value || '').trim() }, () => {
      status.textContent = 'Saved';
      setTimeout(() => (status.textContent = ''), 1500);
    });
  });
});


