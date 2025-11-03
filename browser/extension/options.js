document.addEventListener('DOMContentLoaded', () => {
  const input = document.getElementById('gateway-url');
  const status = document.getElementById('status');
  const save = document.getElementById('save');

  chrome.storage.sync.get(['gatewayBaseUrl'], (res) => {
    if (res.gatewayBaseUrl) input.value = res.gatewayBaseUrl;
  });

  save.addEventListener('click', () => {
    const url = input.value.trim() || 'http://localhost:8000';
    chrome.storage.sync.set({ gatewayBaseUrl: url }, () => {
      status.textContent = 'Saved';
      setTimeout(() => (status.textContent = ''), 1500);
    });
  });
});


