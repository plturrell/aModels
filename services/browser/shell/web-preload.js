const { ipcRenderer } = require('electron');

function notify() {
  ipcRenderer.send('shell:web-state', {
    url: window.location.href,
    title: document.title,
    selection: window.getSelection ? String(window.getSelection()) : '',
    text: document.body ? document.body.innerText.slice(0, 20000) : '',
  });
}

['DOMContentLoaded', 'load'].forEach((event) => {
  window.addEventListener(event, () => {
    setTimeout(notify, 50);
  });
});

const { pushState, replaceState } = window.history;
window.history.pushState = function (...args) {
  const result = pushState.apply(this, args);
  notify();
  return result;
};
window.history.replaceState = function (...args) {
  const result = replaceState.apply(this, args);
  notify();
  return result;
};

window.addEventListener('selectionchange', () => {
  notify();
});

ipcRenderer.on('shell:highlight', (_, payload = {}) => {
  const phrases = Array.isArray(payload.phrases) ? payload.phrases : [];
  clearHighlights();
  phrases.forEach((phrase) => {
    if (!phrase) return;
    highlightPhrase(String(phrase));
  });
});

function highlightPhrase(phrase) {
  if (!phrase || !window.find) return;
  const selection = window.getSelection();
  if (!selection) return;
  selection.removeAllRanges();
  let found = window.find(phrase, false, false, false, false, false, false);
  while (found) {
    const range = selection.rangeCount ? selection.getRangeAt(0).cloneRange() : null;
    if (range) {
      const highlight = document.createElement('mark');
      highlight.dataset.shellHighlight = '1';
      highlight.textContent = range.toString();
      range.deleteContents();
      range.insertNode(highlight);
    }
    found = window.find(phrase, false, false, false, false, false, false);
  }
  selection.removeAllRanges();
}

function clearHighlights() {
  document.querySelectorAll('mark[data-shell-highlight="1"]').forEach((mark) => {
    const parent = mark.parentNode;
    while (mark.firstChild) {
      parent.insertBefore(mark.firstChild, mark);
    }
    parent.removeChild(mark);
    parent.normalize();
  });
}
