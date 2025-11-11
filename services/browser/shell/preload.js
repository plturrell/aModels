const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('shellBridge', {
  onNavigation(callback) {
    if (typeof callback !== 'function') return () => {};
    const handler = (_, payload) => callback(payload);
    ipcRenderer.on('shell:navigation', handler);
    return () => ipcRenderer.removeListener('shell:navigation', handler);
  },
  invoke(action, payload) {
    return ipcRenderer.invoke('shell:action', { action, payload });
  },
});
