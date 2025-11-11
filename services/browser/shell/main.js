const { app, BrowserWindow, BrowserView, session, ipcMain } = require('electron');
const path = require('path');
const fs = require('fs');
const { pathToFileURL } = require('url');

let mainWindow;
let webView;
let panelView;
let latestPageState = {
  url: '',
  title: '',
  selection: '',
  text: '',
};
const defaultHomeUrl = (() => {
  const homePath = path.join(__dirname, 'home', 'index.html');
  if (fs.existsSync(homePath)) {
    return pathToFileURL(homePath).href;
  }
  return 'about:blank';
})();
const repoRootPath = path.resolve(__dirname, '../../..');

function createSplitLayout() {
  if (!mainWindow || !webView || !panelView) return;
  const [width, height] = mainWindow.getContentSize();
  const panelWidth = Math.min(Math.max(Math.floor(width * 0.28), 340), 520);
  webView.setBounds({ x: 0, y: 0, width: width - panelWidth, height });
  panelView.setBounds({ x: width - panelWidth, y: 0, width: panelWidth, height });
  webView.setAutoResize({ width: true, height: true });
  panelView.setAutoResize({ height: true });
}

async function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    title: 'aModels Browser Shell',
    backgroundColor: '#0f172a',
    webPreferences: {
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
    },
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  const extensionPath = path.resolve(__dirname, '../extension');
  try {
    await session.defaultSession.loadExtension(extensionPath, {
      allowFileAccess: true,
    });
  } catch (error) {
    console.warn('Failed to load bundled extension; continuing without it.', error);
  }

  webView = new BrowserView({
    webPreferences: {
      contextIsolation: true,
      preload: path.join(__dirname, 'web-preload.js'),
    },
  });
  panelView = new BrowserView({
    webPreferences: {
      contextIsolation: true,
      preload: path.join(__dirname, 'panel-preload.js'),
    },
  });

  mainWindow.setBrowserView(webView);
  mainWindow.addBrowserView(panelView);
  createSplitLayout();
  mainWindow.on('resize', createSplitLayout);

  const startUrl = process.env.AMODELS_SHELL_START_URL || defaultHomeUrl;
  await webView.webContents.loadURL(startUrl);

  const panelEntry = path.join(__dirname, 'ui', 'dist', 'index.html');
  if (!fs.existsSync(panelEntry)) {
    throw new Error(
      'Panel UI build not found. Run "npm run build:ui" before starting the Electron shell.',
    );
  }
await panelView.webContents.loadFile(panelEntry);
emitNavigation(webView.webContents.getURL(), webView.webContents.getTitle(), latestPageState);

webView.webContents.on('did-navigate', (_, url) => {
  emitNavigation(url, webView.webContents.getTitle(), latestPageState);
});
webView.webContents.on('page-title-updated', (_, __, title) => {
  emitNavigation(webView.webContents.getURL(), title, latestPageState);
});
}

function emitNavigation(url, title, extra = {}) {
  if (!panelView || !panelView.webContents) return;
  panelView.webContents.send('shell:navigation', { url, title, ...extra });
}

ipcMain.on('shell:web-state', (_, payload = {}) => {
  latestPageState = {
    url: payload.url || '',
    title: payload.title || '',
    selection: payload.selection || '',
    text: payload.text || '',
  };
  emitNavigation(latestPageState.url, latestPageState.title, latestPageState);
});

ipcMain.handle('shell:action', async (_, { action, payload }) => {
  switch (action) {
    case 'navigate':
      if (!payload || !payload.url) {
        return { ok: false, error: 'Missing URL' };
      }
      try {
        await webView.webContents.loadURL(payload.url);
        return { ok: true };
      } catch (error) {
        return { ok: false, error: error.message };
      }
    case 'reload':
      try {
        await webView.webContents.reload();
        return { ok: true };
      } catch (error) {
        return { ok: false, error: error.message };
      }
    case 'back':
      try {
        if (webView.webContents.canGoBack()) {
          await webView.webContents.goBack();
        }
        return { ok: true };
      } catch (error) {
        return { ok: false, error: error.message };
      }
    case 'forward':
      try {
        if (webView.webContents.canGoForward()) {
          await webView.webContents.goForward();
        }
        return { ok: true };
      } catch (error) {
        return { ok: false, error: error.message };
      }
    case 'repo-file':
      try {
        if (!payload || !payload.path) {
          return { ok: false, error: 'Missing path' };
        }
        const targetPath = path.resolve(repoRootPath, payload.path);
        if (!targetPath.startsWith(repoRootPath)) {
          return { ok: false, error: 'Path outside repository' };
        }
        if (!fs.existsSync(targetPath)) {
          return { ok: false, error: 'File not found' };
        }
        return { ok: true, url: pathToFileURL(targetPath).href };
      } catch (error) {
        return { ok: false, error: error.message };
      }
    case 'default-home':
      return { ok: true, url: defaultHomeUrl };
    case 'highlight-text':
      try {
        if (!webView || !webView.webContents) {
          return { ok: false, error: 'Web view not ready' };
        }
        const phrases = Array.isArray(payload?.phrases) ? payload.phrases : [];
        webView.webContents.send('shell:highlight', { phrases });
        return { ok: true };
      } catch (error) {
        return { ok: false, error: error.message };
      }
    default:
      return { ok: false, error: `Unknown action ${action}` };
  }
});

app
  .whenReady()
  .then(createWindow)
  .catch((error) => {
    console.error('Failed to create browser window', error);
    app.quit();
  });

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow().catch((error) => console.error('Failed to recreate window', error));
  }
});
