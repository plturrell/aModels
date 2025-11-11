import { useCallback, useEffect, useMemo, useState } from 'react';
import { ResponsiveSankey } from '@nivo/sankey';
import { ResponsiveNetwork } from '@nivo/network';
import sgmiFlow from './data/sgmi_flow.json';

const DEFAULT_GATEWAY = 'http://localhost:8000';
const LEGACY_HOME = 'https://www.perplexity.ai/';
const GROUP_COLOR_PALETTE = [
  '#38bdf8',
  '#c084fc',
  '#f97316',
  '#34d399',
  '#f472b6',
  '#facc15',
  '#60a5fa',
  '#f97316',
  '#22d3ee',
];

const THEME_CLASS = {
  dark: 'theme-dark',
  light: 'theme-light',
};

const STREAM_SPLIT_REGEX = /(\s+)/;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function usePersistentState(key, defaultValue) {
  const [value, setValue] = useState(() => {
    try {
      const stored = window.localStorage.getItem(key);
      return stored ? JSON.parse(stored) : defaultValue;
    } catch (_) {
      return defaultValue;
    }
  });

  useEffect(() => {
    try {
      window.localStorage.setItem(key, JSON.stringify(value));
    } catch (_) {
      /* ignore quota errors */
    }
  }, [key, value]);

  return [value, setValue];
}

export default function App() {
  const [tabMeta, setTabMeta] = useState({ url: '', title: '' });
  const [pageContext, setPageContext] = useState({ url: '', title: '', selection: '', text: '' });
  const [gatewayUrl, setGatewayUrl] = usePersistentState('shellGateway', DEFAULT_GATEWAY);
  const [homeUrl, setHomeUrl] = usePersistentState('shellHomeUrl', '');
  const [defaultHomeUrl, setDefaultHomeUrl] = useState('');
  const [lastResult, setLastResult] = useState('');
  const [status, setStatus] = useState({ tone: 'info', message: 'Ready.' });
  const [logs, setLogs] = useState([]);
  const [chatModel, setChatModel] = usePersistentState('shellChatModel', 'auto');
  const [chatPrompt, setChatPrompt] = useState('');
  const [chatBusy, setChatBusy] = useState(false);
  const [summaryBusy, setSummaryBusy] = useState(false);
  const [summaryLines, setSummaryLines] = useState([]);
  const [summaryCitations, setSummaryCitations] = useState([]);
  const [followUps, setFollowUps] = useState([]);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [paletteQuery, setPaletteQuery] = useState('');
  const [paletteIndex, setPaletteIndex] = useState(0);
  const [modelMetrics, setModelMetrics] = useState(null);
  const [telemetryEntries, setTelemetryEntries] = useState([]);
  const [metricsBusy, setMetricsBusy] = useState(false);
  const [benchBusy, setBenchBusy] = useState(false);
  const [benchmarkHistory, setBenchmarkHistory] = useState([]);
  const [selectedJob, setSelectedJob] = useState(null);
  const [selectedTransition, setSelectedTransition] = useState(null);
  const [theme, setTheme] = usePersistentState('shellTheme', 'dark');

  useEffect(() => {
    const unsubscribe =
      window.shellBridge?.onNavigation?.((payload = {}) => {
        const nextMeta = {
          url: payload.url || '',
          title: payload.title || '',
        };
        setTabMeta(nextMeta);
        setPageContext({
          url: payload.url || '',
          title: payload.title || '',
          selection: payload.selection || '',
          text: payload.text || '',
        });
        setSummaryLines([]);
        setSummaryCitations([]);
        setFollowUps([]);
        setSelectedJob(null);
        setSelectedTransition(null);
        window.shellBridge?.invoke?.('highlight-text', { phrases: [] });
      }) ?? (() => {});
    return unsubscribe;
  }, []);

  useEffect(() => {
    (async () => {
      try {
        const homeRes = await window.shellBridge?.invoke?.('default-home');
        if (homeRes?.ok && homeRes.url) {
          setDefaultHomeUrl(homeRes.url);
          setHomeUrl((current) => {
            if (!current || current === '' || current === LEGACY_HOME) {
              return homeRes.url;
            }
            return current;
          });
        }
      } catch (_) {
        /* ignore */
      }
    })();
  }, [setHomeUrl]);

  useEffect(() => {
    if (!gatewayUrl) return;
    refreshTelemetry();
    const interval = setInterval(() => {
      refreshTelemetry();
    }, 30000);
    return () => clearInterval(interval);
  }, [gatewayUrl, refreshTelemetry]);

  useEffect(() => {
    const body = document.body;
    body.classList.remove('theme-dark', 'theme-light');
    body.classList.add(theme === 'light' ? 'theme-light' : 'theme-dark');
    let meta = document.querySelector('meta[name="color-scheme"]');
    if (!meta) {
      meta = document.createElement('meta');
      meta.setAttribute('name', 'color-scheme');
      document.head.appendChild(meta);
    }
    meta.setAttribute('content', theme === 'light' ? 'light dark' : 'dark light');
  }, [theme]);

  const pushLog = useCallback((line) => {
    setLogs((prev) => {
      const next = [...prev, `[${new Date().toLocaleTimeString()}] ${line}`];
      if (next.length > 80) next.shift();
      return next;
    });
  }, []);

  const callGateway = useCallback(
    async (path, options = {}) => {
      const {
        method = 'GET',
        body,
        label = 'Request',
        truncate = 400,
        headers = { 'Content-Type': 'application/json' },
      } = options;

      const base = (gatewayUrl || '').replace(/\/+$/, '');
      if (!base) {
        setStatus({ tone: 'warning', message: 'Set a gateway URL first.' });
        return null;
      }
      try {
        setStatus({ tone: 'info', message: `${label} in progress...` });
        pushLog(`${label} -> ${path}`);
        const response = await fetch(`${base}${path}`, {
          method,
          headers,
          body: body ? JSON.stringify(body) : undefined,
        });
        const text = await response.text();
        const payload = text ? safeParse(text) : null;
        if (!response.ok) {
          const reason = payload?.detail || payload?.message || response.statusText;
          throw new Error(reason || 'Gateway returned an error.');
        }
        const summary = payload ? truncateText(JSON.stringify(payload), truncate) : 'ok';
        setStatus({ tone: 'success', message: `${label} succeeded.` });
        setLastResult(summary);
        pushLog(`${label} OK`);
        return payload;
      } catch (error) {
        setStatus({ tone: 'error', message: `${label} failed: ${error.message}` });
        pushLog(`${label} FAIL ${error.message}`);
        setLastResult(error.stack || error.message);
        return null;
      }
    },
    [gatewayUrl, pushLog],
  );

  const handleNavigate = async () => {
    const target = prompt('Navigate browser to URL:', tabMeta.url || defaultHomeUrl || 'https://');
    if (!target) return;
    const trimmed = target.trim();
    const url = /^https?:\/\//i.test(trimmed) ? trimmed : `https://${trimmed}`;
    const result = await window.shellBridge?.invoke?.('navigate', { url });
    if (result?.ok) {
      setStatus({ tone: 'success', message: `Navigated to ${url}` });
    } else {
      setStatus({ tone: 'error', message: result?.error || 'Navigation failed.' });
    }
  };

  const handleReload = async () => {
    const result = await window.shellBridge?.invoke?.('reload');
    if (result?.ok) {
      setStatus({ tone: 'success', message: 'Browser reloaded.' });
    } else {
      setStatus({ tone: 'error', message: result?.error || 'Reload failed.' });
    }
  };

  const handleBack = async () => {
    const result = await window.shellBridge?.invoke?.('back');
    if (!result?.ok && result?.error) {
      setStatus({ tone: 'warning', message: result.error });
    }
  };

  const handleForward = async () => {
    const result = await window.shellBridge?.invoke?.('forward');
    if (!result?.ok && result?.error) {
      setStatus({ tone: 'warning', message: result.error });
    }
  };

  const handleOpenHome = async () => {
    const target = (homeUrl || '').trim() || defaultHomeUrl;
    if (!target) {
      setStatus({ tone: 'warning', message: 'Set a home URL first.' });
      return;
    }
    const result = await window.shellBridge?.invoke?.('navigate', { url: target });
    if (result?.ok) {
      setStatus({ tone: 'success', message: 'Opened home page.' });
    } else {
      setStatus({ tone: 'error', message: result?.error || 'Failed to open home page.' });
    }
  };

  const openHomePage = useCallback(async () => {
    if (!defaultHomeUrl) return;
    const result = await window.shellBridge?.invoke?.('navigate', { url: defaultHomeUrl });
    if (!result?.ok) {
      setStatus({ tone: 'error', message: result?.error || 'Navigation failed.' });
    }
  }, [defaultHomeUrl]);

  const openRepoPath = useCallback(
    async (relativePath) => {
      try {
        const resolved = await window.shellBridge?.invoke?.('repo-file', { path: relativePath });
        if (!resolved?.ok || !resolved.url) {
          throw new Error(resolved?.error || 'Unable to resolve path');
        }
        const nav = await window.shellBridge.invoke('navigate', { url: resolved.url });
        if (!nav?.ok) {
          throw new Error(nav?.error || 'Navigation failed');
        }
      } catch (error) {
        setStatus({ tone: 'error', message: error.message });
        pushLog(`Open ${relativePath} FAIL ${error.message}`);
      }
    },
    [pushLog],
  );

  const quickLinks = useMemo(() => {
    const entries = [];
    if (defaultHomeUrl) {
      entries.push({
        label: 'aModels Home',
        description: 'Curated workspace overview',
        onClick: openHomePage,
        group: 'Navigation',
      });
    }
    entries.push({
      label: 'Training Dashboard',
      description: 'Live training metrics UI',
      onClick: () => openRepoPath('web/training/dashboard.html'),
      group: 'Navigation',
    });
    entries.push({
      label: 'LocalAI Test UI',
      description: 'Local inference playground',
      onClick: () => openRepoPath('services/localai/web/index.html'),
      group: 'Navigation',
    });
    entries.push({
      label: 'Docs README',
      description: 'Reference documentation index',
      onClick: () => openRepoPath('docs/README.md'),
      group: 'Navigation',
    });
    entries.push({
      label: 'Gateway Compose',
      description: 'Inspect infrastructure/docker/compose.yml',
      onClick: () => openRepoPath('infrastructure/docker/compose.yml'),
      group: 'Navigation',
    });
    return entries;
  }, [defaultHomeUrl, openHomePage, openRepoPath]);

  const actions = useMemo(
    () => [
      {
        key: 'health',
        title: 'Gateway Health',
        description: 'Ping /healthz endpoint across the gateway.',
        handler: () => callGateway('/healthz', { label: 'Gateway health' }),
        group: 'Gateway',
      },
      {
        key: 'telemetry',
        title: 'Telemetry Recent',
        description: 'Retrieve recent telemetry payloads.',
        handler: () => callGateway('/telemetry/recent', { label: 'Telemetry recent', truncate: 800 }),
        group: 'Gateway',
      },
      {
        key: 'agentflow',
        title: 'Run AgentFlow',
        description: 'Trigger /agentflow/run with demo payload.',
        handler: () =>
          callGateway('/agentflow/run', {
            label: 'AgentFlow run',
            method: 'POST',
            body: { flow_id: 'demo', input: { q: 'hello agent' } },
            truncate: 640,
          }),
        group: 'Gateway',
      },
      {
        key: 'sql',
        title: 'SQL Sample',
        description: 'Execute SELECT 1 via /data/sql.',
        handler: () =>
          callGateway('/data/sql', {
            label: 'SQL demo',
            method: 'POST',
            body: { query: 'SELECT 1 as ok', args: [] },
          }),
        group: 'Gateway',
      },
      {
        key: 'sap-data-products',
        title: 'SAP Data Products',
        description: 'List SAP BDC data products.',
        handler: () =>
          callGateway('/v2/sap-bdc/data-products', {
            label: 'SAP data products',
            truncate: 800,
          }),
        group: 'SAP',
      },
      {
        key: 'sap-formation',
        title: 'SAP Formation',
        description: 'Get SAP BDC formation details.',
        handler: () =>
          callGateway('/sap-bdc/formation', {
            label: 'SAP formation',
            truncate: 800,
          }),
        group: 'SAP',
      },
      {
        key: 'sap-intelligent-apps',
        title: 'SAP Intelligent Applications',
        description: 'List SAP intelligent applications.',
        handler: () =>
          callGateway('/sap-bdc/intelligent-applications', {
            label: 'SAP intelligent applications',
            truncate: 800,
          }),
        group: 'SAP',
      },
      {
        key: 'redis',
        title: 'Redis Roundtrip',
        description: 'Set/get demo key via gateway cache endpoints.',
        handler: async () => {
          await callGateway('/redis/set', {
            label: 'Redis set',
            method: 'POST',
            body: { key: 'demo', value: '42', ex: 45 },
          });
          await callGateway('/redis/get?key=demo', { label: 'Redis get' });
        },
        group: 'Gateway',
      },
    ],
    [callGateway],
  );

  const handleSummarise = useCallback(async () => {
    const base = (gatewayUrl || '').replace(/\/+$/, '');
    if (!base) {
      setStatus({ tone: 'warning', message: 'Set a gateway URL first.' });
      return;
    }
    const contextBlock = (pageContext.selection || pageContext.text || '').trim();
    if (!contextBlock) {
      setStatus({ tone: 'warning', message: 'No page content available to summarise.' });
      return;
    }
    setSummaryBusy(true);
    setStatus({ tone: 'info', message: 'Summarising current page...' });
    pushLog('Page summary -> /localai/chat');
    try {
      const trimmedContext = contextBlock.slice(0, 6000);
      const response = await fetch(`${base}/localai/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: chatModel || 'auto',
          messages: [
            {
              role: 'system',
              content:
                'Return JSON {"summary":[],"citations":[],"followups":[]} with no extra prose. Citations should include snippet fields.',
            },
            {
              role: 'user',
              content: `Title: ${pageContext.title}\nURL: ${pageContext.url}\nSelection: ${pageContext.selection}\nContext:\n${trimmedContext}`,
            },
          ],
          max_tokens: 512,
          temperature: 0.2,
        }),
      });
      const raw = await response.text();
      const parsed = safeParse(raw);
      if (!response.ok) {
        const reason = parsed?.detail || parsed?.message || response.statusText;
        throw new Error(reason || 'Gateway returned an error.');
      }
      let summary = [];
      let citations = [];
      let followups = [];
      let candidate = parsed;
      if (parsed?.choices) {
        const content = parsed.choices?.[0]?.message?.content?.trim();
        candidate = safeParse(content) || content;
      }
      if (candidate && typeof candidate === 'string') {
        summary = candidate.split(/\n+/).map((line) => line.trim()).filter(Boolean);
      } else if (candidate && typeof candidate === 'object') {
        if (Array.isArray(candidate.summary)) {
          summary = candidate.summary.filter((line) => typeof line === 'string' && line.trim());
        }
        if (Array.isArray(candidate.citations)) {
          citations = candidate.citations
            .map((item) =>
              typeof item === 'string'
                ? { snippet: item }
                : { snippet: item.snippet || item.text || '' },
            )
            .filter((item) => item.snippet);
        }
        if (Array.isArray(candidate.followups)) {
          followups = candidate.followups.filter((item) => typeof item === 'string' && item.trim());
        }
      }
      if (!summary.length && parsed?.choices) {
        const fallback = parsed.choices?.[0]?.message?.content?.trim();
        if (fallback) {
          summary = fallback.split(/\n+/).map((line) => line.trim()).filter(Boolean);
        }
      }
      if (!summary.length) {
        summary = ['No summary returned.'];
      }
      if (!citations.length) {
        citations = deriveCitations(pageContext);
      }
      setSummaryLines(summary);
      setSummaryCitations(citations);
      setFollowUps(followups);
      const phrases = citations.map((item) => item.snippet).filter(Boolean).slice(0, 6);
      if (phrases.length) {
        window.shellBridge?.invoke?.('highlight-text', { phrases });
      } else {
        window.shellBridge?.invoke?.('highlight-text', { phrases: [] });
      }
      setStatus({ tone: 'success', message: 'Page summary ready.' });
      pushLog('Page summary OK');
    } catch (error) {
      setStatus({ tone: 'error', message: `Page summary failed: ${error.message}` });
      pushLog(`Page summary FAIL ${error.message}`);
      setSummaryLines([]);
      setSummaryCitations([]);
      setFollowUps([]);
      window.shellBridge?.invoke?.('highlight-text', { phrases: [] });
    } finally {
      setSummaryBusy(false);
    }
  }, [gatewayUrl, pageContext, chatModel, pushLog]);

  const refreshTelemetry = useCallback(async () => {
    const base = (gatewayUrl || '').replace(/\/+$/, '');
    if (!base) {
      setStatus({ tone: 'warning', message: 'Set a gateway URL first.' });
      return;
    }
    setMetricsBusy(true);
    pushLog('Metrics refresh -> /api/v1/models');
    try {
      const modelsResponse = await fetch(`${base}/api/v1/models`);
      const modelsPayload = await modelsResponse.json();
      if (!modelsResponse.ok) {
        const reason = modelsPayload?.detail || modelsPayload?.message || modelsResponse.statusText;
        throw new Error(reason || 'Unable to load models');
      }
      setModelMetrics(computeModelMetrics(modelsPayload));

      let telemetryPayload = [];
      try {
        const telemetryResponse = await fetch(`${base}/telemetry/recent`);
        if (telemetryResponse.ok) {
          telemetryPayload = await telemetryResponse.json();
        }
      } catch (innerError) {
        pushLog(`Telemetry fetch FAIL ${innerError.message}`);
      }
      setTelemetryEntries(normaliseTelemetryEntries(telemetryPayload));
      setBenchmarkHistory((prev) => (prev.length ? prev : []));
      setStatus({ tone: 'success', message: 'Metrics refreshed.' });
      pushLog('Metrics refresh OK');
    } catch (error) {
      setStatus({ tone: 'error', message: `Metrics refresh failed: ${error.message}` });
      pushLog(`Metrics refresh FAIL ${error.message}`);
    } finally {
      setMetricsBusy(false);
    }
  }, [gatewayUrl, pushLog]);

  const triggerBenchmark = useCallback(async () => {
    const base = (gatewayUrl || '').replace(/\/+$/, '');
    if (!base) {
      setStatus({ tone: 'warning', message: 'Set a gateway URL first.' });
      return;
    }
    setBenchBusy(true);
    setStatus({ tone: 'info', message: 'Running benchmark (MCQ)...' });
    pushLog('Benchmark -> /api/v1/benchmark');
    try {
      const response = await fetch(`${base}/api/v1/benchmark`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          inference_request: {
            task_type: 'mcq',
            question: 'Which service provides LocalAI inference?',
            choices: [
              'Agentflow service',
              'LocalAI service',
              'Training service',
              'Analytics service',
            ],
          },
          correct_answer: 'LocalAI service',
        }),
      });
      const payload = await response.json();
      if (!response.ok) {
        const reason = payload?.detail || payload?.message || response.statusText;
        throw new Error(reason || 'Benchmark failed');
      }
      const entry = {
        timestamp: new Date().toLocaleTimeString(),
        latency: payload.latency_ms || payload.latency || 0,
        confidence: payload.confidence ?? null,
        correct: Boolean(payload.is_correct),
        model: payload.model_name || 'unknown',
      };
      setBenchmarkHistory((prev) => [entry, ...prev].slice(0, 20));
      setStatus({
        tone: 'success',
        message: `Benchmark ${entry.correct ? 'passed' : 'failed'} (${entry.latency.toFixed(
          0,
        )}ms)`,
      });
      pushLog('Benchmark OK');
      await refreshTelemetry();
    } catch (error) {
      setStatus({ tone: 'error', message: `Benchmark failed: ${error.message}` });
      pushLog(`Benchmark FAIL ${error.message}`);
    } finally {
      setBenchBusy(false);
    }
  }, [gatewayUrl, refreshTelemetry, pushLog]);

  const paletteEntries = useMemo(() => {
    const entries = [];
    quickLinks.forEach((item) => {
      entries.push({
        label: item.label,
        description: item.description,
        group: item.group || 'Navigation',
        onSelect: () => {
          item.onClick();
          setPaletteOpen(false);
        },
      });
    });

    actions.forEach((action) => {
      entries.push({
        label: action.title,
        description: action.description,
        group: action.group || 'Gateway',
        onSelect: () => {
          action.handler();
          setPaletteOpen(false);
        },
      });
    });

    entries.push({
      label: 'Summarise current page',
      description: 'Generate a LocalAI summary with citations',
      group: 'AI Assistant',
      onSelect: () => {
        handleSummarise();
        setPaletteOpen(false);
      },
    });

    entries.push({
      label: 'Refresh metrics',
      description: 'Pull /api/v1/models and telemetry/recent',
      group: 'Telemetry',
      onSelect: () => {
        refreshTelemetry();
        setPaletteOpen(false);
      },
    });

    entries.push({
      label: 'Run benchmark (MCQ)',
      description: 'POST /api/v1/benchmark with sample MCQ workload',
      group: 'Telemetry',
      onSelect: () => {
        triggerBenchmark();
        setPaletteOpen(false);
      },
    });

    return entries;
  }, [quickLinks, actions, handleSummarise, openHomePage, refreshTelemetry, triggerBenchmark, openRepoPath]);

  const filteredPalette = useMemo(() => {
    if (!paletteQuery.trim()) {
      return paletteEntries;
    }
    const needle = paletteQuery.trim().toLowerCase();
    return paletteEntries.filter((entry) =>
      [entry.label, entry.description, entry.group]
        .filter(Boolean)
        .some((value) => value.toLowerCase().includes(needle)),
    );
  }, [paletteEntries, paletteQuery]);

const handlePaletteSelect = useCallback(
  (entry) => {
    if (!entry) return;
    entry.onSelect?.();
  },
  [],
);

const streamDelay = (chunk) => Math.min(80, 10 + chunk.length * 4);

const streamText = useCallback(
  async (text) => {
    const content = (text || '').toString();
    if (!content) {
      setLastResult('');
      return;
    }
    const pieces = content.split(STREAM_SPLIT_REGEX).filter((piece) => piece.length);
    let accumulator = '';
    setLastResult('');
    for (const piece of pieces) {
      accumulator += piece;
      setLastResult(accumulator);
      // eslint-disable-next-line no-await-in-loop
      await sleep(streamDelay(piece));
    }
  },
  [],
);


  useEffect(() => {
    function onKeyDown(event) {
      const key = event.key.toLowerCase();
      if ((event.metaKey || event.ctrlKey) && key === 'k') {
        event.preventDefault();
        setPaletteOpen((open) => !open);
        setPaletteQuery('');
        setPaletteIndex(0);
        return;
      }
      if (!paletteOpen) return;
      if (key === 'escape') {
        event.preventDefault();
        setPaletteOpen(false);
        return;
      }
      if (key === 'arrowdown') {
        event.preventDefault();
        setPaletteIndex((idx) => Math.min(idx + 1, Math.max(filteredPalette.length - 1, 0)));
      } else if (key === 'arrowup') {
        event.preventDefault();
        setPaletteIndex((idx) => Math.max(idx - 1, 0));
      } else if (key === 'enter') {
        event.preventDefault();
        const entry = filteredPalette[paletteIndex];
        if (entry) {
          handlePaletteSelect(entry);
        }
      }
    }

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [paletteOpen, filteredPalette, paletteIndex, handlePaletteSelect]);

  useEffect(() => {
    if (!paletteOpen) {
      setPaletteQuery('');
      setPaletteIndex(0);
    } else {
      setPaletteIndex(0);
    }
  }, [paletteOpen]);

useEffect(() => {
  setPaletteIndex((idx) => Math.min(idx, Math.max(filteredPalette.length - 1, 0)));
}, [filteredPalette]);

const benchmarkSpark = useMemo(() => {
    if (!benchmarkHistory.length) return null;
    const latencies = benchmarkHistory.map((entry) => entry.latency || 0);
    const maxLatency = Math.max(...latencies, 1);
    const minLatency = Math.min(...latencies, 0);
    const height = 60;
    const width = 200;
    const points = benchmarkHistory.map((entry, index) => {
      const x = (index / Math.max(benchmarkHistory.length - 1, 1)) * width;
      const normalised = maxLatency === minLatency ? 0.5 : (entry.latency - minLatency) / (maxLatency - minLatency);
      const y = height - normalised * height;
      return `${x},${y}`;
    });
    return {
      points: points.reverse().join(' '),
      maxLatency,
      minLatency,
    };
  }, [benchmarkHistory]);

const sankeyData = useMemo(() => {
    const dataset = sgmiFlow?.sankey;
    if (!dataset) return null;
    const topLinks = dataset.links.slice(0, 40);
    const nodeIds = new Set();
    topLinks.forEach((link) => {
      nodeIds.add(link.source);
      nodeIds.add(link.target);
    });
    const nodes = dataset.nodes.filter((node) => nodeIds.has(node.id));
    const validIds = new Set(nodes.map((node) => node.id));
    const links = topLinks.filter(
      (link) => validIds.has(link.source) && validIds.has(link.target) && link.value > 0,
    );
    return { nodes, links };
  }, []);

const networkData = useMemo(() => {
  const dataset = sgmiFlow?.network;
  if (!dataset) return null;
  const topLinks = dataset.links.slice(0, 80);
  const degree = {};
  topLinks.forEach((link) => {
    degree[link.source] = (degree[link.source] || 0) + link.value;
    degree[link.target] = (degree[link.target] || 0) + link.value;
  });
  const nodeMap = new Map(dataset.nodes.map((node) => [node.id, node]));
  const nodeIds = Array.from(new Set(topLinks.flatMap((link) => [link.source, link.target])));
  const nodes = nodeIds.map((id) => {
    const original = nodeMap.get(id) || {};
    const jobMeta = original.meta || {};
    const group = jobMeta.application || original.group || 'unknown';
    const deg = degree[id] || 1;
    return {
      id,
      radius: Math.min(18, 6 + Math.log2(deg + 1) * 4),
      color: colorForGroup(group),
      group,
      type: jobMeta.type || original.type,
      meta: jobMeta,
      data: { meta: jobMeta },
    };
  });
  const links = topLinks.map((link) => ({
    source: link.source,
    target: link.target,
      value: Math.max(1, link.value),
      distance: 80,
    }));
    return { nodes, links };
  }, []);

  const jobMetaMap = useMemo(() => {
    const map = new Map();
    (sgmiFlow?.sankey?.nodes || []).forEach((node) => {
      map.set(node.id, node.meta || {});
    });
    return map;
  }, []);

  const topTransitions = useMemo(() => {
    if (!sankeyData) return [];
    return sankeyData.links.slice(0, 12);
  }, [sankeyData]);

  const handleSankeyEntityClick = useCallback(
    (entity) => {
      if (!entity) return;
      if (entity.source && entity.target) {
        const sourceId = entity.source.id || entity.source;
        const targetId = entity.target.id || entity.target;
        setSelectedTransition({
          source: sourceId,
          target: targetId,
          value: entity.value,
          sourceMeta: entity.source.meta || jobMetaMap.get(sourceId) || {},
          targetMeta: entity.target.meta || jobMetaMap.get(targetId) || {},
        });
        setSelectedJob(null);
      } else if (entity.id) {
        setSelectedJob({
          id: entity.id,
          meta: entity.meta || jobMetaMap.get(entity.id) || {},
        });
        setSelectedTransition(null);
      }
    },
    [jobMetaMap],
  );

  const handleNetworkNodeClick = useCallback(
    (node) => {
      if (!node) return;
      setSelectedJob({
        id: node.id,
        meta: node.meta || node.data?.meta || jobMetaMap.get(node.id) || {},
      });
      setSelectedTransition(null);
    },
    [jobMetaMap],
  );

  const selectTransition = useCallback(
    (link) => {
      if (!link) return;
      setSelectedTransition({
        source: link.source,
        target: link.target,
        value: link.value,
        sourceMeta: jobMetaMap.get(link.source) || {},
        targetMeta: jobMetaMap.get(link.target) || {},
      });
      setSelectedJob(null);
    },
    [jobMetaMap],
  );

  const clearSelection = useCallback(() => {
    setSelectedJob(null);
    setSelectedTransition(null);
  }, []);

  const handleChat = async () => {
    const prompt = chatPrompt.trim();
    if (!prompt) {
      setStatus({ tone: 'warning', message: 'Enter a prompt before sending.' });
      return;
    }
    setChatBusy(true);
    setFollowUps([]);
    try {
      const payload = await callGateway('/localai/chat', {
        label: 'LocalAI chat',
        method: 'POST',
        body: {
          model: chatModel || 'auto',
          messages: [{ role: 'user', content: prompt }],
        },
        truncate: 1200,
      });
      const message =
        payload?.choices?.[0]?.message?.content ||
        payload?.message ||
        JSON.stringify(payload ?? {}, null, 2);
      await streamText(message || '');
      setChatPrompt('');
    } finally {
      setChatBusy(false);
    }
  };

  return (
    <div className={`shell-panel ${THEME_CLASS[theme] || THEME_CLASS.dark}`}>
      {paletteOpen && (
        <div className="palette-backdrop" role="dialog" aria-modal="true">
          <div className="palette-panel">
            <input
              autoFocus
              className="palette-input"
              placeholder="Search commands, gateway actions, pages..."
              value={paletteQuery}
              onChange={(event) => {
                setPaletteQuery(event.target.value);
                setPaletteIndex(0);
              }}
            />
            <div className="palette-shortcut">Press Esc to close • ↑/↓ to navigate</div>
            <div className="palette-results">
              {filteredPalette.length === 0 ? (
                <div className="palette-empty">No matches.</div>
              ) : (
                filteredPalette.map((entry, index) => (
                  <button
                    key={`${entry.group}-${entry.label}-${index}`}
                    type="button"
                    className={`palette-item ${index === paletteIndex ? 'active' : ''}`}
                    onMouseEnter={() => setPaletteIndex(index)}
                    onClick={() => handlePaletteSelect(entry)}
                  >
                    <div className="palette-item-header">
                      <span className="palette-item-label">{entry.label}</span>
                      <span className="palette-item-group">{entry.group}</span>
                    </div>
                    {entry.description && (
                      <div className="palette-item-desc">{entry.description}</div>
                    )}
                  </button>
                ))
              )}
            </div>
          </div>
        </div>
      )}

      <header className="panel-header">
        <div>
          <h1>aModels Browser</h1>
          <div style={{ fontSize: 12, color: '#94a3b8' }}>Unified assistant + telemetry control</div>
        </div>
        <div className="toolbar">
          <button
            type="button"
            className="toolbar-button"
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          >
            {theme === 'dark' ? 'Light Mode' : 'Dark Mode'}
          </button>
          <button type="button" onClick={handleBack}>
            Back
          </button>
          <button type="button" onClick={handleForward}>
            Forward
          </button>
          <button type="button" onClick={handleReload}>
            Reload
          </button>
          <button type="button" onClick={handleNavigate}>
            Navigate...
          </button>
        </div>
      </header>

      <section className="panel-section">
        <div className="metrics-header">
          <h2>Telemetry</h2>
          <button
            type="button"
            className="toolbar-button"
            onClick={refreshTelemetry}
            disabled={metricsBusy}
          >
            {metricsBusy ? 'Refreshing...' : 'Refresh'}
          </button>
        </div>
        <div className="telemetry-grid">
          <div className="telemetry-card">
            <span className="metric-label">Models</span>
            <span className="metric-value">{modelMetrics?.modelCount ?? '--'}</span>
            <span className="metric-sub">{modelMetrics ? `${modelMetrics.domainCount} domains` : 'n/a'}</span>
          </div>
          <div className="telemetry-card">
            <span className="metric-label">Avg Accuracy</span>
            <span className="metric-value">
              {modelMetrics?.avgAccuracy != null
                ? `${(modelMetrics.avgAccuracy * 100).toFixed(1)}%`
                : '--'}
            </span>
            <span className="metric-sub">Based on gateway /api/v1/models</span>
          </div>
          <div className="telemetry-card">
            <span className="metric-label">Avg Latency</span>
            <span className="metric-value">
              {modelMetrics?.avgLatency != null ? `${modelMetrics.avgLatency.toFixed(0)}ms` : '--'}
            </span>
            <span className="metric-sub">Mean latency for listed models</span>
          </div>
          <div className="telemetry-card">
            <span className="metric-label">Avg Throughput</span>
            <span className="metric-value">
              {modelMetrics?.avgThroughput != null
                ? `${modelMetrics.avgThroughput.toFixed(1)}/s`
                : '--'}
            </span>
            <span className="metric-sub">Tokens per second equivalent</span>
          </div>
        </div>
        <div className="benchmark-block">
          <div className="benchmark-header">
            <div>
              <h3>Benchmark (MCQ)</h3>
              <span className="benchmark-sub">Tracks latency and confidence across runs</span>
            </div>
            <button
              type="button"
              className="toolbar-button"
              disabled={benchBusy}
              onClick={triggerBenchmark}
            >
              {benchBusy ? 'Running...' : 'Run Benchmark'}
            </button>
          </div>
          {benchmarkHistory.length === 0 ? (
            <div className="telemetry-empty">No benchmark runs yet.</div>
          ) : (
            <div className="benchmark-body">
              {benchmarkSpark && (
                <svg
                  className="benchmark-spark"
                  viewBox="0 0 200 60"
                  preserveAspectRatio="none"
                >
                  <polyline points={benchmarkSpark.points} />
                </svg>
              )}
              <div className="benchmark-stats">
                <div>
                  <span className="metric-label">Latest Latency</span>
                  <span className="metric-value">
                    {benchmarkHistory[0]?.latency != null
                      ? `${benchmarkHistory[0].latency.toFixed(0)}ms`
                      : '--'}
                  </span>
                </div>
                <div>
                  <span className="metric-label">Latest Confidence</span>
                  <span className="metric-value">
                    {benchmarkHistory[0]?.confidence != null
                      ? `${(benchmarkHistory[0].confidence * 100).toFixed(1)}%`
                      : '--'}
                  </span>
                </div>
                <div>
                  <span className="metric-label">Pass Rate</span>
                  <span className="metric-value">
                    {computePassRate(benchmarkHistory)}%
                  </span>
                </div>
              </div>
              <div className="benchmark-history">
                {benchmarkHistory.slice(0, 5).map((entry, index) => (
                  <div key={index} className="telemetry-item">
                    <div>
                      <div className="telemetry-item-title">
                        {entry.model} • {entry.correct ? 'PASS' : 'FAIL'}
                      </div>
                      <div className="telemetry-item-details">
                        {entry.latency.toFixed(0)}ms •
                        {entry.confidence != null
                          ? ` ${(entry.confidence * 100).toFixed(1)}%`
                          : ''}
                      </div>
                    </div>
                    <div className="telemetry-item-meta">{entry.timestamp}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
        <div className="telemetry-list">
          <h3>Recent Telemetry</h3>
          {telemetryEntries.length === 0 ? (
            <div className="telemetry-empty">No telemetry entries returned from gateway.</div>
          ) : (
            telemetryEntries.map((entry, index) => (
              <div key={index} className="telemetry-item">
                <div>
                  <div className="telemetry-item-title">{entry.title}</div>
                  {entry.details && <div className="telemetry-item-details">{entry.details}</div>}
                </div>
                <div className="telemetry-item-meta">{entry.timestamp}</div>
              </div>
            ))
          )}
        </div>
      </section>

      <section className="panel-section">
        <h2>Browser Control</h2>
        <div style={{ display: 'flex', gap: 10, marginBottom: 12 }}>
          <input
            type="text"
            value={homeUrl}
            onChange={(event) => setHomeUrl(event.target.value)}
            placeholder="https://example.com or file:///..."
            style={{
              flex: 1,
              padding: '10px 14px',
              borderRadius: 10,
              border: '1px solid rgba(148,163,184,0.2)',
              background: 'rgba(15,23,42,0.6)',
              color: 'inherit',
            }}
          />
          <button type="button" className="primary-button" onClick={handleOpenHome}>
            Open Home
          </button>
        </div>
        <div className="actions-grid">
          {quickLinks.map((item) => (
            <button
              key={item.label}
              type="button"
              className="action-card"
              onClick={item.onClick}
            >
              <strong>{item.label}</strong>
              <span style={{ fontSize: 12, color: '#94a3b8' }}>{item.description}</span>
            </button>
          ))}
        </div>
      </section>

      {sankeyData && networkData && (
        <section className="panel-section">
          <div className="graph-header">
            <h2>SGMI Extract Flow</h2>
            <span className="graph-sub">Derived from data/training/sgmi/json_with_changes.json</span>
          </div>
          <div className="graph-grid">
            <div className="graph-card">
              <div className="graph-card-header">
                <h3>Job Transition Sankey</h3>
                <span className="graph-meta">Top 40 transitions</span>
              </div>
              <div className="graph-canvas">
                <ResponsiveSankey
                  data={sankeyData}
                  margin={{ top: 12, right: 12, bottom: 12, left: 12 }}
                  align="justify"
                  colors={{ scheme: 'category10' }}
                  nodeOpacity={1}
                  nodeThickness={12}
                  nodeSpacing={12}
                  nodeBorderWidth={1}
                  nodeBorderColor={{ from: 'color', modifiers: [['darker', 0.6]] }}
                  linkOpacity={0.4}
                  linkBlendMode="multiply"
                  enableLinkGradient
                  labelPadding={12}
                  labelOrientation="vertical"
                  valueFormat=">-.0f"
                  onClick={handleSankeyEntityClick}
                  nodeTooltip={({ node }) => (
                    <div className="graph-tooltip">
                      <strong>{node.id}</strong>
                      <div>{node.meta?.application || 'No application'}</div>
                    </div>
                  )}
                  linkTooltip={({ link }) => (
                    <div className="graph-tooltip">
                      <strong>
                        {link.source.id} → {link.target.id}
                      </strong>
                      <div>{link.value} transitions</div>
                    </div>
                  )}
                />
              </div>
            </div>
            <div className="graph-card">
              <div className="graph-card-header">
                <h3>Job Network</h3>
                <span className="graph-meta">Clustered by application</span>
              </div>
              <div className="graph-canvas">
                <ResponsiveNetwork
                  data={networkData}
                  margin={{ top: 12, right: 12, bottom: 12, left: 12 }}
                  nodeColor={(node) => node.color}
                  nodeBorderWidth={1}
                  nodeBorderColor={{ from: 'color', modifiers: [['darker', 0.6]] }}
                  nodeSize={(node) => node.radius}
                  linkThickness={(link) => Math.max(1.5, Math.log2(link.value + 1) * 1.5)}
                  linkColor={{ from: 'source', modifiers: [['darker', 0.2]] }}
                  distanceMin={25}
                  distanceMax={160}
                  repulsivity={36}
                  iterations={70}
                  linkDistance={(link) => link.distance || 80}
                  onClick={(node) => handleNetworkNodeClick(node)}
                  nodeTooltip={({ node }) => (
                    <div className="graph-tooltip">
                      <strong>{node.id}</strong>
                      <div>{node.meta?.application || 'No application'}</div>
                    </div>
                  )}
                />
              </div>
            </div>
          </div>
          <div className="graph-detail-block">
            <div className="graph-detail-card">
              <div className="graph-card-header">
                <h3>Selection Details</h3>
                {(selectedJob || selectedTransition) && (
                  <button type="button" className="toolbar-button" onClick={clearSelection}>
                    Clear
                  </button>
                )}
              </div>
              {selectedJob ? (
                <JobDetail job={selectedJob} />
              ) : selectedTransition ? (
                <TransitionDetail transition={selectedTransition} />
              ) : (
                <div className="telemetry-empty">
                  Click a node or transition to inspect metadata.
                </div>
              )}
            </div>
            <div className="graph-detail-card">
              <div className="graph-card-header">
                <h3>Top Transitions</h3>
                <span className="graph-meta">Click a row to inspect the transition</span>
              </div>
              {topTransitions.length === 0 ? (
                <div className="telemetry-empty">No transitions found in dataset.</div>
              ) : (
                <table className="graph-table">
                  <thead>
                    <tr>
                      <th>Source</th>
                      <th>Target</th>
                      <th>Count</th>
                      <th>Applications</th>
                    </tr>
                  </thead>
                  <tbody>
                    {topTransitions.map((link) => {
                      const sourceMeta = jobMetaMap.get(link.source) || {};
                      const targetMeta = jobMetaMap.get(link.target) || {};
                      const isActive =
                        selectedTransition &&
                        selectedTransition.source === link.source &&
                        selectedTransition.target === link.target;
                      return (
                        <tr
                          key={`${link.source}-${link.target}`}
                          className={isActive ? 'active' : ''}
                          onClick={() => selectTransition(link)}
                        >
                          <td>{link.source}</td>
                          <td>{link.target}</td>
                          <td>{link.value}</td>
                          <td>
                            {(sourceMeta.application || '—')}
                            {' → '}
                            {targetMeta.application || '—'}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        </section>
      )}

      <section className="panel-section">
        <h2>Active Tab</h2>
        <div className="tab-meta">
          <strong>{tabMeta.title || 'Untitled tab'}</strong>
          <span style={{ wordBreak: 'break-all', color: '#94a3b8' }}>{tabMeta.url || '--'}</span>
        </div>
      </section>

      <section className="panel-section">
        <h2>Gateway</h2>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 12 }}>
          <input
            type="text"
            value={gatewayUrl}
            onChange={(event) => setGatewayUrl(event.target.value)}
            placeholder="http://localhost:8000"
            style={{
              flex: 1,
              padding: '10px 14px',
              borderRadius: 10,
              border: '1px solid rgba(148,163,184,0.2)',
              background: 'rgba(15,23,42,0.6)',
              color: 'inherit',
            }}
          />
          <div className="status-pill" data-state={status.tone}>
            <span
              style={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                background:
                  status.tone === 'success'
                    ? '#34d399'
                    : status.tone === 'error'
                    ? '#f87171'
                    : '#a5b4fc',
              }}
            />
            {status.message}
          </div>
        </div>
        <div className="actions-grid">
          {actions.map((action) => (
            <button
              key={action.key}
              type="button"
              className="action-card"
              onClick={action.handler}
            >
              <strong>{action.title}</strong>
              <span style={{ fontSize: 12, color: '#94a3b8' }}>{action.description}</span>
            </button>
          ))}
        </div>
      </section>

      <section className="panel-section">
        <h2>LocalAI Chat</h2>
        <div className="chat-area">
          <textarea
            value={chatPrompt}
            onChange={(event) => setChatPrompt(event.target.value)}
            placeholder="Ask the assistant anything about the current page..."
          />
          <div className="chat-controls">
            <input
              type="text"
              value={chatModel}
              onChange={(event) => setChatModel(event.target.value)}
              placeholder="Model (auto, gemma, etc.)"
            />
            <button
              type="button"
              className="primary-button"
              disabled={chatBusy}
              onClick={handleChat}
            >
              {chatBusy ? 'Sending...' : 'Send'}
            </button>
          </div>
        </div>
      </section>

      <section className="panel-section">
        <h2>Page Summary</h2>
        <p style={{ color: '#94a3b8', fontSize: 13, marginTop: 0 }}>
          Generate a retrieval-augmented overview of the current page with lightweight citations.
        </p>
        <div className="chat-controls" style={{ marginTop: 12 }}>
          <button
            type="button"
            className="primary-button"
            style={{ flex: 'unset' }}
            disabled={summaryBusy}
            onClick={handleSummarise}
          >
            {summaryBusy ? 'Summarising...' : 'Summarise Page'}
          </button>
        </div>
        {summaryLines.length ? (
          <div style={{ marginTop: 16 }}>
            <div className="result-block" style={{ maxHeight: 220 }}>
              <ul className="summary-list">
                {summaryLines.map((line, idx) => (
                  <li key={idx}>{line}</li>
                ))}
              </ul>
            </div>
            {summaryCitations.length > 0 && (
              <div style={{ marginTop: 12, display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                {summaryCitations.map((citation, idx) => (
                  <div
                    key={idx}
                    style={{
                      background: 'rgba(59, 130, 246, 0.14)',
                      border: '1px solid rgba(96, 165, 250, 0.4)',
                      borderRadius: 12,
                      padding: '10px 12px',
                      maxWidth: '320px',
                      fontSize: 12,
                    }}
                  >
                    <strong style={{ display: 'block', marginBottom: 4 }}>[{idx + 1}]</strong>
                    <span>{citation.snippet}</span>
                  </div>
                ))}
              </div>
            )}
            {followUps.length > 0 && (
              <div className="followups">
                <div className="followups-title">Follow-up suggestions</div>
                <div className="followups-list">
                  {followUps.map((item, idx) => (
                    <button
                      key={idx}
                      type="button"
                      className="followup-chip"
                      onClick={() => setChatPrompt(item)}
                    >
                      {item}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : summaryBusy ? null : (
          <div style={{ color: '#64748b', fontSize: 13, marginTop: 12 }}>
            No summary generated yet.
          </div>
        )}
      </section>

      <section className="panel-section">
        <h2>Result</h2>
        <div className="result-block">{lastResult || 'No response yet.'}</div>
      </section>

      <section className="panel-section" style={{ flex: 1 }}>
        <h2>Event Log</h2>
        <div className="log-area">
          {logs.length === 0 ? (
            <span style={{ color: '#64748b' }}>Awaiting interactions...</span>
          ) : (
            logs.map((line, index) => <div key={index}>{line}</div>)
          )}
        </div>
      </section>
    </div>
  );
}

function truncateText(str, limit) {
  if (!str || str.length <= limit) return str;
  return `${str.slice(0, limit)}...`;
}

function safeParse(text) {
  try {
    return JSON.parse(text);
  } catch (_) {
    return { message: text };
  }
}

function deriveCitations(context, limit = 3) {
  const snippets = [];
  const seen = new Set();

  if (context.selection) {
    const trimmed = context.selection.trim();
    if (trimmed) {
      snippets.push(trimmed);
      seen.add(trimmed);
    }
  }

  const paragraphs = (context.text || '')
    .split(/\n{2,}/)
    .map((paragraph) => paragraph.trim())
    .filter((paragraph) => paragraph.length > 60);

  for (const paragraph of paragraphs) {
    if (snippets.length >= limit) break;
    if (seen.has(paragraph)) continue;
    snippets.push(paragraph);
    seen.add(paragraph);
  }

  return snippets.slice(0, limit).map((snippet) => ({
    snippet: snippet.length > 220 ? `${snippet.slice(0, 220)}...` : snippet,
    url: context.url,
  }));
}

function computeModelMetrics(payload) {
  if (!payload) return null;
  const models = Array.isArray(payload) ? payload : Object.values(payload);
  if (!models.length) {
    return { modelCount: 0, domainCount: 0, avgAccuracy: null, avgLatency: null, avgThroughput: null };
  }

  let count = 0;
  let acc = 0;
  let latency = 0;
  let throughput = 0;
  const domains = new Set();

  models.forEach((model) => {
    if (!model) return;
    count += 1;
    if (model.performance) {
      if (typeof model.performance.accuracy === 'number') acc += model.performance.accuracy;
      if (typeof model.performance.latency_ms === 'number') latency += model.performance.latency_ms;
      if (typeof model.performance.throughput_tps === 'number') {
        throughput += model.performance.throughput_tps;
      }
    }
    if (model.domain) {
      domains.add(model.domain);
    }
  });

  return {
    modelCount: count,
    domainCount: domains.size,
    avgAccuracy: count ? acc / count : null,
    avgLatency: count ? latency / count : null,
    avgThroughput: count ? throughput / count : null,
  };
}

function normaliseTelemetryEntries(payload) {
  if (!payload) return [];
  if (!Array.isArray(payload)) return [];
  return payload.slice(0, 5).map((entry) => ({
    title: entry?.event || entry?.name || 'Telemetry event',
    details: entry?.message || entry?.detail || entry?.status || '',
    timestamp: entry?.timestamp || entry?.time || entry?.created_at || '',
  }));
}

function computePassRate(history) {
  if (!history || !history.length) return '--';
  const passes = history.filter((entry) => entry.correct).length;
  return ((passes / history.length) * 100).toFixed(0);
}

const groupColorCache = {};
function colorForGroup(group) {
  const key = group || 'unknown';
  if (!groupColorCache[key]) {
    const index = Object.keys(groupColorCache).length % GROUP_COLOR_PALETTE.length;
    groupColorCache[key] = GROUP_COLOR_PALETTE[index];
  }
  return groupColorCache[key];
}

function formatSchedule(meta) {
  const schedule = meta?.schedule;
  if (!schedule) return 'Not specified';
  const parts = [];
  const weekDays = schedule.WeekDays || schedule.weekDays;
  if (Array.isArray(weekDays) && weekDays.length) {
    const filtered = weekDays.filter((day) => day && day !== 'NONE');
    if (filtered.length) {
      parts.push(`Weekdays: ${filtered.join(', ')}`);
    }
  }
  const monthDays = schedule.MonthDays || schedule.monthDays;
  if (Array.isArray(monthDays) && monthDays.length) {
    parts.push(`Month days: ${monthDays.join(', ')}`);
  }
  const fromTime = schedule.FromTime || schedule.fromTime;
  const toTime = schedule.ToTime || schedule.toTime;
  if (fromTime || toTime) {
    parts.push(`Window: ${fromTime || '--'} - ${toTime || '--'}`);
  }
  if (schedule.DaysRelation || schedule.daysRelation) {
    parts.push(`Relation: ${schedule.DaysRelation || schedule.daysRelation}`);
  }
  return parts.length ? parts.join(' • ') : 'Not specified';
}

function formatVariables(variables) {
  if (!Array.isArray(variables) || !variables.length) return null;
  const entries = [];
  variables.forEach((item) => {
    if (typeof item === 'object' && item !== null) {
      Object.entries(item).forEach(([key, value]) => {
        entries.push(`${key}=${value}`);
      });
    }
  });
  return entries.length ? entries.join(' • ') : null;
}

function JobDetail({ job }) {
  if (!job) return null;
  const meta = job.meta || {};
  const schedule = formatSchedule(meta);
  const variables = formatVariables(meta.variables);
  return (
    <div className="detail-container">
      <div className="detail-title">{job.id}</div>
      <div className="detail-sub">{meta.application || 'No application specified'}</div>
      <dl className="detail-list">
        <dt>Type</dt>
        <dd>{meta.type || '—'}</dd>
        <dt>Host</dt>
        <dd>{meta.host || '—'}</dd>
        <dt>Run as</dt>
        <dd>{meta.run_as || '—'}</dd>
        <dt>Priority</dt>
        <dd>{meta.priority || '—'}</dd>
        <dt>Control-M server</dt>
        <dd>{meta.controlm_server || '—'}</dd>
        <dt>Schedule</dt>
        <dd>{schedule}</dd>
        {meta.description && (
          <>
            <dt>Description</dt>
            <dd>{meta.description}</dd>
          </>
        )}
        {variables && (
          <>
            <dt>Variables</dt>
            <dd>{variables}</dd>
          </>
        )}
        {meta.command && (
          <>
            <dt>Command</dt>
            <dd>
              <code className="detail-code">{meta.command}</code>
            </dd>
          </>
        )}
      </dl>
    </div>
  );
}

function TransitionDetail({ transition }) {
  if (!transition) return null;
  const { source, target, value, sourceMeta = {}, targetMeta = {} } = transition;
  return (
    <div className="detail-container">
      <div className="detail-title">
        {source} → {target}
      </div>
      <div className="detail-sub">{value} transitions observed</div>
      <div className="detail-pairs">
        <div className="detail-column">
          <h4>Source</h4>
          <div>{sourceMeta.application || '—'}</div>
          {sourceMeta.host && <div>{sourceMeta.host}</div>}
          {sourceMeta.command && <code className="detail-code">{sourceMeta.command}</code>}
        </div>
        <div className="detail-column">
          <h4>Target</h4>
          <div>{targetMeta.application || '—'}</div>
          {targetMeta.host && <div>{targetMeta.host}</div>}
          {targetMeta.command && <code className="detail-code">{targetMeta.command}</code>}
        </div>
      </div>
    </div>
  );
}
