const STORAGE_KEY = "postgres_admin_history_v1";
const HISTORY_LIMIT = 50;

const DOM = {
    status: document.getElementById("statusBanner"),
    tableList: document.getElementById("tableList"),
    tableSearch: document.getElementById("tableSearch"),
    columnBody: document.getElementById("columnBody"),
    selectedTable: document.getElementById("selectedTable"),
    rowLimit: document.getElementById("rowLimit"),
    queryInput: document.getElementById("queryInput"),
    runQuery: document.getElementById("runQuery"),
    explainQuery: document.getElementById("explainQuery"),
    formatQuery: document.getElementById("formatQuery"),
    clearQuery: document.getElementById("clearQuery"),
    resultHead: document.getElementById("resultHead"),
    resultBody: document.getElementById("resultBody"),
    resultInfo: document.getElementById("resultInfo"),
    copyResults: document.getElementById("copyResults"),
    gatewayInfo: document.getElementById("gatewayInfo"),
    historyList: document.getElementById("historyList"),
    clearHistory: document.getElementById("clearHistory"),
};

const state = {
    apiBase: window.location.origin,
    tables: [],
    filteredTables: [],
    selected: null,
    defaultLimit: 200,
    allowMutations: false,
    lastResultJSON: "",
    history: loadHistory(),
};

let editor = null;

function setStatus(message, variant = "info") {
    DOM.status.textContent = message;
    DOM.status.classList.remove("ok", "error");
    if (variant === "ok") DOM.status.classList.add("ok");
    if (variant === "error") DOM.status.classList.add("error");
}

async function fetchJSON(url, options = {}) {
    const response = await fetch(url, {
        ...options,
        headers: {
            "Content-Type": "application/json",
            ...(options.headers || {}),
        },
    });
    if (!response.ok) {
        let detail = `${response.status}`;
        try {
            const payload = await response.json();
            detail = payload.detail || payload.message || detail;
        } catch (_) {
            // ignore JSON parse issues
        }
        throw new Error(detail);
    }
    return response.json();
}

function loadHistory() {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return [];
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) {
            return parsed.slice(0, HISTORY_LIMIT);
        }
    } catch (error) {
        console.warn("Failed to parse history", error);
    }
    return [];
}

function saveHistory() {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(state.history.slice(0, HISTORY_LIMIT)));
    } catch (error) {
        console.warn("Failed to persist history", error);
    }
}

function renderHistory() {
    DOM.historyList.innerHTML = "";
    if (!state.history.length) {
        const li = document.createElement("li");
        li.className = "empty";
        li.textContent = "No queries yet.";
        DOM.historyList.appendChild(li);
        return;
    }

    state.history.forEach((entry, index) => {
        const li = document.createElement("li");
        li.dataset.index = String(index);

        const snippet = document.createElement("span");
        snippet.className = "sql-snippet";
        snippet.textContent = entry.sql.length > 120 ? `${entry.sql.slice(0, 117)}…` : entry.sql;

        const meta = document.createElement("div");
        meta.className = "meta";
        meta.innerHTML = `<span>${formatTimestamp(entry.savedAt)}</span><span>${entry.mode === "explain" ? "Explain" : "Run"}</span>`;

        li.appendChild(snippet);
        li.appendChild(meta);
        DOM.historyList.appendChild(li);
    });
}

function pushHistory(entry) {
    const normalizedSQL = entry.sql.trim();
    state.history = state.history.filter((item) => !(item.sql.trim() === normalizedSQL && item.mode === entry.mode));
    state.history.unshift({ ...entry, sql: normalizedSQL });
    if (state.history.length > HISTORY_LIMIT) {
        state.history.length = HISTORY_LIMIT;
    }
    saveHistory();
    renderHistory();
}

function clearHistory() {
    state.history = [];
    saveHistory();
    renderHistory();
}

function currentLimit() {
    const value = Number.parseInt(DOM.rowLimit.value, 10);
    if (Number.isNaN(value) || value <= 0) {
        return state.defaultLimit;
    }
    return value;
}

function getCurrentSQL() {
    if (editor) {
        return editor.getValue();
    }
    return DOM.queryInput.value;
}

function setEditorValue(value) {
    if (editor) {
        const cursor = editor.getCursor();
        editor.setValue(value);
        editor.setCursor(cursor);
    } else {
        DOM.queryInput.value = value;
    }
}

function updateQueryTemplate() {
    if (!state.selected) return;
    const { schema, table } = state.selected;
    const template = `SELECT * FROM ${schema}.${table} LIMIT ${currentLimit()};`;
    setEditorValue(template);
}

function renderTableList() {
    DOM.tableList.innerHTML = "";
    if (!state.filteredTables.length) {
        const item = document.createElement("li");
        item.className = "empty";
        item.textContent = "No tables found.";
        DOM.tableList.appendChild(item);
        return;
    }

    for (const table of state.filteredTables) {
        const item = document.createElement("li");
        item.dataset.schema = table.table_schema;
        item.dataset.table = table.table_name;
        item.textContent = `${table.table_schema}.${table.table_name}`;
        if (
            state.selected &&
            state.selected.schema === table.table_schema &&
            state.selected.table === table.table_name
        ) {
            item.classList.add("active");
        }
        item.addEventListener("click", () => selectTable(table.table_schema, table.table_name));
        DOM.tableList.appendChild(item);
    }
}

function renderColumns(columns) {
    DOM.columnBody.innerHTML = "";
    if (!columns.length) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");
        cell.colSpan = 4;
        cell.className = "empty";
        cell.textContent = "No columns available for this table.";
        row.appendChild(cell);
        DOM.columnBody.appendChild(row);
        return;
    }

    for (const column of columns) {
        const row = document.createElement("tr");
        row.innerHTML = `
            <td>${column.column_name}</td>
            <td>${column.data_type}</td>
            <td>${column.is_nullable}</td>
            <td>${column.column_default ?? ""}</td>
        `;
        DOM.columnBody.appendChild(row);
    }
}

function renderResults(result) {
    state.lastResultJSON = JSON.stringify(result.rows ?? [], null, 2);
    DOM.resultHead.innerHTML = "";
    DOM.resultBody.innerHTML = "";

    if (!result.columns?.length) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");
        cell.className = "empty";
        cell.textContent = `Statement executed. ${result.row_count || 0} row(s) affected.`;
        row.appendChild(cell);
        DOM.resultBody.appendChild(row);
        DOM.copyResults.disabled = true;
        return;
    }

    const headRow = document.createElement("tr");
    for (const column of result.columns) {
        const cell = document.createElement("th");
        cell.textContent = column;
        headRow.appendChild(cell);
    }
    DOM.resultHead.appendChild(headRow);

    if (!result.rows || result.rows.length === 0) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");
        cell.colSpan = result.columns.length;
        cell.className = "empty";
        cell.textContent = "No rows returned.";
        row.appendChild(cell);
        DOM.resultBody.appendChild(row);
        DOM.copyResults.disabled = true;
        return;
    }

    for (const rowData of result.rows) {
        const row = document.createElement("tr");
        for (const column of result.columns) {
            const cell = document.createElement("td");
            const value = rowData[column];
            cell.textContent = value === null || value === undefined ? "NULL" : String(value);
            row.appendChild(cell);
        }
        DOM.resultBody.appendChild(row);
    }

    DOM.copyResults.disabled = false;
    let info = `${result.row_count} row${result.row_count === 1 ? "" : "s"}`;
    if (result.truncated) {
        info += " · truncated to limit";
    }
    DOM.resultInfo.textContent = info;
}

function renderExplain(result) {
    DOM.resultHead.innerHTML = "<tr><th>Plan</th></tr>";
    DOM.resultBody.innerHTML = "";

    let planText = "";
    if (Array.isArray(result.rows) && result.rows.length) {
        const firstRow = result.rows[0];
        const firstKey = Object.keys(firstRow)[0];
        const rawPlan = firstRow[firstKey];
        if (typeof rawPlan === "string") {
            try {
                const parsed = JSON.parse(rawPlan);
                planText = JSON.stringify(parsed, null, 2);
            } catch (_) {
                planText = rawPlan;
            }
        } else if (rawPlan) {
            planText = JSON.stringify(rawPlan, null, 2);
        }
    }

    if (!planText) {
        planText = JSON.stringify(result.rows ?? [], null, 2);
    }

    state.lastResultJSON = planText;

    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.className = "plan-cell";
    const pre = document.createElement("pre");
    pre.className = "plan-json";
    pre.textContent = planText;
    cell.appendChild(pre);
    row.appendChild(cell);
    DOM.resultBody.appendChild(row);

    DOM.copyResults.disabled = false;
    DOM.resultInfo.textContent = "Explain plan ready.";
}

function loadTables() {
    return fetchJSON(`${state.apiBase}/db/tables`).then((data) => {
        state.tables = data.tables || [];
        state.filteredTables = state.tables;
        renderTableList();
    });
}

function loadColumns(schema, table) {
    return fetchJSON(`${state.apiBase}/db/table/${encodeURIComponent(schema)}/${encodeURIComponent(table)}`).then((columns) => {
        renderColumns(columns);
    });
}

async function executeQuery(mode = "run", options = {}) {
    const skipHistory = options.skipHistory || false;
    const sql = getCurrentSQL().trim();
    if (!sql) {
        setStatus("Query is empty.", "error");
        return;
    }

    const limit = currentLimit();
    let payloadSQL = sql;
    const payload = {};

    if (mode === "explain") {
        if (!/^explain\b/i.test(payloadSQL)) {
            payloadSQL = `EXPLAIN (FORMAT JSON) ${payloadSQL.replace(/;+\s*$/, "")}`;
        }
        setStatus("Generating explain plan…");
    } else {
        setStatus("Running query…");
        if (limit) {
            payload.limit = limit;
        }
    }

    payload.sql = payloadSQL;
    DOM.runQuery.disabled = true;
    DOM.explainQuery.disabled = true;

    try {
        const result = await fetchJSON(`${state.apiBase}/db/query`, {
            method: "POST",
            body: JSON.stringify(payload),
        });

        if (mode === "explain") {
            renderExplain(result);
        } else {
            renderResults(result);
        }

        if (!skipHistory) {
            pushHistory({ sql, mode, limit: mode === "run" ? limit : null, savedAt: new Date().toISOString() });
        }

        setStatus(
            mode === "explain"
                ? "Explain plan generated successfully."
                : state.allowMutations
                ? "Query executed (mutations enabled)."
                : "Query executed.",
            "ok",
        );
    } catch (error) {
        console.error("Query failed", error);
        setStatus(`Query failed: ${error.message}`, "error");
        DOM.resultInfo.textContent = "Query failed.";
        DOM.copyResults.disabled = true;
        state.lastResultJSON = "";
    } finally {
        DOM.runQuery.disabled = false;
        DOM.explainQuery.disabled = false;
    }
}

async function selectTable(schema, table) {
    state.selected = { schema, table };
    renderTableList();
    DOM.selectedTable.textContent = `${schema}.${table}`;
    updateQueryTemplate();
    try {
        await Promise.all([loadColumns(schema, table), executeQuery("run", { skipHistory: true })]);
    } catch (error) {
        console.error("Failed to load table", error);
        setStatus(`Failed to load table: ${error.message}`, "error");
    }
}

function filterTables(event) {
    const term = event.target.value.toLowerCase();
    state.filteredTables = term
        ? state.tables.filter((entry) => `${entry.table_schema}.${entry.table_name}`.toLowerCase().includes(term))
        : state.tables;
    renderTableList();
}

function copyResultsToClipboard() {
    if (!state.lastResultJSON) return;
    navigator.clipboard
        .writeText(state.lastResultJSON)
        .then(() => setStatus("Copied JSON to clipboard.", "ok"))
        .catch((error) => {
            console.error("Clipboard copy failed", error);
            setStatus("Failed to copy to clipboard.", "error");
        });
}

function loadConfig() {
    return fetchJSON(`${window.location.origin}/api/v1/config`)
        .then((config) => {
            const dbUrl = config.db_admin_gateway_url || config.telemetry_gateway_url || "";
            state.apiBase = dbUrl || window.location.origin;
            DOM.gatewayInfo.textContent = `Gateway: ${state.apiBase}`;
            return config;
        })
        .catch((error) => {
            console.warn("Config load failed, defaulting to same origin", error);
            state.apiBase = window.location.origin;
            DOM.gatewayInfo.textContent = "Gateway: (fallback) same origin";
            return { db_admin_enabled: true };
        });
}

function fetchStatus() {
    return fetchJSON(`${state.apiBase}/db/status`).then((status) => {
        state.defaultLimit = status.default_limit || state.defaultLimit;
        state.allowMutations = Boolean(status.allow_mutations);
        DOM.rowLimit.value = state.defaultLimit;

        if (!status.enabled) {
            setStatus("Database admin gateway is disabled.", "error");
            DOM.runQuery.disabled = true;
            DOM.explainQuery.disabled = true;
            return false;
        }

        setStatus(
            state.allowMutations ? "Gateway ready – mutations enabled" : "Gateway ready – read-only",
            "ok",
        );
        return true;
    });
}

function formatCurrentQuery() {
    if (!window.sqlFormatter) {
        setStatus("SQL formatter library is not loaded.", "error");
        return;
    }
    const sql = getCurrentSQL();
    if (!sql.trim()) {
        setStatus("Nothing to format.", "error");
        return;
    }
    const formatted = window.sqlFormatter.format(sql, { language: "postgresql" });
    setEditorValue(formatted);
    setStatus("Query formatted.", "ok");
}

function clearCurrentQuery() {
    setEditorValue("");
    setStatus("Editor cleared.", "ok");
}

function handleHistoryClick(event) {
    const item = event.target.closest("li");
    if (!item || item.classList.contains("empty")) {
        return;
    }
    const index = Number.parseInt(item.dataset.index, 10);
    if (Number.isNaN(index) || !state.history[index]) {
        return;
    }
    const entry = state.history[index];
    setEditorValue(entry.sql);
    setStatus(`Loaded ${entry.mode === "explain" ? "explain" : "run"} query from history.`, "ok");
}

function formatTimestamp(value) {
    if (!value) return "";
    try {
        const date = new Date(value);
        return date.toLocaleString();
    } catch (_) {
        return value;
    }
}

function initEditor() {
    if (window.CodeMirror) {
        editor = window.CodeMirror.fromTextArea(DOM.queryInput, {
            mode: "text/x-pgsql",
            theme: "material-palenight",
            lineNumbers: true,
            autofocus: true,
            extraKeys: {
                "Cmd-Enter": () => executeQuery("run").catch(() => {}),
                "Ctrl-Enter": () => executeQuery("run").catch(() => {}),
                "Cmd-Shift-Enter": () => executeQuery("explain").catch(() => {}),
                "Ctrl-Shift-Enter": () => executeQuery("explain").catch(() => {}),
                "Alt-F": () => {
                    formatCurrentQuery();
                    return false;
                },
            },
        });
    }
}

function bindEvents() {
    DOM.tableSearch.addEventListener("input", filterTables);
    DOM.runQuery.addEventListener("click", () => executeQuery("run"));
    DOM.explainQuery.addEventListener("click", () => executeQuery("explain"));
    DOM.formatQuery.addEventListener("click", formatCurrentQuery);
    DOM.clearQuery.addEventListener("click", clearCurrentQuery);
    DOM.copyResults.addEventListener("click", copyResultsToClipboard);
    DOM.rowLimit.addEventListener("change", updateQueryTemplate);
    DOM.clearHistory.addEventListener("click", clearHistory);
    DOM.historyList.addEventListener("click", handleHistoryClick);
}

async function bootstrap() {
    initEditor();
    renderHistory();
    const config = await loadConfig();
    if (config && config.db_admin_enabled === false) {
        setStatus("Database admin gateway disabled in config.", "error");
        DOM.runQuery.disabled = true;
        DOM.explainQuery.disabled = true;
        return;
    }

    bindEvents();
    const ready = await fetchStatus();
    if (!ready) return;

    try {
        await loadTables();
    } catch (error) {
        console.error("Failed to list tables", error);
        setStatus(`Failed to list tables: ${error.message}`, "error");
    }
}

bootstrap().catch((error) => {
    console.error("Failed to initialise DB admin UI", error);
    setStatus(`Bootstrap error: ${error.message}`, "error");
});
