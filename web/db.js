const DOM = {
    status: document.getElementById("statusBanner"),
    tableList: document.getElementById("tableList"),
    tableSearch: document.getElementById("tableSearch"),
    columnBody: document.getElementById("columnBody"),
    selectedTable: document.getElementById("selectedTable"),
    rowLimit: document.getElementById("rowLimit"),
    queryInput: document.getElementById("queryInput"),
    runQuery: document.getElementById("runQuery"),
    resultHead: document.getElementById("resultHead"),
    resultBody: document.getElementById("resultBody"),
    resultInfo: document.getElementById("resultInfo"),
    copyResults: document.getElementById("copyResults"),
    gatewayInfo: document.getElementById("gatewayInfo"),
};

const state = {
    apiBase: window.location.origin,
    tables: [],
    filteredTables: [],
    selected: null,
    defaultLimit: 200,
    allowMutations: false,
    lastResultJSON: "",
};

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
            // ignore JSON parse errors
        }
        throw new Error(detail);
    }
    return response.json();
}

function currentLimit() {
    const value = Number.parseInt(DOM.rowLimit.value, 10);
    if (Number.isNaN(value) || value <= 0) {
        return state.defaultLimit;
    }
    return value;
}

function updateQueryTemplate() {
    if (!state.selected) return;
    const { schema, table } = state.selected;
    DOM.queryInput.value = `SELECT * FROM ${schema}.${table} LIMIT ${currentLimit()};`;
}

function renderTableList() {
    DOM.tableList.innerHTML = "";
    if (state.filteredTables.length === 0) {
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

async function loadTables() {
    const response = await fetchJSON(`${state.apiBase}/db/tables`);
    state.tables = response.tables || [];
    state.filteredTables = state.tables;
    renderTableList();
}

async function loadColumns(schema, table) {
    const columns = await fetchJSON(`${state.apiBase}/db/table/${encodeURIComponent(schema)}/${encodeURIComponent(table)}`);
    renderColumns(columns);
}

async function runQuery() {
    if (!state.selected) return;
    DOM.runQuery.disabled = true;
    setStatus("Running query…");
    try {
        const payload = {
            sql: DOM.queryInput.value,
            limit: currentLimit(),
        };
        const result = await fetchJSON(`${state.apiBase}/db/query`, {
            method: "POST",
            body: JSON.stringify(payload),
        });
        renderResults(result);
        setStatus(state.allowMutations ? "Query executed (mutations enabled)" : "Query executed", "ok");
    } catch (error) {
        console.error("Query failed", error);
        setStatus(`Query failed: ${error.message}`, "error");
        DOM.resultInfo.textContent = "Query failed.";
        DOM.copyResults.disabled = true;
    } finally {
        DOM.runQuery.disabled = false;
    }
}

async function selectTable(schema, table) {
    state.selected = { schema, table };
    renderTableList();
    DOM.selectedTable.textContent = `${schema}.${table}`;
    updateQueryTemplate();
    try {
        await Promise.all([loadColumns(schema, table), runQuery()]);
    } catch (error) {
        console.error("Failed to load table", error);
        setStatus(`Failed to load table: ${error.message}`, "error");
    }
}

function filterTables(event) {
    const term = event.target.value.toLowerCase();
    state.filteredTables = term
        ? state.tables.filter((entry) =>
              `${entry.table_schema}.${entry.table_name}`.toLowerCase().includes(term),
          )
        : state.tables;
    renderTableList();
}

async function copyResultsToClipboard() {
    if (!state.lastResultJSON) return;
    try {
        await navigator.clipboard.writeText(state.lastResultJSON);
        setStatus("Copied JSON to clipboard.", "ok");
    } catch (error) {
        console.error("Clipboard copy failed", error);
        setStatus("Failed to copy to clipboard.", "error");
    }
}

async function loadConfig() {
    try {
        const config = await fetchJSON(`${window.location.origin}/api/v1/config`);
        const dbUrl = config.db_admin_gateway_url || config.telemetry_gateway_url || "";
        state.apiBase = dbUrl || window.location.origin;
        DOM.gatewayInfo.textContent = `Gateway: ${state.apiBase}`;
        return config;
    } catch (error) {
        console.warn("Config load failed, defaulting to same origin", error);
        state.apiBase = window.location.origin;
        DOM.gatewayInfo.textContent = "Gateway: (fallback) same origin";
        return { db_admin_enabled: true };
    }
}

async function fetchStatus() {
    const status = await fetchJSON(`${state.apiBase}/db/status`);
    state.defaultLimit = status.default_limit || state.defaultLimit;
    state.allowMutations = Boolean(status.allow_mutations);
    DOM.rowLimit.value = state.defaultLimit;

    if (!status.enabled) {
        setStatus("Database admin gateway is disabled.", "error");
        DOM.runQuery.disabled = true;
        return false;
    }

    setStatus(
        state.allowMutations ? "Gateway ready – mutations enabled" : "Gateway ready – read-only",
        "ok",
    );
    return true;
}

function bindEvents() {
    DOM.tableSearch.addEventListener("input", filterTables);
    DOM.runQuery.addEventListener("click", runQuery);
    DOM.copyResults.addEventListener("click", copyResultsToClipboard);
    DOM.rowLimit.addEventListener("change", updateQueryTemplate);
    DOM.queryInput.addEventListener("keydown", (event) => {
        if (event.metaKey && event.key.toLowerCase() === "enter") {
            runQuery().catch(() => {});
        }
    });
}

async function bootstrap() {
    const config = await loadConfig();
    if (config && config.db_admin_enabled === false) {
        setStatus("Database admin gateway disabled in config.", "error");
        DOM.runQuery.disabled = true;
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
