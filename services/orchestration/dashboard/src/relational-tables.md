---
title: Relational Tables Library
description: Browse and explore database tables
---

```js
import {html} from "@observablehq/stdlib";
import {emptyStateNoData} from "../components/emptyState.js";
```

# Relational Tables Library

<div class="fade-in">

Browse your database tables, view processing status, and access table details.

</div>

## Connect to Database

```js
html`<div class="card">
  <h3>Database Connection</h3>
  <p style="color: #86868b; margin-top: 8px;">
    To browse tables, process them through the Relational API endpoint:
  </p>
  <div style="margin-top: 16px; padding: 16px; background: #F2F2F7; border-radius: 8px; font-family: monospace; font-size: 12px;">
    POST /api/relational/process<br/>
    {<br/>
    &nbsp;&nbsp;"table": "table_name",<br/>
    &nbsp;&nbsp;"schema": "schema_name",<br/>
    &nbsp;&nbsp;"database_url": "postgres://...",<br/>
    &nbsp;&nbsp;"database_type": "postgres"<br/>
    }
  </div>
</div>`
```

## Processed Tables

```js
html`<div class="card">
  <h3>Processed Tables</h3>
  <p style="color: #86868b; margin-top: 8px;">
    View processed tables in the Results dashboard using a request ID.
  </p>
  <div style="margin-top: 16px;">
    <a href="/relational-results" class="button-primary">View Results</a>
  </div>
</div>`
```

