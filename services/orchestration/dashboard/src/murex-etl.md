---
title: Murex ETL to SAP Dashboard
description: Monitor ETL transformations and SAP GL journal entries
---

```js
import {html} from "@observablehq/stdlib";
import {emptyStateNoRequest} from "../components/emptyState.js";
```

# Murex ETL to SAP Dashboard

<div class="fade-in">

Monitor ETL transformations from Murex trades to SAP GL journal entries. Track data flow, transformations, and SAP integration status.

</div>

## ETL Overview

```js
html`<div class="card">
  <h3>ETL Pipeline</h3>
  <p style="color: #86868b; margin-top: 8px;">
    The Murex ETL pipeline transforms trade data into SAP GL journal entries:
  </p>
  <div style="margin-top: 16px; padding: 16px; background: #F2F2F7; border-radius: 8px;">
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
      <div style="width: 40px; height: 40px; border-radius: 50%; background: #007AFF; color: white; display: flex; align-items: center; justify-content: center; font-weight: 600;">1</div>
      <div>
        <div style="font-weight: 600;">Murex API</div>
        <div style="font-size: 12px; color: #86868b;">Extract trades from Murex</div>
      </div>
    </div>
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
      <div style="width: 40px; height: 40px; border-radius: 50%; background: #34C759; color: white; display: flex; align-items: center; justify-content: center; font-weight: 600;">2</div>
      <div>
        <div style="font-weight: 600;">Transform</div>
        <div style="font-size: 12px; color: #86868b;">Map to SAP GL format</div>
      </div>
    </div>
    <div style="display: flex; align-items: center; gap: 12px;">
      <div style="width: 40px; height: 40px; border-radius: 50%; background: #FF9500; color: white; display: flex; align-items: center; justify-content: center; font-weight: 600;">3</div>
      <div>
        <div style="font-weight: 600;">SAP GL</div>
        <div style="font-size: 12px; color: #86868b;">Load journal entries</div>
      </div>
    </div>
  </div>
</div>`
```

## Transformation Mapping

```js
html`<div class="card">
  <h3>Field Mappings</h3>
  <div style="margin-top: 16px;">
    <table style="width: 100%; border-collapse: collapse;">
      <thead>
        <tr style="border-bottom: 1px solid #E5E5EA;">
          <th style="text-align: left; padding: 12px; font-weight: 600;">Murex Field</th>
          <th style="text-align: left; padding: 12px; font-weight: 600;">SAP GL Field</th>
          <th style="text-align: left; padding: 12px; font-weight: 600;">Transformation</th>
        </tr>
      </thead>
      <tbody>
        <tr style="border-bottom: 1px solid #F2F2F7;">
          <td style="padding: 12px;">trade_id</td>
          <td style="padding: 12px;">entry_id</td>
          <td style="padding: 12px; color: #86868b;">JE-{trade_id}</td>
        </tr>
        <tr style="border-bottom: 1px solid #F2F2F7;">
          <td style="padding: 12px;">trade_date</td>
          <td style="padding: 12px;">entry_date</td>
          <td style="padding: 12px; color: #86868b;">Identity</td>
        </tr>
        <tr style="border-bottom: 1px solid #F2F2F7;">
          <td style="padding: 12px;">notional_amount</td>
          <td style="padding: 12px;">debit_amount</td>
          <td style="padding: 12px; color: #86868b;">Identity</td>
        </tr>
        <tr style="border-bottom: 1px solid #F2F2F7;">
          <td style="padding: 12px;">notional_amount</td>
          <td style="padding: 12px;">credit_amount</td>
          <td style="padding: 12px; color: #86868b;">Copy</td>
        </tr>
        <tr>
          <td style="padding: 12px;">counterparty_id</td>
          <td style="padding: 12px;">account</td>
          <td style="padding: 12px; color: #86868b;">Lookup</td>
        </tr>
      </tbody>
    </table>
  </div>
</div>`
```

## ETL Status

```js
html`<div class="card">
  <h3>ETL Status</h3>
  <p style="color: #86868b; margin-top: 8px;">
    View ETL status in the Results dashboard using a request ID.
  </p>
  <div style="margin-top: 16px;">
    <a href="/murex-results" class="button-primary">View Results</a>
  </div>
</div>`
```

