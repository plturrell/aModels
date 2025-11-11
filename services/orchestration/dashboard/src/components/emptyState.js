/**
 * Beautiful empty states for dashboards
 * Designed with Jobs & Ive lens - inviting, not empty
 */

import {html} from "@observablehq/stdlib";

/**
 * Empty state for no request ID
 */
export function emptyStateNoRequest() {
  return html`
    <div style="
      text-align: center;
      padding: 64px 32px;
      background: white;
      border-radius: 12px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    ">
      <div style="
        font-size: 48px;
        margin-bottom: 16px;
      ">📊</div>
      <h3 style="
        font-size: 24px;
        font-weight: 600;
        color: #1d1d1f;
        margin: 0 0 8px 0;
      ">No Request Selected</h3>
      <p style="
        font-size: 16px;
        color: #86868b;
        margin: 0 0 24px 0;
      ">Enter a request ID to view processing status</p>
      <p style="
        font-size: 14px;
        color: #86868b;
        margin: 0;
      ">Or use <code style="background: #F2F2F7; padding: 2px 6px; border-radius: 4px;">?request_id=xxx</code> in the URL</p>
    </div>
  `;
}

/**
 * Empty state for no data
 */
export function emptyStateNoData(message = "No data available") {
  return html`
    <div style="
      text-align: center;
      padding: 64px 32px;
      background: white;
      border-radius: 12px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    ">
      <div style="
        font-size: 48px;
        margin-bottom: 16px;
      ">📭</div>
      <h3 style="
        font-size: 24px;
        font-weight: 600;
        color: #1d1d1f;
        margin: 0 0 8px 0;
      ">${message}</h3>
      <p style="
        font-size: 16px;
        color: #86868b;
        margin: 0;
      ">Try processing some documents first</p>
    </div>
  `;
}

/**
 * Empty state for loading
 */
export function emptyStateLoading(message = "Loading...") {
  return html`
    <div style="
      text-align: center;
      padding: 64px 32px;
      background: white;
      border-radius: 12px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    ">
      <div style="
        width: 48px;
        height: 48px;
        border: 3px solid #F2F2F7;
        border-top-color: #007AFF;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin: 0 auto 16px;
      "></div>
      <p style="
        font-size: 16px;
        color: #86868b;
        margin: 0;
      ">${message}</p>
      <style>
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      </style>
    </div>
  `;
}

/**
 * Empty state for errors
 */
export function emptyStateError(error, retry = null) {
  return html`
    <div style="
      text-align: center;
      padding: 64px 32px;
      background: white;
      border-radius: 12px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08);
      border: 1px solid #FFEBEE;
    ">
      <div style="
        font-size: 48px;
        margin-bottom: 16px;
      ">⚠️</div>
      <h3 style="
        font-size: 24px;
        font-weight: 600;
        color: #FF3B30;
        margin: 0 0 8px 0;
      ">Error</h3>
      <p style="
        font-size: 16px;
        color: #86868b;
        margin: 0 0 24px 0;
      ">${error.message || error}</p>
      ${retry ? html`
        <button onclick=${retry} style="
          padding: 12px 24px;
          font-size: 16px;
          font-weight: 500;
          background: #007AFF;
          color: white;
          border: none;
          border-radius: 8px;
          cursor: pointer;
        ">Retry</button>
      ` : null}
    </div>
  `;
}

