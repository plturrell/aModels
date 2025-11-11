/**
 * Export utilities for charts and data
 * Uses Observable Stdlib for file downloads
 */

import {DOM} from "@observablehq/stdlib";

/**
 * Export chart as PNG
 */
export function exportChartPNG(chartElement, filename = "chart.png") {
  if (!chartElement) return;
  
  // Get SVG from Plot chart
  const svg = chartElement.querySelector("svg");
  if (!svg) return;
  
  // Convert SVG to PNG
  const svgData = new XMLSerializer().serializeToString(svg);
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  const img = new Image();
  
  img.onload = () => {
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);
    
    canvas.toBlob((blob) => {
      DOM.download(blob, filename, "image/png");
    });
  };
  
  img.src = "data:image/svg+xml;base64," + btoa(unescape(encodeURIComponent(svgData)));
}

/**
 * Export chart as SVG
 */
export function exportChartSVG(chartElement, filename = "chart.svg") {
  if (!chartElement) return;
  
  const svg = chartElement.querySelector("svg");
  if (!svg) return;
  
  const svgData = new XMLSerializer().serializeToString(svg);
  const blob = new Blob([svgData], {type: "image/svg+xml"});
  DOM.download(blob, filename, "image/svg+xml");
}

/**
 * Export data as JSON
 */
export function exportJSON(data, filename = "data.json") {
  const json = JSON.stringify(data, null, 2);
  const blob = new Blob([json], {type: "application/json"});
  DOM.download(blob, filename, "application/json");
}

/**
 * Export data as CSV
 */
export function exportCSV(data, filename = "data.csv") {
  if (!data || data.length === 0) return;
  
  // Get headers from first object
  const headers = Object.keys(data[0]);
  
  // Create CSV rows
  const rows = [
    headers.join(","),
    ...data.map(row => 
      headers.map(header => {
        const value = row[header];
        // Escape quotes and wrap in quotes if contains comma
        if (typeof value === "string" && (value.includes(",") || value.includes('"'))) {
          return `"${value.replace(/"/g, '""')}"`;
        }
        return value ?? "";
      }).join(",")
    )
  ];
  
  const csv = rows.join("\n");
  const blob = new Blob([csv], {type: "text/csv"});
  DOM.download(blob, filename, "text/csv");
}

