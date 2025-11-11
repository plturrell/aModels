"""Domain performance monitoring dashboard.

This module provides a web dashboard for monitoring domain performance metrics.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import http.server
import socketserver
import urllib.parse

logger = logging.getLogger(__name__)


class DomainDashboard:
    """Web dashboard for domain performance monitoring."""
    
    def __init__(
        self,
        metrics_collector,
        port: int = 8085,
        data_dir: str = "./dashboard_data"
    ):
        """Initialize dashboard.
        
        Args:
            metrics_collector: DomainMetricsCollector instance
            port: Port to serve dashboard on
            data_dir: Directory for dashboard data files
        """
        self.metrics_collector = metrics_collector
        self.port = port
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_dashboard_data(self, domain_ids: Optional[List[str]] = None) -> str:
        """Generate dashboard HTML.
        
        Args:
            domain_ids: Optional list of domains to display
        
        Returns:
            HTML content for dashboard
        """
        if domain_ids is None:
            domain_ids = self.metrics_collector._get_all_domain_ids()
        
        # Collect metrics for all domains
        dashboard_data = {}
        for domain_id in domain_ids:
            metrics = self.metrics_collector.collect_domain_metrics(domain_id, time_window_days=30)
            dashboard_data[domain_id] = metrics
        
        # Generate comparison
        comparison = self.metrics_collector.get_domain_comparison(domain_ids)
        
        # Generate HTML
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Domain Performance Dashboard</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .domain-card {{
            background-color: white;
            border-radius: 5px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric {{
            display: inline-block;
            margin: 10px;
            padding: 10px;
            background-color: #ecf0f1;
            border-radius: 3px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-label {{
            font-size: 12px;
            color: #7f8c8d;
        }}
        .trend-up {{ color: #27ae60; }}
        .trend-down {{ color: #e74c3c; }}
        .trend-stable {{ color: #f39c12; }}
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        .comparison-table th, .comparison-table td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .comparison-table th {{
            background-color: #34495e;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Domain Performance Dashboard</h1>
        <p>Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div id="domains">
"""
        
        # Add domain cards
        for domain_id, metrics in dashboard_data.items():
            performance = metrics.get("performance", {}).get("latest", {})
            trends = metrics.get("trends", {}).get("direction", {})
            
            html += f"""
        <div class="domain-card">
            <h2>{domain_id}</h2>
            <div class="metric">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value">{performance.get('accuracy', 0):.3f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Latency (ms)</div>
                <div class="metric-value">{performance.get('latency_ms', 0):.1f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Training Loss</div>
                <div class="metric-value">{performance.get('training_loss', 0):.4f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Validation Loss</div>
                <div class="metric-value">{performance.get('validation_loss', 0):.4f}</div>
            </div>
            <div style="margin-top: 10px;">
                <strong>Trends:</strong>
                {', '.join([f"{k}: <span class='trend-{trends.get(k, 'stable')}'>{trends.get(k, 'stable')}</span>" for k in ['accuracy', 'latency_ms']])}
            </div>
        </div>
"""
        
        # Add comparison table
        if comparison.get("rankings"):
            html += """
    <div class="domain-card">
        <h2>Domain Rankings</h2>
        <table class="comparison-table">
            <tr>
                <th>Domain</th>
                <th>Accuracy</th>
                <th>Latency (ms)</th>
            </tr>
"""
            for ranking in comparison.get("rankings", {}).get("accuracy", []):
                domain_id = ranking.get("domain_id")
                accuracy = ranking.get("value", 0)
                latency = next(
                    (r.get("value", 0) for r in comparison.get("rankings", {}).get("latency", []) if r.get("domain_id") == domain_id),
                    0
                )
                html += f"""
            <tr>
                <td>{domain_id}</td>
                <td>{accuracy:.3f}</td>
                <td>{latency:.1f}</td>
            </tr>
"""
            html += """
        </table>
    </div>
"""
        
        html += """
    </div>
</body>
</html>
"""
        
        return html
    
    def serve(self):
        """Start HTTP server for dashboard."""
        class DashboardHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, dashboard_instance=None, **kwargs):
                self.dashboard = dashboard_instance
                super().__init__(*args, **kwargs)
            
            def do_GET(self):
                if self.path == '/' or self.path == '/dashboard':
                    html = self.dashboard.generate_dashboard_data()
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(html.encode())
                elif self.path.startswith('/api/metrics'):
                    # API endpoint for metrics
                    params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                    domain_id = params.get('domain_id', [None])[0]
                    
                    if domain_id:
                        metrics = self.dashboard.metrics_collector.collect_domain_metrics(domain_id)
                    else:
                        domain_ids = self.dashboard.metrics_collector._get_all_domain_ids()
                        metrics = {
                            "domains": {
                                did: self.dashboard.metrics_collector.collect_domain_metrics(did)
                                for did in domain_ids
                            }
                        }
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(metrics).encode())
                else:
                    self.send_response(404)
                    self.end_headers()
        
        handler = lambda *args, **kwargs: DashboardHandler(*args, dashboard_instance=self, **kwargs)
        
        with socketserver.TCPServer(("", self.port), handler) as httpd:
            logger.info(f"Domain dashboard serving on http://localhost:{self.port}")
            httpd.serve_forever()

