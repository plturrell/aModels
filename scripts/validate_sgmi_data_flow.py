#!/usr/bin/env python3
"""
Validation script for SGMI data flow consistency.
Checks data consistency across Postgres, Redis, and Neo4j storage systems.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    print("ERROR: psycopg2 not installed. Install with: pip install psycopg2-binary")
    sys.exit(1)

try:
    import redis
except ImportError:
    print("ERROR: redis not installed. Install with: pip install redis")
    sys.exit(1)

try:
    from neo4j import GraphDatabase
except ImportError:
    print("ERROR: neo4j not installed. Install with: pip install neo4j")
    sys.exit(1)


class DataFlowValidator:
    """Validates data consistency across storage systems."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "project_id": config.get("project_id", "sgmi-demo"),
            "systems": {},
            "consistency": {},
            "issues": [],
            "summary": {}
        }
        
    def validate_postgres(self) -> Dict[str, Any]:
        """Validate Postgres data."""
        print("Validating Postgres data...")
        dsn = self.config.get("postgres_dsn")
        project_id = self.config.get("project_id", "sgmi-demo")
        
        try:
            conn = psycopg2.connect(dsn)
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # Count nodes
            cur.execute("""
                SELECT COUNT(*) as count 
                FROM glean_nodes 
                WHERE properties_json->>'project_id' = %s
            """, (project_id,))
            node_count = cur.fetchone()["count"]
            
            # Count edges
            cur.execute("""
                SELECT COUNT(*) as count 
                FROM glean_edges e
                JOIN glean_nodes n1 ON e.source_id = n1.id
                WHERE n1.properties_json->>'project_id' = %s
            """, (project_id,))
            edge_count = cur.fetchone()["count"]
            
            # Count by type
            cur.execute("""
                SELECT kind, COUNT(*) as count
                FROM glean_nodes
                WHERE properties_json->>'project_id' = %s
                GROUP BY kind
            """, (project_id,))
            type_counts = {row["kind"]: row["count"] for row in cur.fetchall()}
            
            # Check for information theory metrics
            cur.execute("""
                SELECT COUNT(*) as count
                FROM glean_nodes
                WHERE properties_json->>'project_id' = %s
                AND properties_json ? 'metadata_entropy'
            """, (project_id,))
            metrics_count = cur.fetchone()["count"]
            
            # Sample node properties
            cur.execute("""
                SELECT properties_json
                FROM glean_nodes
                WHERE properties_json->>'project_id' = %s
                LIMIT 1
            """, (project_id,))
            sample = cur.fetchone()
            sample_props = sample["properties_json"] if sample else {}
            
            cur.close()
            conn.close()
            
            result = {
                "status": "ok",
                "nodes": node_count,
                "edges": edge_count,
                "type_counts": type_counts,
                "metrics_count": metrics_count,
                "sample_properties": sample_props,
                "has_data": node_count > 0 and edge_count > 0
            }
            
            print(f"  ✓ Postgres: {node_count} nodes, {edge_count} edges")
            return result
            
        except Exception as e:
            print(f"  ✗ Postgres validation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "nodes": 0,
                "edges": 0,
                "has_data": False
            }
    
    def validate_redis(self) -> Dict[str, Any]:
        """Validate Redis data."""
        print("Validating Redis data...")
        redis_url = self.config.get("redis_url", "redis://localhost:6379/0")
        
        try:
            # Parse Redis URL
            if redis_url.startswith("redis://"):
                redis_url = redis_url.replace("redis://", "")
            
            parts = redis_url.split("/")
            host_port = parts[0].split(":")
            host = host_port[0] if len(host_port) > 0 else "localhost"
            port = int(host_port[1]) if len(host_port) > 1 else 6379
            db = int(parts[1]) if len(parts) > 1 else 0
            
            r = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            r.ping()
            
            # Count schema nodes
            node_keys = list(r.scan_iter(match="schema:node:*", count=1000))
            node_count = len(node_keys)
            
            # Count schema edges
            edge_keys = list(r.scan_iter(match="schema:edge:*", count=1000))
            edge_count = len(edge_keys)
            
            # Check extract entities queue
            entity_count = r.llen("extract:entities")
            
            # Sample node data
            sample_node = None
            if node_keys:
                sample_key = node_keys[0]
                sample_node = r.hgetall(sample_key)
            
            result = {
                "status": "ok",
                "nodes": node_count,
                "edges": edge_count,
                "entities": entity_count,
                "sample_node": sample_node,
                "has_data": node_count > 0 or edge_count > 0
            }
            
            print(f"  ✓ Redis: {node_count} nodes, {edge_count} edges, {entity_count} entities")
            return result
            
        except Exception as e:
            print(f"  ✗ Redis validation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "nodes": 0,
                "edges": 0,
                "has_data": False
            }
    
    def validate_neo4j(self) -> Dict[str, Any]:
        """Validate Neo4j data."""
        print("Validating Neo4j data...")
        uri = self.config.get("neo4j_uri", "bolt://localhost:7687")
        username = self.config.get("neo4j_username", "neo4j")
        password = self.config.get("neo4j_password", "amodels123")
        project_id = self.config.get("project_id", "sgmi-demo")
        
        try:
            driver = GraphDatabase.driver(uri, auth=(username, password))
            
            with driver.session() as session:
                # Count nodes
                result = session.run("MATCH (n:Node) RETURN COUNT(n) as count")
                node_count = result.single()["count"]
                
                # Count edges
                result = session.run("MATCH ()-[r:RELATIONSHIP]->() RETURN COUNT(r) as count")
                edge_count = result.single()["count"]
                
                # Count by type
                result = session.run("""
                    MATCH (n:Node)
                    RETURN n.type as type, COUNT(n) as count
                """)
                type_counts = {record["type"]: record["count"] for record in result}
                
                # Check for information theory metrics
                result = session.run("""
                    MATCH (n:Node)
                    WHERE n.properties_json CONTAINS 'metadata_entropy'
                    RETURN COUNT(n) as count
                """)
                metrics_count = result.single()["count"]
                
                # Sample node
                result = session.run("""
                    MATCH (n:Node)
                    RETURN n LIMIT 1
                """)
                sample = result.single()
                sample_node = dict(sample["n"]) if sample else {}
            
            driver.close()
            
            result = {
                "status": "ok",
                "nodes": node_count,
                "edges": edge_count,
                "type_counts": type_counts,
                "metrics_count": metrics_count,
                "sample_node": sample_node,
                "has_data": node_count > 0 and edge_count > 0
            }
            
            print(f"  ✓ Neo4j: {node_count} nodes, {edge_count} edges")
            return result
            
        except Exception as e:
            print(f"  ✗ Neo4j validation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "nodes": 0,
                "edges": 0,
                "has_data": False
            }
    
    def check_consistency(self) -> Dict[str, Any]:
        """Check consistency across systems."""
        print("\nChecking data consistency...")
        
        pg = self.results["systems"].get("postgres", {})
        redis_data = self.results["systems"].get("redis", {})
        neo4j = self.results["systems"].get("neo4j", {})
        
        consistency = {
            "node_consistency": {},
            "edge_consistency": {},
            "issues": []
        }
        
        # Node consistency
        pg_nodes = pg.get("nodes", 0)
        redis_nodes = redis_data.get("nodes", 0)
        neo4j_nodes = neo4j.get("nodes", 0)
        
        consistency["node_consistency"] = {
            "postgres": pg_nodes,
            "redis": redis_nodes,
            "neo4j": neo4j_nodes,
            "max": max(pg_nodes, redis_nodes, neo4j_nodes),
            "min": min(pg_nodes, redis_nodes, neo4j_nodes),
            "variance": max(pg_nodes, redis_nodes, neo4j_nodes) - min(pg_nodes, redis_nodes, neo4j_nodes)
        }
        
        # Edge consistency
        pg_edges = pg.get("edges", 0)
        redis_edges = redis_data.get("edges", 0)
        neo4j_edges = neo4j.get("edges", 0)
        
        consistency["edge_consistency"] = {
            "postgres": pg_edges,
            "redis": redis_edges,
            "neo4j": neo4j_edges,
            "max": max(pg_edges, redis_edges, neo4j_edges),
            "min": min(pg_edges, redis_edges, neo4j_edges),
            "variance": max(pg_edges, redis_edges, neo4j_edges) - min(pg_edges, redis_edges, neo4j_edges)
        }
        
        # Check for issues
        node_variance = consistency["node_consistency"]["variance"]
        edge_variance = consistency["edge_consistency"]["variance"]
        
        if node_variance > 0:
            max_nodes = consistency["node_consistency"]["max"]
            variance_pct = (node_variance / max_nodes * 100) if max_nodes > 0 else 0
            if variance_pct > 5:  # More than 5% variance
                consistency["issues"].append({
                    "type": "node_count_mismatch",
                    "severity": "high" if variance_pct > 20 else "medium",
                    "message": f"Node count variance: {node_variance} ({variance_pct:.1f}%)",
                    "details": consistency["node_consistency"]
                })
        
        if edge_variance > 0:
            max_edges = consistency["edge_consistency"]["max"]
            variance_pct = (edge_variance / max_edges * 100) if max_edges > 0 else 0
            if variance_pct > 5:  # More than 5% variance
                consistency["issues"].append({
                    "type": "edge_count_mismatch",
                    "severity": "high" if variance_pct > 20 else "medium",
                    "message": f"Edge count variance: {edge_variance} ({variance_pct:.1f}%)",
                    "details": consistency["edge_consistency"]
                })
        
        # Check for missing data
        if pg_nodes == 0 and pg_edges == 0:
            consistency["issues"].append({
                "type": "missing_data",
                "severity": "high",
                "message": "No data found in Postgres",
                "system": "postgres"
            })
        
        if neo4j_nodes == 0 and neo4j_edges == 0:
            consistency["issues"].append({
                "type": "missing_data",
                "severity": "high",
                "message": "No data found in Neo4j",
                "system": "neo4j"
            })
        
        return consistency
    
    def validate(self) -> Dict[str, Any]:
        """Run all validations."""
        print("=" * 60)
        print("SGMI Data Flow Validation")
        print("=" * 60)
        print(f"Project ID: {self.config.get('project_id', 'sgmi-demo')}")
        print(f"Timestamp: {self.results['timestamp']}\n")
        
        # Validate each system
        self.results["systems"]["postgres"] = self.validate_postgres()
        self.results["systems"]["redis"] = self.validate_redis()
        self.results["systems"]["neo4j"] = self.validate_neo4j()
        
        # Check consistency
        self.results["consistency"] = self.check_consistency()
        
        # Collect all issues
        for system, data in self.results["systems"].items():
            if data.get("status") == "error":
                self.results["issues"].append({
                    "type": "system_error",
                    "severity": "high",
                    "message": f"{system.capitalize()} validation failed",
                    "error": data.get("error"),
                    "system": system
                })
        
        self.results["issues"].extend(self.results["consistency"].get("issues", []))
        
        # Generate summary
        total_issues = len(self.results["issues"])
        high_severity = sum(1 for issue in self.results["issues"] if issue.get("severity") == "high")
        
        self.results["summary"] = {
            "total_issues": total_issues,
            "high_severity": high_severity,
            "status": "ok" if total_issues == 0 else ("error" if high_severity > 0 else "warning"),
            "systems_ok": sum(1 for s in self.results["systems"].values() if s.get("status") == "ok"),
            "systems_total": len(self.results["systems"])
        }
        
        # Print summary
        print("\n" + "=" * 60)
        print("Validation Summary")
        print("=" * 60)
        print(f"Status: {self.results['summary']['status'].upper()}")
        print(f"Systems OK: {self.results['summary']['systems_ok']}/{self.results['summary']['systems_total']}")
        print(f"Total Issues: {total_issues}")
        print(f"High Severity: {high_severity}")
        
        if self.results["issues"]:
            print("\nIssues:")
            for issue in self.results["issues"]:
                print(f"  [{issue['severity'].upper()}] {issue['message']}")
        
        return self.results


def main():
    parser = argparse.ArgumentParser(description="Validate SGMI data flow consistency")
    parser.add_argument("--postgres-dsn", default=os.getenv("POSTGRES_CATALOG_DSN", "postgresql://postgres:postgres@localhost:5432/amodels?sslmode=disable"))
    parser.add_argument("--redis-url", default=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-username", default=os.getenv("NEO4J_USERNAME", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", "amodels123"))
    parser.add_argument("--project-id", default=os.getenv("PROJECT_ID", "sgmi-demo"))
    parser.add_argument("--output", help="Output JSON report file")
    
    args = parser.parse_args()
    
    config = {
        "postgres_dsn": args.postgres_dsn,
        "redis_url": args.redis_url,
        "neo4j_uri": args.neo4j_uri,
        "neo4j_username": args.neo4j_username,
        "neo4j_password": args.neo4j_password,
        "project_id": args.project_id
    }
    
    validator = DataFlowValidator(config)
    results = validator.validate()
    
    # Write output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nReport written to: {output_path}")
    else:
        # Default output location
        repo_root = Path(__file__).parent.parent
        report_dir = repo_root / "reports" / "sgmi_validation"
        report_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"validation_{timestamp}.json"
        with open(report_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nReport written to: {report_file}")
    
    # Exit with appropriate code
    if results["summary"]["status"] == "error":
        sys.exit(1)
    elif results["summary"]["status"] == "warning":
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

