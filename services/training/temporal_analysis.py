"""Temporal pattern analysis for schema evolution.

This module provides functions to analyze schema changes over time
and learn temporal patterns from change history, integrating with:
- Extract service Postgres (updated_at_utc timestamps)
- Neo4j knowledge graph (node/edge properties with timestamps)
- Glean Catalog (exported_at from manifests)
- json_with_changes.json (historical change data)
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import statistics
import os

logger = logging.getLogger(__name__)


class SchemaEvolutionAnalyzer:
    """Analyzes schema evolution patterns from change history."""
    
    def __init__(self):
        self.change_patterns = []
        self.temporal_sequences = []
        self.evolution_statistics = {}
    
    def analyze_change_history(
        self,
        json_with_changes: Dict[str, Any],
        project_id: Optional[str] = None,
        system_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze schema change history from json_with_changes.json.
        
        Args:
            json_with_changes: Dictionary containing change history
            project_id: Optional project ID for context
            system_id: Optional system ID for context
        
        Returns:
            Dictionary with temporal analysis:
            - change_patterns: Patterns of changes over time
            - temporal_sequences: Sequences of changes
            - evolution_statistics: Statistics about evolution
        """
        logger.info("Analyzing schema evolution from change history")
        
        analysis = {
            "change_patterns": [],
            "temporal_sequences": [],
            "evolution_statistics": {},
            "project_id": project_id,
            "system_id": system_id,
        }
        
        # Parse change history structure
        # json_with_changes.json typically contains:
        # - Tables with columns
        # - Change history for each column
        # - Timestamps or version information
        
        changes = self._extract_changes(json_with_changes)
        
        if not changes:
            logger.warning("No changes found in change history")
            return analysis
        
        # Analyze change patterns
        change_patterns = self._analyze_change_patterns(changes)
        analysis["change_patterns"] = change_patterns
        
        # Extract temporal sequences
        temporal_sequences = self._extract_temporal_sequences(changes)
        analysis["temporal_sequences"] = temporal_sequences
        
        # Calculate evolution statistics
        evolution_stats = self._calculate_evolution_statistics(changes)
        analysis["evolution_statistics"] = evolution_stats
        
        logger.info(
            f"Analyzed {len(changes)} changes, found {len(change_patterns)} patterns, "
            f"{len(temporal_sequences)} temporal sequences"
        )
        
        return analysis
    
    def _extract_changes(self, json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract change records from JSON data."""
        changes = []
        
        # Handle different JSON structures
        if isinstance(json_data, dict):
            # Look for tables or change records
            if "tables" in json_data:
                tables = json_data["tables"]
            elif "data" in json_data:
                tables = json_data["data"]
            else:
                # Assume top-level keys are tables
                tables = json_data
            
            if isinstance(tables, dict):
                for table_name, table_data in tables.items():
                    if isinstance(table_data, dict):
                        # Extract column changes
                        if "columns" in table_data:
                            columns = table_data["columns"]
                        elif "schema" in table_data:
                            columns = table_data["schema"]
                        else:
                            columns = table_data
                        
                        if isinstance(columns, dict):
                            for col_name, col_data in columns.items():
                                if isinstance(col_data, dict):
                                    # Look for change indicators
                                    change = self._extract_column_change(
                                        table_name, col_name, col_data
                                    )
                                    if change:
                                        changes.append(change)
                        elif isinstance(columns, list):
                            for col_data in columns:
                                if isinstance(col_data, dict):
                                    col_name = col_data.get("name", col_data.get("column", ""))
                                    change = self._extract_column_change(
                                        table_name, col_name, col_data
                                    )
                                    if change:
                                        changes.append(change)
        
        return changes
    
    def _extract_column_change(
        self,
        table_name: str,
        col_name: str,
        col_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract a single column change record."""
        change = {
            "table": table_name,
            "column": col_name,
            "type": col_data.get("type", col_data.get("data_type", "")),
            "nullable": col_data.get("nullable", False),
            "timestamp": col_data.get("timestamp", col_data.get("changed_at", "")),
            "change_type": None,
        }
        
        # Detect change type
        if "added" in col_data or col_data.get("status") == "added":
            change["change_type"] = "added"
        elif "removed" in col_data or col_data.get("status") == "removed":
            change["change_type"] = "removed"
        elif "modified" in col_data or col_data.get("status") == "modified":
            change["change_type"] = "modified"
        elif "previous_type" in col_data:
            change["change_type"] = "type_changed"
            change["previous_type"] = col_data["previous_type"]
        else:
            # Assume it's a current state (not a change)
            change["change_type"] = "current"
        
        return change
    
    def _analyze_change_patterns(self, changes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze patterns in schema changes."""
        patterns = []
        
        # Pattern 1: Change type distribution
        change_type_counts = Counter()
        for change in changes:
            change_type = change.get("change_type", "unknown")
            change_type_counts[change_type] += 1
        
        patterns.append({
            "pattern_type": "change_type_distribution",
            "description": "Distribution of change types",
            "data": dict(change_type_counts),
        })
        
        # Pattern 2: Type transition patterns
        type_transitions = defaultdict(Counter)
        for change in changes:
            if change.get("change_type") == "type_changed":
                prev_type = change.get("previous_type", "")
                new_type = change.get("type", "")
                if prev_type and new_type:
                    type_transitions[prev_type][new_type] += 1
        
        if type_transitions:
            patterns.append({
                "pattern_type": "type_transition_patterns",
                "description": "Patterns of column type changes",
                "data": {k: dict(v) for k, v in type_transitions.items()},
            })
        
        # Pattern 3: Temporal clustering (changes grouped by time)
        temporal_clusters = self._cluster_by_time(changes)
        if temporal_clusters:
            patterns.append({
                "pattern_type": "temporal_clustering",
                "description": "Changes grouped by time periods",
                "data": temporal_clusters,
            })
        
        # Pattern 4: Table evolution patterns
        table_evolution = self._analyze_table_evolution(changes)
        if table_evolution:
            patterns.append({
                "pattern_type": "table_evolution_patterns",
                "description": "Patterns of table-level changes",
                "data": table_evolution,
            })
        
        return patterns
    
    def _cluster_by_time(self, changes: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Cluster changes by time periods."""
        clusters = defaultdict(list)
        
        for change in changes:
            timestamp = change.get("timestamp", "")
            if timestamp:
                # Extract date (YYYY-MM-DD) or use as-is
                date_key = timestamp[:10] if len(timestamp) >= 10 else timestamp
                clusters[date_key].append(change)
            else:
                clusters["unknown"].append(change)
        
        # Convert to list format for better serialization
        return {
            date: changes
            for date, changes in sorted(clusters.items())
            if date != "unknown" or len(changes) > 0
        }
    
    def _analyze_table_evolution(self, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze evolution patterns at table level."""
        table_changes = defaultdict(list)
        
        for change in changes:
            table_name = change.get("table", "")
            if table_name:
                table_changes[table_name].append(change)
        
        evolution_stats = {}
        for table_name, table_change_list in table_changes.items():
            change_types = [c.get("change_type") for c in table_change_list]
            evolution_stats[table_name] = {
                "total_changes": len(table_change_list),
                "change_types": dict(Counter(change_types)),
                "columns_affected": len(set(c.get("column") for c in table_change_list)),
            }
        
        return evolution_stats
    
    def _extract_temporal_sequences(self, changes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract temporal sequences of changes."""
        sequences = []
        
        # Group changes by table
        table_changes = defaultdict(list)
        for change in changes:
            table_name = change.get("table", "")
            if table_name:
                table_changes[table_name].append(change)
        
        # Build sequences for each table
        for table_name, table_change_list in table_changes.items():
            # Sort by timestamp if available
            sorted_changes = sorted(
                table_change_list,
                key=lambda x: x.get("timestamp", "") or ""
            )
            
            if len(sorted_changes) >= 2:
                sequence = {
                    "table": table_name,
                    "changes": sorted_changes,
                    "sequence_length": len(sorted_changes),
                    "time_span": self._calculate_time_span(sorted_changes),
                }
                sequences.append(sequence)
        
        return sequences
    
    def _calculate_time_span(self, changes: List[Dict[str, Any]]) -> Optional[str]:
        """Calculate time span of changes."""
        timestamps = [
            c.get("timestamp", "") for c in changes
            if c.get("timestamp", "")
        ]
        
        if len(timestamps) >= 2:
            try:
                # Try to parse timestamps
                dates = [
                    datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    for ts in timestamps
                    if ts
                ]
                if dates:
                    min_date = min(dates)
                    max_date = max(dates)
                    delta = max_date - min_date
                    return f"{delta.days} days"
            except Exception:
                pass
        
        return None
    
    def _calculate_evolution_statistics(self, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics about schema evolution."""
        stats = {
            "total_changes": len(changes),
            "change_types": {},
            "tables_affected": 0,
            "columns_affected": 0,
            "average_changes_per_table": 0,
        }
        
        if not changes:
            return stats
        
        # Count change types
        change_type_counts = Counter(c.get("change_type") for c in changes)
        stats["change_types"] = dict(change_type_counts)
        
        # Count unique tables and columns
        tables = set(c.get("table") for c in changes if c.get("table"))
        columns = set(
            (c.get("table"), c.get("column"))
            for c in changes
            if c.get("table") and c.get("column")
        )
        
        stats["tables_affected"] = len(tables)
        stats["columns_affected"] = len(columns)
        
        if len(tables) > 0:
            stats["average_changes_per_table"] = len(changes) / len(tables)
        
        # Calculate type distribution
        type_counts = Counter(c.get("type") for c in changes if c.get("type"))
        stats["type_distribution"] = dict(type_counts.most_common(20))
        
        return stats
    
    def predict_future_changes(
        self,
        current_schema: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Predict likely future schema changes based on learned patterns.
        
        Args:
            current_schema: Current schema state
            analysis: Previous temporal analysis results
        
        Returns:
            List of predicted changes with probabilities
        """
        predictions = []
        
        # Use learned patterns to predict
        change_patterns = analysis.get("change_patterns", [])
        type_transitions = None
        
        for pattern in change_patterns:
            if pattern.get("pattern_type") == "type_transition_patterns":
                type_transitions = pattern.get("data", {})
                break
        
        # Predict type changes for current columns
        if isinstance(current_schema, dict):
            for table_name, table_data in current_schema.items():
                if isinstance(table_data, dict):
                    columns = table_data.get("columns", table_data.get("schema", {}))
                    if isinstance(columns, dict):
                        for col_name, col_data in columns.items():
                            if isinstance(col_data, dict):
                                current_type = col_data.get("type", col_data.get("data_type", ""))
                                if current_type and type_transitions:
                                    # Predict likely transitions
                                    if current_type in type_transitions:
                                        transitions = type_transitions[current_type]
                                        total = sum(transitions.values())
                                        for target_type, count in transitions.items():
                                            probability = count / total if total > 0 else 0
                                            predictions.append({
                                                "table": table_name,
                                                "column": col_name,
                                                "predicted_change": "type_transition",
                                                "from_type": current_type,
                                                "to_type": target_type,
                                                "probability": probability,
                                            })
        
        # Sort by probability
        predictions.sort(key=lambda x: x.get("probability", 0), reverse=True)
        
        return predictions


class TemporalPatternLearner:
    """Learns temporal patterns from schema evolution and Glean historical data.
    
    Integrates with:
    - Extract service Postgres (updated_at_utc)
    - Neo4j knowledge graph (metrics_calculated_at, node/edge properties)
    - Glean Catalog (exported_at from export manifests)
    - json_with_changes.json (historical change data)
    """
    
    def __init__(self, extract_client=None, glean_client=None, postgres_dsn=None):
        """Initialize temporal pattern learner.
        
        Args:
            extract_client: Optional ExtractServiceClient for Neo4j queries
            glean_client: Optional GleanTrainingClient for Glean queries
            postgres_dsn: Optional Postgres DSN for direct Postgres queries
        """
        self.evolution_analyzer = SchemaEvolutionAnalyzer()
        self.temporal_patterns = {}
        self.extract_client = extract_client
        self.glean_client = glean_client
        self.postgres_dsn = postgres_dsn or os.getenv("POSTGRES_CATALOG_DSN")
    
    def learn_temporal_patterns(
        self,
        json_with_changes: Optional[Dict[str, Any]] = None,
        glean_metrics: Optional[Dict[str, Any]] = None,
        project_id: Optional[str] = None,
        system_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Learn temporal patterns from change history and Glean metrics.
        
        Integrates temporal data from multiple sources:
        - json_with_changes.json: Historical change data
        - Glean Catalog: Export manifests with exported_at
        - Postgres: Node/edge updated_at_utc timestamps
        - Neo4j: Node properties with metrics_calculated_at
        
        Args:
            json_with_changes: Change history from json_with_changes.json
            glean_metrics: Historical metrics from Glean
            project_id: Optional project ID
            system_id: Optional system ID
        
        Returns:
            Dictionary with temporal patterns:
            - evolution_patterns: Patterns from change history
            - temporal_metrics: Patterns from Glean metrics
            - postgres_temporal: Patterns from Postgres updated_at_utc
            - neo4j_temporal: Patterns from Neo4j timestamped properties
            - combined_insights: Combined insights from all sources
        """
        logger.info("Learning temporal patterns from all available sources")
        
        patterns = {
            "evolution_patterns": {},
            "temporal_metrics": {},
            "postgres_temporal": {},
            "neo4j_temporal": {},
            "combined_insights": {},
        }
        
        # Analyze change history if available
        if json_with_changes:
            evolution_analysis = self.evolution_analyzer.analyze_change_history(
                json_with_changes, project_id, system_id
            )
            patterns["evolution_patterns"] = evolution_analysis
        
        # Analyze temporal metrics from Glean
        if glean_metrics:
            temporal_metrics = self._analyze_temporal_metrics(glean_metrics)
            patterns["temporal_metrics"] = temporal_metrics
        
        # Query Postgres for temporal node/edge changes
        postgres_temporal = self._query_postgres_temporal(project_id, system_id)
        if postgres_temporal:
            patterns["postgres_temporal"] = postgres_temporal
            logger.info(f"Retrieved {len(postgres_temporal.get('node_changes', []))} node changes from Postgres")
        
        # Query Neo4j for temporal patterns
        neo4j_temporal = self._query_neo4j_temporal(project_id, system_id)
        if neo4j_temporal:
            patterns["neo4j_temporal"] = neo4j_temporal
            logger.info(f"Retrieved {len(neo4j_temporal.get('timestamped_nodes', []))} timestamped nodes from Neo4j")
        
        # Combine insights from all sources
        patterns["combined_insights"] = self._combine_insights(
            patterns["evolution_patterns"],
            patterns["temporal_metrics"],
            patterns["postgres_temporal"],
            patterns["neo4j_temporal"]
        )
        
        self.temporal_patterns = patterns
        
        logger.info("Temporal pattern learning completed")
        
        return patterns
    
    def _analyze_temporal_metrics(self, glean_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal patterns from Glean metrics."""
        analysis = {}
        
        # Analyze entropy trends
        entropy_trend = glean_metrics.get("metadata_entropy_trend", [])
        if entropy_trend:
            analysis["entropy_trend"] = self._analyze_trend_direction(entropy_trend)
        
        # Analyze KL divergence trends
        kl_trend = glean_metrics.get("kl_divergence_trend", [])
        if kl_trend:
            analysis["kl_divergence_trend"] = self._analyze_trend_direction(kl_trend)
        
        # Analyze column count trends
        column_trend = glean_metrics.get("column_count_trend", [])
        if column_trend:
            analysis["column_count_trend"] = self._analyze_trend_direction(column_trend)
        
        # Find correlations between metrics and time
        analysis["metric_correlations"] = self._find_metric_correlations(
            entropy_trend, kl_trend, column_trend
        )
        
        return analysis
    
    def _analyze_trend_direction(self, trend: List[Tuple[str, float]]) -> Dict[str, Any]:
        """Analyze the direction and magnitude of a trend."""
        if len(trend) < 2:
            return {"direction": "insufficient_data"}
        
        values = [v for _, v in trend]
        
        # Calculate trend
        first_half_avg = statistics.mean(values[:len(values)//2])
        second_half_avg = statistics.mean(values[len(values)//2:])
        
        change = second_half_avg - first_half_avg
        change_percent = (change / first_half_avg * 100) if first_half_avg > 0 else 0
        
        direction = "increasing" if change > 0 else "decreasing" if change < 0 else "stable"
        
        return {
            "direction": direction,
            "magnitude": abs(change),
            "change_percent": change_percent,
            "first_half_avg": first_half_avg,
            "second_half_avg": second_half_avg,
        }
    
    def _find_metric_correlations(
        self,
        entropy_trend: List[Tuple[str, float]],
        kl_trend: List[Tuple[str, float]],
        column_trend: List[Tuple[str, float]]
    ) -> Dict[str, Any]:
        """Find correlations between different metrics over time."""
        correlations = {}
        
        # Correlate entropy with column count
        if entropy_trend and column_trend:
            entropy_values = [v for _, v in entropy_trend]
            column_values = [v for _, v in column_trend]
            
            if len(entropy_values) == len(column_values) and len(entropy_values) > 1:
                try:
                    correlation = self._calculate_correlation(entropy_values, column_values)
                    correlations["entropy_column_count"] = correlation
                except Exception:
                    pass
        
        return correlations
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)
        
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
        denom_x = sum((v - mean_x) ** 2 for v in x)
        denom_y = sum((v - mean_y) ** 2 for v in y)
        
        if denom_x > 0 and denom_y > 0:
            return numerator / (denom_x ** 0.5 * denom_y ** 0.5)
        
        return 0.0
    
    def _query_postgres_temporal(
        self,
        project_id: Optional[str],
        system_id: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Query Postgres for temporal node/edge changes using updated_at_utc.
        
        Returns:
            Dictionary with temporal change data from Postgres
        """
        if not self.postgres_dsn:
            return None
        
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            
            conn = psycopg2.connect(self.postgres_dsn)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Query nodes with temporal changes (grouped by date)
            cursor.execute("""
                SELECT 
                    DATE(updated_at_utc) as change_date,
                    COUNT(*) as node_count,
                    COUNT(DISTINCT kind) as unique_kinds,
                    COUNT(DISTINCT label) as unique_labels
                FROM glean_nodes
                WHERE updated_at_utc >= NOW() - INTERVAL '30 days'
                GROUP BY DATE(updated_at_utc)
                ORDER BY change_date DESC
            """)
            
            node_changes = []
            for row in cursor.fetchall():
                node_changes.append({
                    "date": str(row["change_date"]),
                    "node_count": row["node_count"],
                    "unique_kinds": row["unique_kinds"],
                    "unique_labels": row["unique_labels"],
                })
            
            # Query edges with temporal changes
            cursor.execute("""
                SELECT 
                    DATE(updated_at_utc) as change_date,
                    COUNT(*) as edge_count,
                    COUNT(DISTINCT label) as unique_labels
                FROM glean_edges
                WHERE updated_at_utc >= NOW() - INTERVAL '30 days'
                GROUP BY DATE(updated_at_utc)
                ORDER BY change_date DESC
            """)
            
            edge_changes = []
            for row in cursor.fetchall():
                edge_changes.append({
                    "date": str(row["change_date"]),
                    "edge_count": row["edge_count"],
                    "unique_labels": row["unique_labels"],
                })
            
            cursor.close()
            conn.close()
            
            return {
                "node_changes": node_changes,
                "edge_changes": edge_changes,
                "source": "postgres",
            }
        except ImportError:
            logger.warning("psycopg2 not available, skipping Postgres temporal queries")
            return None
        except Exception as e:
            logger.warning(f"Postgres temporal query failed: {e}")
            return None
    
    def _query_neo4j_temporal(
        self,
        project_id: Optional[str],
        system_id: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Query Neo4j for temporal patterns from node properties.
        
        Queries nodes with metrics_calculated_at and other timestamp properties.
        
        Returns:
            Dictionary with temporal data from Neo4j
        """
        if not self.extract_client:
            return None
        
        try:
            # Query nodes with metrics timestamps
            query = """
            MATCH (n:Node)
            WHERE n.properties_json IS NOT NULL
            AND n.properties_json CONTAINS 'metrics_calculated_at'
            RETURN n.id as id, n.type as type, n.label as label, 
                   n.properties_json as properties_json
            LIMIT 100
            """
            
            result = self.extract_client.query_knowledge_graph(query)
            
            timestamped_nodes = []
            for row in result.get("data", []):
                try:
                    props_json = json.loads(row.get("properties_json", "{}"))
                    metrics_calculated_at = props_json.get("metrics_calculated_at")
                    if metrics_calculated_at:
                        timestamped_nodes.append({
                            "id": row.get("id"),
                            "type": row.get("type"),
                            "label": row.get("label"),
                            "metrics_calculated_at": metrics_calculated_at,
                            "metadata_entropy": props_json.get("metadata_entropy"),
                            "kl_divergence": props_json.get("kl_divergence"),
                        })
                except (json.JSONDecodeError, KeyError):
                    continue
            
            # Query for temporal patterns in relationships
            edge_query = """
            MATCH (source:Node)-[r:RELATIONSHIP]->(target:Node)
            WHERE r.properties_json IS NOT NULL
            RETURN source.id as source_id, target.id as target_id, 
                   r.label as label, r.properties_json as properties_json
            LIMIT 50
            """
            
            edge_result = self.extract_client.query_knowledge_graph(edge_query)
            
            timestamped_edges = []
            for row in edge_result.get("data", []):
                timestamped_edges.append({
                    "source_id": row.get("source_id"),
                    "target_id": row.get("target_id"),
                    "label": row.get("label"),
                })
            
            return {
                "timestamped_nodes": timestamped_nodes,
                "timestamped_edges": timestamped_edges,
                "source": "neo4j",
            }
        except Exception as e:
            logger.warning(f"Neo4j temporal query failed: {e}")
            return None
    
    def _combine_insights(
        self,
        evolution_patterns: Dict[str, Any],
        temporal_metrics: Dict[str, Any],
        postgres_temporal: Optional[Dict[str, Any]] = None,
        neo4j_temporal: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Combine insights from all temporal data sources."""
        insights = {
            "schema_evolution_rate": None,
            "quality_trends": {},
            "predicted_changes": [],
            "data_sources": [],
        }
        
        # Calculate schema evolution rate
        if evolution_patterns:
            stats = evolution_patterns.get("evolution_statistics", {})
            total_changes = stats.get("total_changes", 0)
            tables_affected = stats.get("tables_affected", 0)
            
            if tables_affected > 0:
                insights["schema_evolution_rate"] = total_changes / tables_affected
            insights["data_sources"].append("json_with_changes")
        
        # Extract quality trends from metrics
        if temporal_metrics:
            entropy_trend = temporal_metrics.get("entropy_trend", {})
            kl_trend = temporal_metrics.get("kl_divergence_trend", {})
            
            insights["quality_trends"] = {
                "entropy": entropy_trend.get("direction", "unknown"),
                "kl_divergence": kl_trend.get("direction", "unknown"),
            }
            insights["data_sources"].append("glean")
        
        # Add Postgres temporal insights
        if postgres_temporal:
            node_changes = postgres_temporal.get("node_changes", [])
            if node_changes:
                total_node_changes = sum(c.get("node_count", 0) for c in node_changes)
                insights["postgres_node_changes"] = total_node_changes
                insights["postgres_change_days"] = len(node_changes)
            insights["data_sources"].append("postgres")
        
        # Add Neo4j temporal insights
        if neo4j_temporal:
            timestamped_nodes = neo4j_temporal.get("timestamped_nodes", [])
            if timestamped_nodes:
                insights["neo4j_timestamped_nodes"] = len(timestamped_nodes)
                # Extract timestamps and calculate metrics
                timestamps = [
                    n.get("metrics_calculated_at") for n in timestamped_nodes
                    if n.get("metrics_calculated_at")
                ]
                if timestamps:
                    insights["neo4j_metrics_snapshots"] = len(timestamps)
            insights["data_sources"].append("neo4j")
        
        return insights
    
    def get_patterns(self) -> Dict[str, Any]:
        """Get learned temporal patterns."""
        return self.temporal_patterns

