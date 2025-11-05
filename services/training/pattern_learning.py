"""Pattern learning algorithms for training pipeline.

This module provides algorithms to learn patterns from knowledge graphs,
Glean Catalog data, and source data to enable predictive modeling.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, Counter
from datetime import datetime
import statistics

logger = logging.getLogger(__name__)


class ColumnTypePatternLearner:
    """Learns patterns in column type distributions and transitions."""
    
    def __init__(self):
        self.type_transitions = defaultdict(Counter)  # from_type -> {to_type: count}
        self.type_frequencies = Counter()  # type -> frequency
        self.type_contexts = defaultdict(list)  # type -> [context features]
        self.learned_patterns = {}
    
    def learn_from_graph(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Learn column type patterns from knowledge graph nodes.
        
        Args:
            nodes: List of graph nodes (from knowledge graph or Glean)
        
        Returns:
            Dictionary with learned patterns:
            - type_distributions: Distribution of column types
            - type_transitions: Patterns of type changes
            - common_patterns: Most common type sequences
        """
        logger.info(f"Learning column type patterns from {len(nodes)} nodes")
        
        # Extract column information from nodes
        columns = []
        for node in nodes:
            key = node.get("key", {}) if isinstance(node.get("key"), dict) else {}
            props = key.get("properties_json", {})
            
            if isinstance(props, str):
                try:
                    props = json.loads(props)
                except json.JSONDecodeError:
                    props = {}
            
            node_type = key.get("kind", "")
            if node_type == "column" and isinstance(props, dict):
                column_type = props.get("type", props.get("data_type", "unknown"))
                table_name = props.get("table_name", "")
                column_name = props.get("name", "")
                
                if column_type and column_type != "unknown":
                    columns.append({
                        "type": column_type,
                        "table": table_name,
                        "column": column_name,
                        "nullable": props.get("nullable", False),
                        "props": props,
                    })
                    self.type_frequencies[column_type] += 1
        
        # Analyze type distributions
        type_distribution = dict(self.type_frequencies.most_common())
        
        # Learn type transitions from temporal data (if available)
        # Group by table to find type changes over time
        table_columns = defaultdict(list)
        for col in columns:
            if col["table"]:
                table_columns[col["table"]].append(col)
        
        # Analyze transitions (simplified - would need temporal data for real transitions)
        transitions = self._analyze_type_transitions(columns)
        
        # Identify common patterns
        common_patterns = self._identify_common_patterns(columns, type_distribution)
        
        self.learned_patterns = {
            "type_distributions": type_distribution,
            "type_transitions": transitions,
            "common_patterns": common_patterns,
            "total_columns": len(columns),
            "unique_types": len(type_distribution),
        }
        
        logger.info(f"Learned patterns: {len(type_distribution)} unique types, {len(transitions)} transition patterns")
        
        return self.learned_patterns
    
    def _analyze_type_transitions(self, columns: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
        """Analyze type transition patterns.
        
        This is a simplified version. In production, would analyze temporal changes
        from Glean historical data or change logs.
        """
        transitions = defaultdict(Counter)
        
        # Group by table to find similar patterns
        table_types = defaultdict(set)
        for col in columns:
            if col["table"]:
                table_types[col["table"]].add(col["type"])
        
        # Find common type co-occurrences (proxy for transitions)
        for table, types in table_types.items():
            type_list = sorted(types)
            for i in range(len(type_list) - 1):
                from_type = type_list[i]
                to_type = type_list[i + 1]
                transitions[from_type][to_type] += 1
        
        return {k: dict(v) for k, v in transitions.items()}
    
    def _identify_common_patterns(
        self,
        columns: List[Dict[str, Any]],
        type_distribution: Dict[str, int]
    ) -> List[Dict[str, Any]]:
        """Identify common column type patterns."""
        patterns = []
        
        # Pattern 1: Most common types
        most_common = sorted(type_distribution.items(), key=lambda x: x[1], reverse=True)[:10]
        patterns.append({
            "pattern_type": "most_common_types",
            "description": "Most frequently occurring column types",
            "data": dict(most_common),
        })
        
        # Pattern 2: Type with nullable
        nullable_by_type = defaultdict(int)
        for col in columns:
            if col.get("nullable"):
                nullable_by_type[col["type"]] += 1
        
        patterns.append({
            "pattern_type": "nullable_distribution",
            "description": "Distribution of nullable columns by type",
            "data": dict(nullable_by_type),
        })
        
        # Pattern 3: Type sequences in tables (common type combinations)
        table_type_sequences = defaultdict(list)
        for col in columns:
            if col["table"]:
                table_type_sequences[col["table"]].append(col["type"])
        
        # Find most common type sequences
        sequence_counts = Counter()
        for table, types in table_type_sequences.items():
            if len(types) >= 3:  # Only consider tables with multiple columns
                # Create signature from sorted unique types
                signature = tuple(sorted(set(types)))
                sequence_counts[signature] += 1
        
        common_sequences = sequence_counts.most_common(10)
        patterns.append({
            "pattern_type": "common_type_sequences",
            "description": "Most common column type combinations in tables",
            "data": {str(k): v for k, v in common_sequences},
        })
        
        return patterns
    
    def predict_type_transition(self, from_type: str, context: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float]]:
        """Predict likely type transitions from a given type.
        
        Args:
            from_type: Source column type
            context: Optional context (table name, other columns, etc.)
        
        Returns:
            List of (to_type, probability) tuples, sorted by probability
        """
        if from_type not in self.type_transitions:
            return []
        
        transitions = self.type_transitions[from_type]
        total = sum(transitions.values())
        
        if total == 0:
            return []
        
        predictions = [
            (to_type, count / total)
            for to_type, count in transitions.items()
        ]
        
        return sorted(predictions, key=lambda x: x[1], reverse=True)
    
    def get_patterns(self) -> Dict[str, Any]:
        """Get learned patterns."""
        return self.learned_patterns


class RelationshipPatternLearner:
    """Learns patterns in table relationships and data flow."""
    
    def __init__(self):
        self.edge_patterns = defaultdict(Counter)  # edge_label -> {target_type: count}
        self.relationship_chains = []  # List of relationship sequences
        self.path_patterns = defaultdict(int)  # path_signature -> count
        self.learned_patterns = {}
    
    def learn_from_graph(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Learn relationship patterns from knowledge graph.
        
        Args:
            nodes: List of graph nodes
            edges: List of graph edges
        
        Returns:
            Dictionary with learned patterns:
            - edge_label_distributions: Distribution of edge types
            - relationship_chains: Common relationship sequences
            - path_patterns: Common paths through the graph
        """
        logger.info(f"Learning relationship patterns from {len(nodes)} nodes, {len(edges)} edges")
        
        # Build node lookup
        node_map = {}
        for node in nodes:
            key = node.get("key", {}) if isinstance(node.get("key"), dict) else {}
            node_id = key.get("id", "")
            node_type = key.get("kind", "")
            if node_id:
                node_map[node_id] = {"type": node_type, "id": node_id}
        
        # Analyze edge patterns
        edge_label_counts = Counter()
        edge_type_targets = defaultdict(Counter)
        
        for edge in edges:
            key = edge.get("key", {}) if isinstance(edge.get("key"), dict) else {}
            label = key.get("label", "unknown")
            source_id = key.get("source_id", "")
            target_id = key.get("target_id", "")
            
            edge_label_counts[label] += 1
            
            if target_id in node_map:
                target_type = node_map[target_id]["type"]
                edge_type_targets[label][target_type] += 1
        
        # Find relationship chains
        chains = self._find_relationship_chains(nodes, edges, node_map)
        
        # Find common paths
        paths = self._find_common_paths(nodes, edges, node_map)
        
        self.learned_patterns = {
            "edge_label_distributions": dict(edge_label_counts),
            "edge_type_targets": {k: dict(v) for k, v in edge_type_targets.items()},
            "relationship_chains": chains,
            "path_patterns": paths,
            "total_edges": len(edges),
            "unique_labels": len(edge_label_counts),
        }
        
        logger.info(f"Learned patterns: {len(edge_label_counts)} unique edge types, {len(chains)} relationship chains")
        
        return self.learned_patterns
    
    def _find_relationship_chains(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        node_map: Dict[str, Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """Find common relationship chains (e.g., table -> column -> column)."""
        # Build adjacency list
        adjacency = defaultdict(list)
        for edge in edges:
            key = edge.get("key", {}) if isinstance(edge.get("key"), dict) else {}
            source_id = key.get("source_id", "")
            target_id = key.get("target_id", "")
            label = key.get("label", "")
            
            if source_id and target_id:
                adjacency[source_id].append((target_id, label))
        
        # Find chains of length 2-3
        chains = []
        chain_counts = Counter()
        
        for node_id in node_map:
            if node_id in adjacency:
                # Find 2-hop and 3-hop paths
                for target_id, label1 in adjacency[node_id]:
                    if target_id in adjacency:
                        for target2_id, label2 in adjacency[target_id]:
                            chain = (node_map[node_id]["type"], label1, node_map[target_id]["type"], label2, node_map[target2_id]["type"])
                            chain_counts[chain] += 1
        
        # Get most common chains
        common_chains = chain_counts.most_common(20)
        chains = [
            {
                "chain": list(chain),
                "count": count,
                "description": f"{chain[0]} --{chain[1]}--> {chain[2]} --{chain[3]}--> {chain[4]}",
            }
            for chain, count in common_chains
        ]
        
        return chains
    
    def _find_common_paths(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        node_map: Dict[str, Dict[str, str]]
    ) -> Dict[str, int]:
        """Find common path patterns through the graph."""
        # Build adjacency list
        adjacency = defaultdict(list)
        for edge in edges:
            key = edge.get("key", {}) if isinstance(edge.get("key"), dict) else {}
            source_id = key.get("source_id", "")
            target_id = key.get("target_id", "")
            label = key.get("label", "")
            
            if source_id and target_id:
                adjacency[source_id].append((target_id, label))
        
        # Find paths from root nodes (tables, systems) to leaves
        paths = []
        for node_id in node_map:
            node_type = node_map[node_id]["type"]
            # Start from tables, systems, projects
            if node_type in ["table", "system", "project", "information-system"]:
                path = self._find_paths_from_node(node_id, adjacency, node_map, max_depth=3)
                paths.extend(path)
        
        # Count path patterns
        path_patterns = Counter()
        for path in paths:
            if len(path) >= 2:
                # Create signature from node types
                signature = " -> ".join(path)
                path_patterns[signature] += 1
        
        return dict(path_patterns.most_common(30))
    
    def _find_paths_from_node(
        self,
        node_id: str,
        adjacency: Dict[str, List[Tuple[str, str]]],
        node_map: Dict[str, Dict[str, str]],
        max_depth: int = 3,
        current_depth: int = 0,
        visited: Optional[set] = None
    ) -> List[List[str]]:
        """Find paths from a node using DFS."""
        if visited is None:
            visited = set()
        
        if current_depth >= max_depth or node_id in visited:
            return []
        
        if node_id not in node_map:
            return []
        
        visited.add(node_id)
        paths = []
        node_type = node_map[node_id]["type"]
        
        if node_id in adjacency:
            for target_id, label in adjacency[node_id]:
                if target_id not in visited:
                    sub_paths = self._find_paths_from_node(
                        target_id, adjacency, node_map, max_depth, current_depth + 1, visited.copy()
                    )
                    if sub_paths:
                        for sub_path in sub_paths:
                            paths.append([node_type] + sub_path)
                    else:
                        # Leaf node
                        if target_id in node_map:
                            paths.append([node_type, node_map[target_id]["type"]])
        
        visited.remove(node_id)
        
        return paths if paths else [[node_type]]
    
    def predict_relationship(self, source_type: str, target_type: str) -> List[Tuple[str, float]]:
        """Predict likely relationship types between source and target.
        
        Args:
            source_type: Source node type
            target_type: Target node type
        
        Returns:
            List of (edge_label, probability) tuples
        """
        # Find edges that connect source_type to target_type
        predictions = []
        
        for edge_label, target_counts in self.edge_patterns.items():
            if target_type in target_counts:
                total = sum(target_counts.values())
                if total > 0:
                    probability = target_counts[target_type] / total
                    predictions.append((edge_label, probability))
        
        return sorted(predictions, key=lambda x: x[1], reverse=True)
    
    def get_patterns(self) -> Dict[str, Any]:
        """Get learned patterns."""
        return self.learned_patterns


class MetadataEntropyPatternLearner:
    """Learns patterns in metadata entropy and information theory metrics."""
    
    def __init__(self):
        self.entropy_patterns = []
        self.kl_divergence_patterns = []
        self.quality_trends = []
        self.learned_patterns = {}
    
    def learn_from_metrics(
        self,
        metrics_data: Dict[str, Any],
        glean_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Learn patterns from information theory metrics.
        
        Args:
            metrics_data: Current metrics from knowledge graph
            glean_metrics: Historical metrics from Glean (optional)
        
        Returns:
            Dictionary with learned patterns:
            - entropy_trends: Patterns in metadata entropy over time
            - kl_divergence_trends: Patterns in KL divergence
            - quality_correlations: Correlations between metrics and quality
        """
        logger.info("Learning metadata entropy patterns")
        
        patterns = {}
        
        # Analyze current metrics
        if metrics_data:
            entropy = metrics_data.get("metadata_entropy")
            kl_div = metrics_data.get("kl_divergence")
            column_count = metrics_data.get("column_count")
            
            if entropy is not None:
                patterns["current_entropy"] = entropy
            if kl_div is not None:
                patterns["current_kl_divergence"] = kl_div
            if column_count is not None:
                patterns["current_column_count"] = column_count
        
        # Analyze historical trends from Glean
        if glean_metrics and glean_metrics.get("metadata_entropy_trend"):
            entropy_trend = glean_metrics["metadata_entropy_trend"]
            patterns["entropy_trend"] = self._analyze_trend(entropy_trend, "entropy")
            patterns["entropy_statistics"] = self._calculate_statistics([v for _, v in entropy_trend])
        
        if glean_metrics and glean_metrics.get("kl_divergence_trend"):
            kl_trend = glean_metrics["kl_divergence_trend"]
            patterns["kl_divergence_trend"] = self._analyze_trend(kl_trend, "kl_divergence")
            patterns["kl_divergence_statistics"] = self._calculate_statistics([v for _, v in kl_trend])
        
        # Identify quality patterns
        if glean_metrics:
            patterns["quality_patterns"] = self._identify_quality_patterns(glean_metrics)
        
        self.learned_patterns = patterns
        
        logger.info(f"Learned {len(patterns)} metadata entropy patterns")
        
        return self.learned_patterns
    
    def _analyze_trend(self, trend: List[Tuple[str, float]], metric_name: str) -> Dict[str, Any]:
        """Analyze temporal trend in a metric."""
        if not trend:
            return {}
        
        values = [v for _, v in trend]
        dates = [d for d, _ in trend]
        
        # Calculate trend direction
        if len(values) >= 2:
            first_half = values[:len(values)//2]
            second_half = values[len(values)//2:]
            avg_first = statistics.mean(first_half) if first_half else 0
            avg_second = statistics.mean(second_half) if second_half else 0
            
            trend_direction = "increasing" if avg_second > avg_first else "decreasing" if avg_second < avg_first else "stable"
            trend_magnitude = abs(avg_second - avg_first) / avg_first if avg_first > 0 else 0
        else:
            trend_direction = "insufficient_data"
            trend_magnitude = 0
        
        return {
            "direction": trend_direction,
            "magnitude": trend_magnitude,
            "values": values,
            "dates": dates,
            "statistics": self._calculate_statistics(values),
        }
    
    def _calculate_statistics(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistics for a list of values."""
        if not values:
            return {}
        
        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
        }
    
    def _identify_quality_patterns(self, glean_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Identify patterns that correlate with data quality."""
        patterns = {}
        
        entropy_trend = glean_metrics.get("metadata_entropy_trend", [])
        kl_trend = glean_metrics.get("kl_divergence_trend", [])
        
        if entropy_trend and kl_trend:
            # Find correlation between entropy and KL divergence
            entropy_values = [v for _, v in entropy_trend]
            kl_values = [v for _, v in kl_trend]
            
            if len(entropy_values) == len(kl_values) and len(entropy_values) > 1:
                # Calculate correlation (Pearson correlation)
                try:
                    # Manual Pearson correlation calculation
                    mean_entropy = statistics.mean(entropy_values)
                    mean_kl = statistics.mean(kl_values)
                    
                    numerator = sum((entropy_values[i] - mean_entropy) * (kl_values[i] - mean_kl) for i in range(len(entropy_values)))
                    denom_entropy = sum((v - mean_entropy) ** 2 for v in entropy_values)
                    denom_kl = sum((v - mean_kl) ** 2 for v in kl_values)
                    
                    if denom_entropy > 0 and denom_kl > 0:
                        correlation = numerator / (denom_entropy ** 0.5 * denom_kl ** 0.5)
                        patterns["entropy_kl_correlation"] = correlation
                    
                    if correlation < -0.5:
                        patterns["quality_insight"] = "High entropy correlates with low KL divergence (good quality)"
                    elif correlation > 0.5:
                        patterns["quality_insight"] = "High entropy correlates with high KL divergence (potential issues)"
                    else:
                        patterns["quality_insight"] = "Entropy and KL divergence are relatively independent"
                except Exception:
                    pass
        
        return patterns
    
    def get_patterns(self) -> Dict[str, Any]:
        """Get learned patterns."""
        return self.learned_patterns


class WorkflowPatternLearner:
    """Learns patterns from Petri nets and workflow structures."""
    
    def __init__(self):
        self.workflow_patterns = defaultdict(Counter)
        self.job_dependencies = defaultdict(list)
        self.sql_patterns = defaultdict(Counter)
        self.learned_patterns = {}
    
    def learn_from_petri_net(
        self,
        petri_net: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Learn workflow patterns from a Petri net structure.
        
        Args:
            petri_net: Petri net structure from catalog or knowledge graph
        
        Returns:
            Dictionary with learned patterns:
            - workflow_patterns: Common workflow structures
            - job_dependencies: Job dependency patterns
            - sql_patterns: SQL query patterns in workflows
        """
        logger.info(f"Learning workflow patterns from Petri net: {petri_net.get('id', 'unknown')}")
        
        transitions = petri_net.get("transitions", [])
        arcs = petri_net.get("arcs", [])
        places = petri_net.get("places", [])
        
        # Learn job dependency patterns
        for arc in arcs:
            source = arc.get("source", "")
            target = arc.get("target", "")
            arc_type = arc.get("type", "")
            
            if arc_type == "place_to_transition":
                # Find the transition this arc points to
                for transition in transitions:
                    if transition.get("id") == target:
                        job_name = transition.get("label", "")
                        if job_name:
                            self.job_dependencies[job_name].append(source)
        
        # Learn SQL patterns from subprocesses
        for transition in transitions:
            subprocesses = transition.get("sub_processes", [])
            for subprocess in subprocesses:
                if subprocess.get("type") == "sql":
                    sql_content = subprocess.get("content", "")
                    if sql_content:
                        # Extract SQL operation type
                        sql_upper = sql_content.upper()
                        if "SELECT" in sql_upper:
                            self.sql_patterns["SELECT"] += 1
                        if "INSERT" in sql_upper:
                            self.sql_patterns["INSERT"] += 1
                        if "UPDATE" in sql_upper:
                            self.sql_patterns["UPDATE"] += 1
                        if "DELETE" in sql_upper:
                            self.sql_patterns["DELETE"] += 1
        
        # Learn workflow patterns (place -> transition -> place)
        workflow_sequences = []
        for arc in arcs:
            if arc.get("type") == "place_to_transition":
                source_place = arc.get("source", "")
                target_transition = arc.get("target", "")
                
                # Find follow-up arcs
                for next_arc in arcs:
                    if next_arc.get("type") == "transition_to_place" and next_arc.get("source") == target_transition:
                        target_place = next_arc.get("target", "")
                        sequence = f"{source_place} -> {target_transition} -> {target_place}"
                        workflow_sequences.append(sequence)
                        self.workflow_patterns[sequence] += 1
        
        self.learned_patterns = {
            "workflow_patterns": dict(self.workflow_patterns.most_common(20)),
            "job_dependencies": {k: v for k, v in self.job_dependencies.items()},
            "sql_patterns": dict(self.sql_patterns),
            "workflow_sequences": workflow_sequences[:20],
            "total_transitions": len(transitions),
            "total_arcs": len(arcs),
        }
        
        logger.info(f"Learned workflow patterns: {len(self.workflow_patterns)} patterns, {len(self.job_dependencies)} job dependencies")
        
        return self.learned_patterns


class SemanticPatternLearner:
    """Learns patterns from semantic embeddings."""
    
    def __init__(self):
        self.semantic_patterns = {}
    
    def learn_from_semantic_embeddings(
        self,
        semantic_embeddings: Dict[str, Any],
        graph_nodes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Learn patterns from semantic embeddings.
        
        Args:
            semantic_embeddings: Dictionary of artifact IDs to embedding metadata
            graph_nodes: List of graph nodes
        
        Returns:
            Dictionary of learned semantic patterns
        """
        patterns = {
            "embedding_clusters": {},
            "classification_patterns": {},
            "semantic_similarity_groups": {},
        }
        
        # Group by classification
        classification_groups = {}
        for artifact_id, embedding_data in semantic_embeddings.items():
            metadata = embedding_data.get("metadata", {})
            classification = metadata.get("table_classification")
            if classification:
                if classification not in classification_groups:
                    classification_groups[classification] = []
                classification_groups[classification].append(artifact_id)
        
        patterns["classification_patterns"] = {
            cls: len(ids) for cls, ids in classification_groups.items()
        }
        
        # Analyze semantic similarity (if scores available)
        score_ranges = {"high": [], "medium": [], "low": []}
        for artifact_id, embedding_data in semantic_embeddings.items():
            score = embedding_data.get("search_score", 0.0)
            if score >= 0.8:
                score_ranges["high"].append(artifact_id)
            elif score >= 0.5:
                score_ranges["medium"].append(artifact_id)
            else:
                score_ranges["low"].append(artifact_id)
        
        patterns["semantic_similarity_groups"] = {
            range_name: len(ids) for range_name, ids in score_ranges.items()
        }
        
        return patterns


class PatternLearningEngine:
    """Main engine for learning patterns from knowledge graphs and Glean data."""
    
    def __init__(self):
        self.column_learner = ColumnTypePatternLearner()
        self.relationship_learner = RelationshipPatternLearner()
        self.metrics_learner = MetadataEntropyPatternLearner()
        self.all_patterns = {}
    
    def learn_patterns(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        metrics: Optional[Dict[str, Any]] = None,
        glean_data: Optional[Dict[str, Any]] = None,
        semantic_embeddings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Learn all patterns from knowledge graph and Glean data.
        
        Args:
            nodes: List of graph nodes
            edges: List of graph edges
            metrics: Current information theory metrics
            glean_data: Historical data from Glean (optional)
        
        Returns:
            Dictionary with all learned patterns:
            - column_patterns: Column type patterns
            - relationship_patterns: Relationship patterns
            - metrics_patterns: Metadata entropy patterns
            - summary: Summary of learned patterns
        """
        logger.info("Starting pattern learning from knowledge graph and Glean data")
        
        # Learn column type patterns
        column_patterns = self.column_learner.learn_from_graph(nodes)
        
        # Learn relationship patterns
        relationship_patterns = self.relationship_learner.learn_from_graph(nodes, edges)
        
        # Learn metrics patterns
        glean_metrics = glean_data.get("metrics", {}) if glean_data else None
        metrics_patterns = self.metrics_learner.learn_from_metrics(metrics, glean_metrics)
        
        # Combine all patterns
        self.all_patterns = {
            "column_patterns": column_patterns,
            "relationship_patterns": relationship_patterns,
            "metrics_patterns": metrics_patterns,
            "summary": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "unique_column_types": column_patterns.get("unique_types", 0),
                "unique_edge_labels": relationship_patterns.get("unique_labels", 0),
                "has_historical_data": glean_data is not None,
                "learned_at": datetime.now().isoformat(),
            },
        }
        
        logger.info(f"Pattern learning completed: {len(self.all_patterns)} pattern categories")
        
        return self.all_patterns
    
    def get_patterns(self) -> Dict[str, Any]:
        """Get all learned patterns."""
        return self.all_patterns
    
    def predict_schema_evolution(
        self,
        current_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        """Predict likely schema evolution patterns.
        
        Args:
            current_type: Current column type
            context: Optional context (table, other columns, etc.)
        
        Returns:
            List of (predicted_type, probability) tuples
        """
        return self.column_learner.predict_type_transition(current_type, context)
    
    def predict_relationship_type(
        self,
        source_type: str,
        target_type: str
    ) -> List[Tuple[str, float]]:
        """Predict likely relationship type between nodes.
        
        Args:
            source_type: Source node type
            target_type: Target node type
        
        Returns:
            List of (edge_label, probability) tuples
        """
        return self.relationship_learner.predict_relationship(source_type, target_type)

