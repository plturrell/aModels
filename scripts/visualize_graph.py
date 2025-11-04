#!/usr/bin/env python3
"""
Visualize Neo4j graph using networkx and matplotlib
"""
import networkx as nx
import matplotlib.pyplot as plt
from neo4j import GraphDatabase
import os
import sys

# Neo4j connection
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "amodels123")

def visualize_graph(limit=100, output_file="graph_visualization.png"):
    """Visualize a subset of the graph."""
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        with driver.session() as session:
            # Fetch nodes and relationships
            result = session.run("""
                MATCH (n:Node)-[r:RELATIONSHIP]->(m:Node)
                RETURN n.id AS source, r.label AS rel, m.id AS target, 
                       n.type AS sourceType, m.type AS targetType
                LIMIT $limit
            """, limit=limit)
            
            # Build networkx graph
            G = nx.DiGraph()
            edges = []
            for record in result:
                source = record["source"]
                target = record["target"]
                rel = record["rel"]
                source_type = record["sourceType"]
                target_type = record["targetType"]
                
                G.add_node(source, type=source_type)
                G.add_node(target, type=target_type)
                G.add_edge(source, target, label=rel)
                edges.append((source, target, rel))
        
        driver.close()
        
        if len(G.nodes()) == 0:
            print("No graph data found. Make sure data has been loaded into Neo4j.")
            return
        
        print(f"Visualizing {len(G.nodes())} nodes and {len(G.edges())} edges...")
        
        # Visualize
        plt.figure(figsize=(20, 20))
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Color nodes by type
        node_colors = []
        for node in G.nodes():
            node_type = G.nodes[node].get("type", "unknown")
            node_type_str = str(node_type).lower()
            if "table" in node_type_str:
                node_colors.append("lightblue")
            elif "view" in node_type_str:
                node_colors.append("lightgreen")
            elif "control" in node_type_str or "job" in node_type_str:
                node_colors.append("lightcoral")
            elif "column" in node_type_str:
                node_colors.append("lightyellow")
            else:
                node_colors.append("lightgray")
        
        # Draw with labels (may be cluttered for large graphs)
        show_labels = len(G.nodes()) < 50
        nx.draw(G, pos, with_labels=show_labels, node_color=node_colors, 
                node_size=500, font_size=8, arrows=True, edge_color="gray",
                alpha=0.7, linewidths=0.5)
        
        plt.title(f"Graph Visualization ({len(G.nodes())} nodes, {len(G.edges())} edges)")
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to {output_file}")
        plt.show()
        
    except Exception as e:
        print(f"Error visualizing graph: {e}", file=sys.stderr)
        sys.exit(1)

def get_graph_stats():
    """Get graph statistics."""
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        with driver.session() as session:
            # Get node counts by type
            node_result = session.run("""
                MATCH (n:Node)
                RETURN n.type AS type, count(n) AS count
                ORDER BY count DESC
            """)
            
            print("\n=== Node Statistics ===")
            for record in node_result:
                print(f"  {record['type']}: {record['count']}")
            
            # Get relationship counts
            rel_result = session.run("""
                MATCH ()-[r:RELATIONSHIP]->()
                RETURN r.label AS label, count(r) AS count
                ORDER BY count DESC
            """)
            
            print("\n=== Relationship Statistics ===")
            for record in rel_result:
                print(f"  {record['label']}: {record['count']}")
            
            # Total counts
            total_result = session.run("""
                MATCH (n:Node)
                WITH count(n) AS nodeCount
                MATCH ()-[r:RELATIONSHIP]->()
                RETURN nodeCount, count(r) AS relCount
            """)
            
            for record in total_result:
                print(f"\n=== Total ===")
                print(f"  Nodes: {record['nodeCount']}")
                print(f"  Relationships: {record['relCount']}")
        
        driver.close()
        
    except Exception as e:
        print(f"Error getting graph stats: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize Neo4j graph")
    parser.add_argument("--limit", type=int, default=100, help="Limit number of nodes to visualize")
    parser.add_argument("--output", type=str, default="graph_visualization.png", help="Output file path")
    parser.add_argument("--stats", action="store_true", help="Show graph statistics instead of visualizing")
    
    args = parser.parse_args()
    
    if args.stats:
        get_graph_stats()
    else:
        visualize_graph(limit=args.limit, output_file=args.output)

