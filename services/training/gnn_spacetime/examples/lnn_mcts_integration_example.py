"""Example: LNN and MCTS integration for narrative intelligence.

Demonstrates:
1. LNN-based continuous-time temporal modeling
2. MCTS for narrative path planning
3. GNN-accelerated MCTS rollouts
4. What-if analysis with MCTS
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from gnn_spacetime.narrative import (
    NarrativeGraph, NarrativeNode, NarrativeEdge, Storyline, NarrativeType
)
from gnn_spacetime.narrative.enhanced_narrative_gnn import EnhancedNarrativeGNN
from gnn_spacetime.data.sample_data_generator import generate_synthetic_corporate_merger


def main():
    """Run LNN and MCTS integration example."""
    print("=" * 80)
    print("LNN and MCTS Integration Example")
    print("=" * 80)
    
    # Step 1: Create narrative graph
    print("\n1. Creating narrative graph...")
    events = generate_synthetic_corporate_merger(
        company_a="TechCorp",
        company_b="StartupInc",
        duration_days=180,
        num_events=20
    )
    
    # Convert to narrative graph (simplified)
    nodes = []
    edges = []
    storylines = {}
    
    for i, event in enumerate(events[:10]):  # Use first 10 events
        source_id = event.get("source", f"node_{i}")
        target_id = event.get("target", f"node_{i+1}")
        
        # Create nodes
        if not any(n.node_id == source_id for n in nodes):
            nodes.append(NarrativeNode(
                node_id=source_id,
                node_type="entity",
                causal_influence=0.5 + i * 0.05,
                explanatory_power=0.6
            ))
        if not any(n.node_id == target_id for n in nodes):
            nodes.append(NarrativeNode(
                node_id=target_id,
                node_type="entity",
                causal_influence=0.5 + (i+1) * 0.05,
                explanatory_power=0.6
            ))
        
        # Create edge
        edges.append(NarrativeEdge(
            source=source_id,
            target=target_id,
            relation_type=event.get("type", "relates_to"),
            causal_strength=0.7,
            counterfactual_importance=0.6
        ))
    
    # Create storyline
    storyline = Storyline(
        storyline_id="merger_story",
        theme="Corporate merger between TechCorp and StartupInc",
        narrative_type=NarrativeType.EXPLANATION
    )
    storylines[storyline.storyline_id] = storyline
    
    graph = NarrativeGraph(nodes, edges, storylines)
    print(f"   Created graph: {len(nodes)} nodes, {len(edges)} edges, {len(storylines)} storylines")
    
    # Step 2: Initialize Enhanced GNN with LNN and MCTS
    print("\n2. Initializing Enhanced Narrative GNN with LNN and MCTS...")
    enhanced_gnn = EnhancedNarrativeGNN(
        narrative_graph=graph,
        use_lnn=True,
        use_mcts=True,
        lnn_time_constant=1.0,
        mcts_rollouts=50,  # Reduced for demo
        mcts_exploration_c=1.414
    )
    print("   ✓ LNN enabled for continuous-time temporal modeling")
    print("   ✓ MCTS enabled for narrative path planning")
    
    # Step 3: Demonstrate LNN-based state update
    print("\n3. Demonstrating LNN-based state update...")
    node_id = nodes[0].node_id
    messages = torch.randn(128)  # Random messages
    prev_state = torch.randn(64)  # Previous state
    time_delta = torch.tensor(0.1)  # Time delta
    
    updated_state = enhanced_gnn.update_node_state_lnn(
        node_id=node_id,
        messages=messages,
        prev_state=prev_state,
        time_delta=time_delta
    )
    print(f"   Updated state shape: {updated_state.shape}")
    print(f"   State change: {torch.norm(updated_state - prev_state).item():.4f}")
    
    # Step 4: Demonstrate MCTS path planning
    print("\n4. Demonstrating MCTS narrative path planning...")
    current_state = {
        "time": 0.0,
        "storyline_id": "merger_story",
        "events": []
    }
    
    action_sequence, best_value = enhanced_gnn.plan_narrative_path_mcts(
        current_state=current_state,
        storyline_id="merger_story",
        num_iterations=30  # Reduced for demo
    )
    print(f"   Planned {len(action_sequence)} actions")
    print(f"   Best value: {best_value:.4f}")
    if action_sequence:
        print(f"   First action: {action_sequence[0]}")
    
    # Step 5: Demonstrate MCTS what-if analysis
    print("\n5. Demonstrating MCTS what-if analysis...")
    counterfactual = {
        "remove_node": nodes[0].node_id,
        "description": "What if the first key player was removed?"
    }
    
    what_if_result = enhanced_gnn.what_if_analysis_mcts(
        counterfactual_condition=counterfactual,
        storyline_id="merger_story",
        num_iterations=30  # Reduced for demo
    )
    print(f"   Original value: {what_if_result.get('original_value', 0):.4f}")
    print(f"   Counterfactual value: {what_if_result.get('counterfactual_value', 0):.4f}")
    print(f"   Difference: {what_if_result.get('difference', 0):.4f}")
    print(f"   Analysis: {what_if_result.get('analysis', 'N/A')}")
    
    # Step 6: Compare with standard GNN
    print("\n6. Comparing with standard narrative GNN...")
    standard_gnn = MultiPurposeNarrativeGNN(narrative_graph=graph)
    
    # Run explanation
    print("\n   Standard GNN explanation:")
    standard_result = standard_gnn.forward(
        graph=graph,
        current_time=0.0,
        task_mode="explain",
        storyline_id="merger_story"
    )
    print(f"   Result type: {type(standard_result)}")
    
    print("\n   Enhanced GNN explanation (with LNN + MCTS):")
    enhanced_result = enhanced_gnn.forward(
        graph=graph,
        current_time=0.0,
        task_mode="explain",
        storyline_id="merger_story"
    )
    print(f"   Result type: {type(enhanced_result)}")
    
    print("\n" + "=" * 80)
    print("Example complete!")
    print("=" * 80)
    print("\nKey Benefits Demonstrated:")
    print("  ✓ LNN provides continuous-time state evolution")
    print("  ✓ MCTS enables intelligent narrative path planning")
    print("  ✓ GNN-accelerated MCTS speeds up exploration")
    print("  ✓ What-if analysis explores counterfactual scenarios")


if __name__ == "__main__":
    main()

