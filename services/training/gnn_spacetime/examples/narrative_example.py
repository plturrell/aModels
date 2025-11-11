"""Example usage of narrative spacetime GNN system.

Demonstrates the full workflow from data loading to explanation/prediction/anomaly detection.
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from gnn_spacetime.narrative import (
    NarrativeGraph, MultiPurposeNarrativeGNN,
    NarrativeNode, NarrativeEdge, Storyline, NarrativeType
)
from gnn_spacetime.data.narrative_data_loader import NarrativeDataLoader
from gnn_spacetime.data.sample_data_generator import (
    generate_synthetic_corporate_merger,
    create_sample_story_themes
)
from gnn_spacetime.evaluation.metrics import (
    evaluate_explanation_quality,
    evaluate_prediction_accuracy
)


def main():
    """Run example narrative intelligence workflow."""
    print("=" * 80)
    print("Narrative Spacetime GNN Example")
    print("=" * 80)
    
    # Step 1: Generate synthetic data
    print("\n1. Generating synthetic corporate merger data...")
    events = generate_synthetic_corporate_merger(
        company_a="TechCorp",
        company_b="StartupInc",
        duration_days=180,
        num_events=20
    )
    print(f"   Generated {len(events)} events")
    
    # Step 2: Load into narrative graph
    print("\n2. Loading events into narrative graph...")
    loader = NarrativeDataLoader()
    story_themes = create_sample_story_themes()
    graph = loader.convert_raw_events_to_narrative_graph(events, story_themes)
    print(f"   Created graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges, {len(graph.storylines)} storylines")
    
    # Step 3: Initialize GNN
    print("\n3. Initializing MultiPurposeNarrativeGNN...")
    gnn = MultiPurposeNarrativeGNN(narrative_graph=graph)
    print("   GNN initialized")
    
    # Step 4: Generate explanation
    print("\n4. Generating explanation...")
    explanation_result = gnn.forward(
        graph,
        current_time=90.0,
        task_mode="explain",
        storyline_id="merger_story"
    )
    print(f"   Explanation Quality: {explanation_result.get('explanatory_quality', 0.0):.2f}")
    print(f"   Explanation:\n   {explanation_result.get('explanation', 'N/A')}")
    
    # Step 5: Predict future
    print("\n5. Predicting future narrative...")
    prediction_result = gnn.forward(
        graph,
        current_time=90.0,
        task_mode="predict",
        storyline_id="merger_story"
    )
    prediction = prediction_result.get("prediction", {})
    print(f"   Predictive Confidence: {prediction_result.get('predictive_confidence', 0.0):.2f}")
    print(f"   Predicted Events: {len(prediction.get('predicted_events', []))}")
    for event in prediction.get("predicted_events", [])[:3]:
        print(f"     - {event.get('description', 'N/A')} at time {event.get('time', 0.0):.1f}")
    
    # Step 6: Detect anomalies
    print("\n6. Detecting anomalies...")
    anomaly_result = gnn.forward(
        graph,
        current_time=90.0,
        task_mode="detect_anomalies",
        storyline_id="merger_story"
    )
    anomalies = anomaly_result.get("anomalies", [])
    print(f"   Anomaly Score: {anomaly_result.get('anomaly_score', 0.0):.2f}")
    print(f"   Detected Anomalies: {len(anomalies)}")
    for anomaly in anomalies[:3]:
        print(f"     - {anomaly.get('type', 'unknown')}: {anomaly.get('description', 'N/A')}")
    
    # Step 7: Evaluate quality
    print("\n7. Evaluating system quality...")
    if explanation_result.get("explanation"):
        quality = evaluate_explanation_quality(
            explanation_result["explanation"],
            key_entities=["TechCorp", "StartupInc", "cultural_resistance"]
        )
        print(f"   Explanation Quality Score: {quality.get('overall_quality', 0.0):.2f}")
        print(f"     - Entity Coverage: {quality.get('entity_coverage', 0.0):.2f}")
        print(f"     - Causal Language: {quality.get('causal_language', 0.0):.2f}")
    
    print("\n" + "=" * 80)
    print("Example complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

