"""Example: Natural language chat interface for narrative intelligence."""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "training"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langgraph_narrative import (
    NarrativeLangGraphAgent,
    LangGraphNarrativeBridge,
    NarrativeConversationMemory
)
from gnn_spacetime.narrative import MultiPurposeNarrativeGNN, NarrativeGraph
from gnn_spacetime.data.narrative_data_loader import NarrativeDataLoader
from gnn_spacetime.data.sample_data_generator import (
    generate_synthetic_corporate_merger,
    create_sample_story_themes
)


def main():
    """Run example narrative chat interface."""
    print("=" * 80)
    print("Narrative Intelligence Chat Interface")
    print("=" * 80)
    
    # Step 1: Load narrative graph
    print("\n1. Loading narrative graph...")
    events = generate_synthetic_corporate_merger(
        company_a="TechCorp",
        company_b="StartupInc",
        duration_days=180,
        num_events=20
    )
    
    loader = NarrativeDataLoader()
    story_themes = create_sample_story_themes()
    graph = loader.convert_raw_events_to_narrative_graph(events, story_themes)
    print(f"   Loaded: {len(graph.nodes)} nodes, {len(graph.edges)} edges, {len(graph.storylines)} storylines")
    
    # Step 2: Initialize components
    print("\n2. Initializing narrative intelligence system...")
    gnn = MultiPurposeNarrativeGNN(narrative_graph=graph)
    bridge = LangGraphNarrativeBridge()
    memory = NarrativeConversationMemory(conversation_graph=graph)
    agent = NarrativeLangGraphAgent(
        narrative_gnn=gnn,
        narrative_graph=graph
    )
    print("   System ready!")
    
    # Step 3: Example queries
    print("\n3. Example queries:")
    print("-" * 80)
    
    queries = [
        "Explain why the merger between TechCorp and StartupInc failed",
        "What's likely to happen with this partnership in the next 6 months?",
        "Are there any unusual patterns in this collaboration timeline?",
        "What if the cultural resistance had been addressed earlier?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 80)
        
        response = agent.chat(query, graph)
        print(f"Response: {response}")
        
        # Update memory
        gnn_result = agent.graph.invoke({
            "user_query": query,
            "query_type": "",
            "narrative_context": {},
            "gnn_results": {},
            "final_response": "",
            "conversation_history": [],
            "extracted_params": {}
        })
        memory.update_from_conversation(
            query,
            gnn_result.get("gnn_results", {}),
            response
        )
    
    # Step 4: Conversation summary
    print("\n4. Conversation Summary:")
    print("-" * 80)
    summary = memory.get_conversation_summary()
    print(f"   Exchanges: {summary['num_exchanges']}")
    print(f"   Entities mentioned: {summary['entities_mentioned']}")
    print(f"   Storylines created: {summary['storylines_created']}")
    
    print("\n" + "=" * 80)
    print("Example complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

