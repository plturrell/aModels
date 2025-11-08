"""LangGraph agent for narrative intelligence orchestration.

Routes natural language queries to appropriate GNN capabilities and generates responses.
"""

import logging
from typing import TypedDict, Annotated, Literal, Optional, Dict, Any, List
import operator

try:
    from langgraph.graph import StateGraph, END, START
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain.chat_models import init_chat_model
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    StateGraph = None
    END = None
    START = None

# Import narrative GNN
import sys
from pathlib import Path
# Try multiple paths for narrative GNN
possible_paths = [
    Path(__file__).parent.parent.parent.parent / "training" / "gnn_spacetime",
    Path(__file__).parent.parent.parent.parent.parent / "services" / "training" / "gnn_spacetime",
]
for path in possible_paths:
    if path.exists():
        sys.path.insert(0, str(path.parent.parent))
        break

try:
    from gnn_spacetime.narrative import MultiPurposeNarrativeGNN, NarrativeGraph
    HAS_NARRATIVE = True
except ImportError:
    HAS_NARRATIVE = False
    MultiPurposeNarrativeGNN = None
    NarrativeGraph = None

logger = logging.getLogger(__name__)


class NarrativeState(TypedDict):
    """State for narrative LangGraph workflow."""
    user_query: str
    query_type: str  # "explain", "predict", "detect_anomalies", "what_if"
    narrative_context: Dict[str, Any]
    gnn_results: Dict[str, Any]
    final_response: str
    conversation_history: Annotated[List[Dict[str, str]], operator.add]
    extracted_params: Dict[str, Any]


class NarrativeLangGraphAgent:
    """LangGraph agent that orchestrates narrative intelligence queries."""
    
    def __init__(
        self,
        narrative_gnn: Optional[MultiPurposeNarrativeGNN] = None,
        narrative_graph: Optional[NarrativeGraph] = None,
        llm_model: str = "openai:gpt-4o-mini"
    ):
        """Initialize narrative LangGraph agent.
        
        Args:
            narrative_gnn: Optional MultiPurposeNarrativeGNN instance
            narrative_graph: Optional NarrativeGraph instance
            llm_model: LLM model for classification and response generation
        """
        if not HAS_LANGGRAPH:
            raise ImportError("LangGraph is required for NarrativeLangGraphAgent")
        
        self.narrative_gnn = narrative_gnn
        self.narrative_graph = narrative_graph
        self.llm_model = llm_model
        
        # Initialize LLM
        self.llm = init_chat_model(llm_model)
        
        # Build workflow graph
        self.graph = self._build_graph()
        
        logger.info("Initialized NarrativeLangGraphAgent")
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow.
        
        Returns:
            Compiled StateGraph
        """
        workflow = StateGraph(NarrativeState)
        
        # Define nodes
        workflow.add_node("classify_query", self.classify_query_type)
        workflow.add_node("extract_context", self.extract_narrative_context)
        workflow.add_node("execute_gnn", self.execute_gnn_reasoning)
        workflow.add_node("generate_response", self.generate_llm_response)
        
        # Define edges
        workflow.set_entry_point("classify_query")
        workflow.add_edge("classify_query", "extract_context")
        workflow.add_edge("extract_context", "execute_gnn")
        workflow.add_edge("execute_gnn", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow.compile()
    
    def classify_query_type(self, state: NarrativeState) -> Dict[str, Any]:
        """Classify query type using LLM.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with query_type
        """
        query = state.get("user_query", "")
        
        classification_prompt = f"""Classify this user query into one of these categories:

- explain: Requesting explanation of past events or why something happened
- predict: Asking about future outcomes or what will happen
- detect_anomalies: Looking for inconsistencies, violations, or unusual patterns
- what_if: Counterfactual scenario analysis or hypothetical questions

Query: {query}

Respond with ONLY the category name (explain, predict, detect_anomalies, or what_if):"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=classification_prompt)])
            query_type = response.content.strip().lower()
            
            # Validate query type
            valid_types = ["explain", "predict", "detect_anomalies", "what_if"]
            if query_type not in valid_types:
                # Default to explain if unclear
                query_type = "explain"
                logger.warning(f"Invalid query type detected, defaulting to 'explain': {response.content}")
            
            logger.info(f"Classified query as: {query_type}")
            return {"query_type": query_type}
        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            return {"query_type": "explain"}  # Default fallback
    
    def extract_narrative_context(self, state: NarrativeState) -> Dict[str, Any]:
        """Extract narrative context from query.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with narrative_context
        """
        query = state.get("user_query", "")
        query_type = state.get("query_type", "explain")
        
        extraction_prompt = f"""Extract from this query the following information:

- Target entities/nodes (company names, people, concepts mentioned)
- Time period of interest (dates, time ranges, "last year", "next 6 months", etc.)
- Narrative themes to focus on (merger, research, collaboration, etc.)
- Specific relationships to analyze

Query: {query}
Query Type: {query_type}

Respond in JSON format:
{{
    "entities": ["entity1", "entity2"],
    "time_period": "description or null",
    "themes": ["theme1", "theme2"],
    "relationships": ["relationship1", "relationship2"],
    "storyline_id": "specific storyline if mentioned or null"
}}"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=extraction_prompt)])
            # Try to parse JSON from response
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                extracted_params = json.loads(json_match.group())
            else:
                # Fallback: create basic structure
                extracted_params = {
                    "entities": [],
                    "time_period": None,
                    "themes": [],
                    "relationships": [],
                    "storyline_id": None
                }
            
            # Determine time point
            time_t = self._extract_time_point(extracted_params.get("time_period"), query)
            
            # Find or create storyline
            storyline_id = extracted_params.get("storyline_id")
            if not storyline_id and self.narrative_graph:
                # Try to find matching storyline
                storyline_id = self._find_matching_storyline(extracted_params.get("themes", []))
            
            narrative_context = {
                "entities": extracted_params.get("entities", []),
                "time_t": time_t,
                "themes": extracted_params.get("themes", []),
                "relationships": extracted_params.get("relationships", []),
                "storyline_id": storyline_id
            }
            
            logger.info(f"Extracted narrative context: {narrative_context}")
            return {
                "narrative_context": narrative_context,
                "extracted_params": extracted_params
            }
        except Exception as e:
            logger.error(f"Error extracting context: {e}")
            return {
                "narrative_context": {
                    "entities": [],
                    "time_t": 0.0,
                    "themes": [],
                    "relationships": [],
                    "storyline_id": None
                },
                "extracted_params": {}
            }
    
    def _extract_time_point(self, time_period: Optional[str], query: str) -> float:
        """Extract time point from time period description.
        
        Args:
            time_period: Time period description
            query: Original query
            
        Returns:
            Time point (float)
        """
        # Simple time extraction (in practice, would use more sophisticated NLP)
        if not time_period:
            return 0.0  # Default
        
        # Look for common time phrases
        time_phrases = {
            "last year": -365.0,
            "next 6 months": 180.0,
            "next year": 365.0,
            "recently": -30.0,
            "soon": 30.0
        }
        
        for phrase, days in time_phrases.items():
            if phrase.lower() in time_period.lower() or phrase.lower() in query.lower():
                return days
        
        return 0.0  # Default
    
    def _find_matching_storyline(self, themes: List[str]) -> Optional[str]:
        """Find matching storyline from themes.
        
        Args:
            themes: List of theme keywords
            
        Returns:
            Storyline ID or None
        """
        if not self.narrative_graph or not themes:
            return None
        
        # Simple keyword matching
        for storyline_id, storyline in self.narrative_graph.storylines.items():
            theme_lower = storyline.theme.lower()
            for theme in themes:
                if theme.lower() in theme_lower:
                    return storyline_id
        
        return None
    
    def execute_gnn_reasoning(self, state: NarrativeState) -> Dict[str, Any]:
        """Execute GNN reasoning based on query type.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with gnn_results
        """
        if not self.narrative_gnn or not self.narrative_graph:
            return {
                "gnn_results": {"error": "GNN or graph not available"},
                "final_response": "Narrative intelligence system not available."
            }
        
        query_type = state.get("query_type", "explain")
        context = state.get("narrative_context", {})
        time_t = context.get("time_t", 0.0)
        storyline_id = context.get("storyline_id")
        
        try:
            if query_type == "what_if":
                # Handle counterfactual queries
                results = self._answer_what_if_query(state, context)
            else:
                # Standard GNN query
                results = self.narrative_gnn.forward(
                    self.narrative_graph,
                    current_time=time_t,
                    task_mode=query_type,
                    storyline_id=storyline_id
                )
            
            logger.info(f"GNN reasoning complete for {query_type}")
            return {"gnn_results": results}
        except Exception as e:
            logger.error(f"Error executing GNN reasoning: {e}")
            return {
                "gnn_results": {"error": str(e)},
                "final_response": f"Error processing narrative query: {e}"
            }
    
    def _answer_what_if_query(
        self,
        state: NarrativeState,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle what-if counterfactual queries.
        
        Args:
            state: Current state
            context: Narrative context
            
        Returns:
            Counterfactual analysis results
        """
        from gnn_spacetime.narrative import ExplanationGenerator
        
        if not self.narrative_graph:
            return {"error": "Graph not available"}
        
        # Extract counterfactual condition from query
        query = state.get("user_query", "")
        
        # Use LLM to extract counterfactual condition
        extraction_prompt = f"""Extract the counterfactual condition from this query:

Query: {query}

Respond in JSON format:
{{
    "action": "remove" or "modify",
    "node_id": "entity to modify",
    "modifications": {{"key": "value"}} or null
}}"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=extraction_prompt)])
            import json
            import re
            
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                condition = json.loads(json_match.group())
            else:
                condition = {"action": "modify", "node_id": None, "modifications": {}}
            
            # Use explanation generator for what-if analysis
            explanation_gen = ExplanationGenerator(self.narrative_graph)
            storyline_id = context.get("storyline_id")
            
            if storyline_id:
                storyline = self.narrative_graph.get_storyline(storyline_id)
                if storyline:
                    result = explanation_gen.answer_what_if(
                        self.narrative_graph,
                        condition,
                        storyline_id
                    )
                    return {
                        "task": "what_if",
                        "counterfactual_analysis": result
                    }
            
            return {"error": "Could not process what-if query"}
        except Exception as e:
            logger.error(f"Error processing what-if query: {e}")
            return {"error": str(e)}
    
    def generate_llm_response(self, state: NarrativeState) -> Dict[str, Any]:
        """Generate natural language response from GNN results.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with final_response
        """
        query = state.get("user_query", "")
        query_type = state.get("query_type", "explain")
        gnn_results = state.get("gnn_results", {})
        conversation_history = state.get("conversation_history", [])
        
        # Build response prompt
        if query_type == "explain":
            response_prompt = self._build_explanation_prompt(query, gnn_results)
        elif query_type == "predict":
            response_prompt = self._build_prediction_prompt(query, gnn_results)
        elif query_type == "detect_anomalies":
            response_prompt = self._build_anomaly_prompt(query, gnn_results)
        elif query_type == "what_if":
            response_prompt = self._build_what_if_prompt(query, gnn_results)
        else:
            response_prompt = f"Query: {query}\n\nGNN Results: {gnn_results}\n\nGenerate a helpful response."
        
        # Add conversation history for context
        messages = []
        for hist in conversation_history[-3:]:  # Last 3 exchanges
            if hist.get("role") == "user":
                messages.append(HumanMessage(content=hist.get("content", "")))
            elif hist.get("role") == "assistant":
                messages.append(AIMessage(content=hist.get("content", "")))
        
        messages.append(HumanMessage(content=response_prompt))
        
        try:
            response = self.llm.invoke(messages)
            final_response = response.content
            
            logger.info("Generated LLM response")
            return {
                "final_response": final_response,
                "conversation_history": [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": final_response}
                ]
            }
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "final_response": f"I encountered an error generating a response: {e}"
            }
    
    def _build_explanation_prompt(self, query: str, gnn_results: Dict[str, Any]) -> str:
        """Build prompt for explanation generation.
        
        Args:
            query: User query
            gnn_results: GNN results
            
        Returns:
            Prompt string
        """
        explanation = gnn_results.get("explanation", "No explanation available")
        
        return f"""Transform this structured explanation into a compelling, human-readable narrative:

User Query: {query}

Structured Explanation: {explanation}

Make the response:
- Engaging and story-like
- Include character motivations and context
- Highlight key turning points and causal chains
- Maintain factual accuracy
- Be concise but comprehensive
- Use natural, conversational language"""
    
    def _build_prediction_prompt(self, query: str, gnn_results: Dict[str, Any]) -> str:
        """Build prompt for prediction generation.
        
        Args:
            query: User query
            gnn_results: GNN results
            
        Returns:
            Prompt string
        """
        prediction = gnn_results.get("prediction", {})
        confidence = gnn_results.get("predictive_confidence", 0.0)
        events = prediction.get("predicted_events", [])
        
        return f"""Based on this narrative prediction, generate a helpful response:

User Query: {query}

Prediction Confidence: {confidence:.1%}
Predicted Events: {events}

Provide:
- Clear summary of likely future outcomes
- Confidence level and reasoning
- Key events to watch for
- Potential implications
- Use natural, conversational language"""
    
    def _build_anomaly_prompt(self, query: str, gnn_results: Dict[str, Any]) -> str:
        """Build prompt for anomaly detection response.
        
        Args:
            query: User query
            gnn_results: GNN results
            
        Returns:
            Prompt string
        """
        anomalies = gnn_results.get("anomalies", [])
        anomaly_score = gnn_results.get("anomaly_score", 0.0)
        
        return f"""Based on this anomaly detection analysis, generate a helpful response:

User Query: {query}

Anomaly Score: {anomaly_score:.1%}
Detected Anomalies: {anomalies}

Provide:
- Summary of detected anomalies
- Severity assessment
- Potential causes or explanations
- Recommendations for investigation
- Use clear, professional language"""
    
    def _build_what_if_prompt(self, query: str, gnn_results: Dict[str, Any]) -> str:
        """Build prompt for what-if analysis response.
        
        Args:
            query: User query
            gnn_results: GNN results
            
        Returns:
            Prompt string
        """
        analysis = gnn_results.get("counterfactual_analysis", {})
        
        return f"""Based on this counterfactual analysis, generate a helpful response:

User Query: {query}

Counterfactual Analysis: {analysis}

Provide:
- Comparison of original vs counterfactual scenarios
- Key differences and implications
- Insights about what might have been different
- Use engaging, analytical language"""
    
    def chat(self, user_message: str, narrative_graph: Optional[NarrativeGraph] = None) -> str:
        """Main chat interface.
        
        Args:
            user_message: User's natural language query
            narrative_graph: Optional narrative graph (uses self.narrative_graph if None)
            
        Returns:
            Natural language response
        """
        graph = narrative_graph or self.narrative_graph
        if not graph:
            return "Narrative graph not available. Please provide a narrative graph."
        
        # Update graph reference
        if self.narrative_gnn:
            self.narrative_gnn.narrative_graph = graph
        
        # Run workflow
        initial_state = {
            "user_query": user_message,
            "query_type": "",
            "narrative_context": {},
            "gnn_results": {},
            "final_response": "",
            "conversation_history": [],
            "extracted_params": {}
        }
        
        try:
            result = self.graph.invoke(initial_state)
            return result.get("final_response", "I couldn't generate a response.")
        except Exception as e:
            logger.error(f"Error in chat workflow: {e}")
            return f"I encountered an error: {e}"

