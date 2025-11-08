"""Bridge between LangGraph and narrative GNN for bidirectional data flow."""

import logging
from typing import Dict, List, Optional, Any
import json
import re

try:
    from langchain.chat_models import init_chat_model
    from langchain_core.messages import HumanMessage
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    init_chat_model = None
    HumanMessage = None

logger = logging.getLogger(__name__)


class LangGraphNarrativeBridge:
    """Bridge for converting between natural language and GNN parameters."""
    
    def __init__(self, llm_model: str = "openai:gpt-4o-mini"):
        """Initialize bridge.
        
        Args:
            llm_model: LLM model for extraction and enrichment
        """
        if not HAS_LANGCHAIN:
            raise ImportError("LangChain is required for LangGraphNarrativeBridge")
        
        self.llm = init_chat_model(llm_model)
        logger.info("Initialized LangGraphNarrativeBridge")
    
    def langgraph_to_gnn(self, natural_language_query: str) -> Dict[str, Any]:
        """Convert natural language query to GNN-executable parameters.
        
        Args:
            natural_language_query: Natural language query
            
        Returns:
            Dict with GNN parameters
        """
        extraction_prompt = f"""Extract from this natural language query the following information:

- Target entities/nodes (company names, people, concepts, systems mentioned)
- Time period of interest (dates, time ranges, relative times like "last year", "next 6 months")
- Narrative themes to focus on (merger, research, collaboration, etc.)
- Specific relationships to analyze
- Storyline identifiers if mentioned

Query: {natural_language_query}

Respond in JSON format:
{{
    "entities": ["entity1", "entity2"],
    "time_period": "description or null",
    "time_point": 0.0,
    "themes": ["theme1", "theme2"],
    "relationships": ["relationship1", "relationship2"],
    "storyline_id": "specific storyline if mentioned or null",
    "query_type": "explain|predict|detect_anomalies|what_if"
}}"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=extraction_prompt)])
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                extracted = json.loads(json_match.group())
            else:
                # Fallback structure
                extracted = {
                    "entities": [],
                    "time_period": None,
                    "time_point": 0.0,
                    "themes": [],
                    "relationships": [],
                    "storyline_id": None,
                    "query_type": "explain"
                }
            
            # Convert time period to time point
            if extracted.get("time_period"):
                extracted["time_point"] = self._parse_time_period(extracted["time_period"])
            
            logger.debug(f"Extracted GNN parameters: {extracted}")
            return extracted
        except Exception as e:
            logger.error(f"Error converting query to GNN params: {e}")
            return {
                "entities": [],
                "time_period": None,
                "time_point": 0.0,
                "themes": [],
                "relationships": [],
                "storyline_id": None,
                "query_type": "explain"
            }
    
    def _parse_time_period(self, time_period: str) -> float:
        """Parse time period description to time point.
        
        Args:
            time_period: Time period description
            
        Returns:
            Time point (float, days from now)
        """
        if not time_period:
            return 0.0
        
        time_period_lower = time_period.lower()
        
        # Common time phrases
        time_map = {
            "last year": -365.0,
            "last month": -30.0,
            "last week": -7.0,
            "yesterday": -1.0,
            "today": 0.0,
            "tomorrow": 1.0,
            "next week": 7.0,
            "next month": 30.0,
            "next 6 months": 180.0,
            "next year": 365.0,
            "in 6 months": 180.0,
            "in a year": 365.0
        }
        
        for phrase, days in time_map.items():
            if phrase in time_period_lower:
                return days
        
        # Try to extract numbers
        import re
        numbers = re.findall(r'\d+', time_period)
        if numbers:
            num = int(numbers[0])
            if "month" in time_period_lower:
                return num * 30.0
            elif "year" in time_period_lower:
                return num * 365.0
            elif "day" in time_period_lower or "day" in time_period_lower:
                return float(num)
        
        return 0.0
    
    def gnn_to_langgraph(
        self,
        gnn_output: Dict[str, Any],
        query_type: str
    ) -> Dict[str, Any]:
        """Enrich GNN structured output for LLM response generation.
        
        Args:
            gnn_output: GNN output dict
            query_type: Type of query ("explain", "predict", "detect_anomalies")
            
        Returns:
            Enriched output dict
        """
        if query_type == "explain":
            return self._enrich_explanation(gnn_output)
        elif query_type == "predict":
            return self._enrich_prediction(gnn_output)
        elif query_type == "detect_anomalies":
            return self._enrich_anomalies(gnn_output)
        elif query_type == "what_if":
            return self._enrich_what_if(gnn_output)
        else:
            return gnn_output
    
    def _enrich_explanation(self, gnn_output: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich explanation output with narrative flair.
        
        Args:
            gnn_output: GNN explanation output
            
        Returns:
            Enriched explanation
        """
        explanation = gnn_output.get("explanation", "")
        if not explanation:
            return gnn_output
        
        enrichment_prompt = f"""Transform this structured explanation into a compelling narrative:

Structured Explanation: {explanation}

Make it:
- Engaging and story-like with narrative flow
- Include character motivations and context
- Highlight key turning points and dramatic moments
- Maintain factual accuracy
- Use vivid but professional language
- Structure as a coherent story with beginning, middle, and implications

Enhanced Narrative:"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=enrichment_prompt)])
            enriched_explanation = response.content
            
            return {
                **gnn_output,
                "enriched_explanation": enriched_explanation,
                "original_explanation": explanation
            }
        except Exception as e:
            logger.error(f"Error enriching explanation: {e}")
            return gnn_output
    
    def _enrich_prediction(self, gnn_output: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich prediction output.
        
        Args:
            gnn_output: GNN prediction output
            
        Returns:
            Enriched prediction
        """
        prediction = gnn_output.get("prediction", {})
        events = prediction.get("predicted_events", [])
        confidence = gnn_output.get("predictive_confidence", 0.0)
        
        enrichment_prompt = f"""Based on this narrative prediction, create a compelling forecast:

Prediction Confidence: {confidence:.1%}
Predicted Events: {json.dumps(events, indent=2)}

Provide:
- Clear, engaging summary of likely future outcomes
- Confidence level with reasoning
- Key events to watch for with timeline
- Potential implications and consequences
- Use narrative style that tells the story of what's coming

Forecast Narrative:"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=enrichment_prompt)])
            enriched_prediction = response.content
            
            return {
                **gnn_output,
                "enriched_prediction": enriched_prediction,
                "original_prediction": prediction
            }
        except Exception as e:
            logger.error(f"Error enriching prediction: {e}")
            return gnn_output
    
    def _enrich_anomalies(self, gnn_output: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich anomaly detection output.
        
        Args:
            gnn_output: GNN anomaly output
            
        Returns:
            Enriched anomaly report
        """
        anomalies = gnn_output.get("anomalies", [])
        anomaly_score = gnn_output.get("anomaly_score", 0.0)
        
        enrichment_prompt = f"""Based on this anomaly detection analysis, create a clear alert report:

Anomaly Score: {anomaly_score:.1%}
Detected Anomalies: {json.dumps(anomalies, indent=2)}

Provide:
- Executive summary of detected anomalies
- Severity assessment with clear categorization
- Potential causes or explanations for each anomaly
- Recommendations for investigation and action
- Use clear, professional language suitable for alerts

Anomaly Report:"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=enrichment_prompt)])
            enriched_report = response.content
            
            return {
                **gnn_output,
                "enriched_report": enriched_report,
                "original_anomalies": anomalies
            }
        except Exception as e:
            logger.error(f"Error enriching anomalies: {e}")
            return gnn_output
    
    def _enrich_what_if(self, gnn_output: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich what-if analysis output.
        
        Args:
            gnn_output: GNN what-if output
            
        Returns:
            Enriched counterfactual analysis
        """
        analysis = gnn_output.get("counterfactual_analysis", {})
        
        enrichment_prompt = f"""Based on this counterfactual analysis, create an engaging comparison:

Counterfactual Analysis: {json.dumps(analysis, indent=2)}

Provide:
- Clear comparison of original vs counterfactual scenarios
- Key differences and their implications
- Insights about what might have been different
- Lessons learned from the comparison
- Use engaging, analytical language

Counterfactual Narrative:"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=enrichment_prompt)])
            enriched_analysis = response.content
            
            return {
                **gnn_output,
                "enriched_analysis": enriched_analysis,
                "original_analysis": analysis
            }
        except Exception as e:
            logger.error(f"Error enriching what-if analysis: {e}")
            return gnn_output

