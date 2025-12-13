"""
Decision/Suggestion Agent

Provides investment recommendations and actionable insights by synthesizing:
- Summary analysis
- SWOT analysis
- Financial metrics
- Risk assessment
- Market context
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from .base_agent import BaseAgent
from utils import config, get_logger

logger = get_logger("decision_agent")


class DecisionAgent(BaseAgent):
    """Agent for investment decisions and recommendations."""
    
    def __init__(self):
        """Initialize decision agent."""
        super().__init__(
            agent_name="Decision Agent",
            temperature=0.3  # Balanced temperature for thoughtful analysis
        )
    
    def analyze(
        self,
        doc_id: str,
        risk_tolerance: str = "moderate",
        investment_horizon: str = "medium_term"
    ) -> Dict[str, Any]:
        """
        Generate investment recommendation and rationale.
        
        Args:
            doc_id: Document identifier
            risk_tolerance: Investor risk tolerance (conservative/moderate/aggressive)
            investment_horizon: Investment timeframe (short_term/medium_term/long_term)
        
        Returns:
            Dictionary containing recommendation and analysis
        """
        logger.info(f"Generating investment decision for {doc_id}")
        
        # Load previous analyses if available
        summary_analysis = self._load_analysis(doc_id, "summary_analysis")
        swot_analysis = self._load_analysis(doc_id, "swot_analysis")
        metrics_analysis = self._load_analysis(doc_id, "metrics_analysis")
        
        # Create vector store for additional context
        vector_store = self.create_vector_store(doc_id)
        
        # Assess different investment factors
        valuation_assessment = self._assess_valuation(doc_id, metrics_analysis, vector_store)
        risk_assessment = self._assess_risks(doc_id, swot_analysis, vector_store)
        growth_potential = self._assess_growth(doc_id, swot_analysis, vector_store)
        competitive_position = self._assess_competitive_position(doc_id, swot_analysis, vector_store)
        
        # Generate overall recommendation
        recommendation = self._generate_recommendation(
            summary_analysis=summary_analysis,
            swot_analysis=swot_analysis,
            metrics_analysis=metrics_analysis,
            valuation=valuation_assessment,
            risks=risk_assessment,
            growth=growth_potential,
            competitive=competitive_position,
            risk_tolerance=risk_tolerance,
            investment_horizon=investment_horizon
        )
        
        # Compile results
        result = {
            "doc_id": doc_id,
            "agent": self.agent_name,
            "recommendation": recommendation,
            "assessments": {
                "valuation": valuation_assessment,
                "risks": risk_assessment,
                "growth_potential": growth_potential,
                "competitive_position": competitive_position
            },
            "parameters": {
                "risk_tolerance": risk_tolerance,
                "investment_horizon": investment_horizon
            },
            "metadata": {
                "analyses_used": {
                    "summary": summary_analysis is not None,
                    "swot": swot_analysis is not None,
                    "metrics": metrics_analysis is not None
                }
            }
        }
        
        # Save analysis
        self.save_analysis(doc_id, result, "decision_analysis")
        
        logger.info(f"✅ Investment decision generated for {doc_id}")
        return result
    
    def _load_analysis(self, doc_id: str, filename: str) -> Optional[Dict]:
        """Load previous analysis results."""
        analysis_path = config.paths.final_dir / doc_id / f"{filename}.json"
        
        if not analysis_path.exists():
            logger.warning(f"Analysis not found: {filename}")
            return None
        
        with open(analysis_path, 'r') as f:
            return json.load(f)
    
    def _assess_valuation(
        self,
        doc_id: str,
        metrics_analysis: Optional[Dict],
        vector_store
    ) -> Dict[str, Any]:
        """Assess company valuation."""
        
        if metrics_analysis:
            metrics = metrics_analysis.get('metrics', {})
            xbrl_metrics = metrics.get('xbrl_metrics', {})
            ratios = metrics.get('calculated_ratios', {})
            
            # Use metrics for valuation assessment
            assessment_text = f"""
Based on the following financial data, assess the company's valuation:

KEY METRICS:
- Revenue: {xbrl_metrics.get('revenue', {}).get('value', 'N/A')}
- Net Income: {xbrl_metrics.get('net_income', {}).get('value', 'N/A')}
- Assets: {xbrl_metrics.get('assets', {}).get('value', 'N/A')}
- Equity: {xbrl_metrics.get('equity', {}).get('value', 'N/A')}

RATIOS:
- Gross Margin: {ratios.get('gross_margin', 'N/A')}
- Operating Margin: {ratios.get('operating_margin', 'N/A')}
- Net Margin: {ratios.get('net_margin', 'N/A')}
- ROE: {ratios.get('roe', 'N/A')}

Provide:
1. Valuation assessment (Undervalued/Fair/Overvalued)
2. Key factors supporting this assessment
3. Comparison to typical industry multiples (if possible)

Assessment:
"""
        else:
            # Use RAG to find valuation-related information
            chunks = self.retrieve_relevant_chunks(
                doc_id=doc_id,
                query="What are the key financial metrics and valuation indicators?",
                k=3,
                vector_store=vector_store
            )
            
            context = "\n\n".join([c['text'][:800] for c in chunks])
            
            assessment_text = f"""
Based on the following information from the filing, assess the company's valuation:

{context}

Provide:
1. Valuation assessment
2. Key supporting factors
3. Concerns or limitations

Assessment:
"""
        
        assessment = self.generate_response(assessment_text)
        
        return {
            "assessment": assessment,
            "confidence": "medium" if metrics_analysis else "low"
        }
    
    def _assess_risks(
        self,
        doc_id: str,
        swot_analysis: Optional[Dict],
        vector_store
    ) -> Dict[str, Any]:
        """Assess investment risks."""
        
        if swot_analysis:
            swot = swot_analysis.get('swot_analysis', {})
            weaknesses = swot.get('weaknesses', {}).get('items', [])
            threats = swot.get('threats', {}).get('items', [])
            
            risk_text = f"""
Based on the following weaknesses and threats identified in the SWOT analysis:

WEAKNESSES:
{self._format_swot_items(weaknesses, 'weakness')}

THREATS:
{self._format_swot_items(threats, 'threat')}

Provide a comprehensive risk assessment:
1. Overall risk level (Low/Medium/High)
2. Most critical risks to investment thesis
3. Risk mitigation strategies
4. Risk/reward balance

Risk Assessment:
"""
        else:
            # Use RAG
            chunks = self.retrieve_relevant_chunks(
                doc_id=doc_id,
                query="What are the key risks and uncertainties facing the company?",
                k=5,
                vector_store=vector_store
            )
            
            context = "\n\n".join([c['text'][:800] for c in chunks])
            
            risk_text = f"""
Based on the following risk factors from the filing:

{context}

Provide:
1. Overall risk level
2. Key risks to consider
3. Risk mitigation

Risk Assessment:
"""
        
        assessment = self.generate_response(risk_text)
        
        return {
            "assessment": assessment,
            "confidence": "high" if swot_analysis else "medium"
        }
    
    def _assess_growth(
        self,
        doc_id: str,
        swot_analysis: Optional[Dict],
        vector_store
    ) -> Dict[str, Any]:
        """Assess growth potential."""
        
        if swot_analysis:
            swot = swot_analysis.get('swot_analysis', {})
            strengths = swot.get('strengths', {}).get('items', [])
            opportunities = swot.get('opportunities', {}).get('items', [])
            
            growth_text = f"""
Based on the following strengths and opportunities:

STRENGTHS:
{self._format_swot_items(strengths, 'strength')}

OPPORTUNITIES:
{self._format_swot_items(opportunities, 'opportunity')}

Assess the company's growth potential:
1. Growth outlook (Low/Moderate/High)
2. Key growth drivers
3. Timeframe for growth realization
4. Growth risks and challenges

Growth Assessment:
"""
        else:
            # Use RAG
            chunks = self.retrieve_relevant_chunks(
                doc_id=doc_id,
                query="What are the company's growth opportunities and strategic initiatives?",
                k=4,
                vector_store=vector_store
            )
            
            context = "\n\n".join([c['text'][:800] for c in chunks])
            
            growth_text = f"""
Based on the following information about growth:

{context}

Provide:
1. Growth outlook
2. Key growth drivers
3. Growth timeline

Growth Assessment:
"""
        
        assessment = self.generate_response(growth_text)
        
        return {
            "assessment": assessment,
            "confidence": "high" if swot_analysis else "medium"
        }
    
    def _assess_competitive_position(
        self,
        doc_id: str,
        swot_analysis: Optional[Dict],
        vector_store
    ) -> Dict[str, Any]:
        """Assess competitive position."""
        
        chunks = self.retrieve_relevant_chunks(
            doc_id=doc_id,
            query="What is the company's competitive position and market share?",
            k=3,
            vector_store=vector_store
        )
        
        context = "\n\n".join([c['text'][:800] for c in chunks])
        
        competitive_text = f"""
Based on the following information about competitive position:

{context}

Assess:
1. Competitive position (Weak/Moderate/Strong/Dominant)
2. Key competitive advantages
3. Competitive threats
4. Sustainability of competitive position

Competitive Assessment:
"""
        
        assessment = self.generate_response(competitive_text)
        
        return {
            "assessment": assessment,
            "confidence": "medium"
        }
    
    def _format_swot_items(self, items: List[Dict], key: str) -> str:
        """Format SWOT items."""
        if not items:
            return "None identified"
        
        formatted = []
        for item in items:
            if key in item:
                formatted.append(f"- {item[key]}")
        
        return "\n".join(formatted) if formatted else "None identified"
    
    def _generate_recommendation(
        self,
        summary_analysis: Optional[Dict],
        swot_analysis: Optional[Dict],
        metrics_analysis: Optional[Dict],
        valuation: Dict,
        risks: Dict,
        growth: Dict,
        competitive: Dict,
        risk_tolerance: str,
        investment_horizon: str
    ) -> Dict[str, Any]:
        """Generate final investment recommendation."""
        
        # Compile all assessments
        prompt = f"""
You are an investment analyst providing a recommendation based on comprehensive analysis.

INVESTOR PROFILE:
- Risk Tolerance: {risk_tolerance}
- Investment Horizon: {investment_horizon}

ANALYSIS SUMMARY:

VALUATION ASSESSMENT:
{valuation.get('assessment', 'Not available')}

RISK ASSESSMENT:
{risks.get('assessment', 'Not available')}

GROWTH POTENTIAL:
{growth.get('assessment', 'Not available')}

COMPETITIVE POSITION:
{competitive.get('assessment', 'Not available')}

Based on all available information, provide:

1. INVESTMENT RATING: Choose one of:
   - Strong Buy
   - Buy
   - Hold
   - Sell
   - Strong Sell

2. CONFIDENCE LEVEL: (Low/Medium/High)

3. KEY RATIONALE: (3-5 bullet points explaining the rating)

4. TARGET INVESTOR PROFILE: Who should consider this investment?

5. KEY RISKS TO MONITOR: What should investors watch closely?

6. TIMEFRAME: Expected timeframe for thesis to play out

7. ACTIONABLE RECOMMENDATIONS: Specific next steps for investors

Provide response in JSON format:
{{
  "rating": "Buy/Sell/Hold/etc",
  "confidence": "Low/Medium/High",
  "rationale": ["point 1", "point 2", ...],
  "target_investor": "description",
  "risks_to_monitor": ["risk 1", "risk 2", ...],
  "timeframe": "description",
  "action_items": ["action 1", "action 2", ...]
}}

JSON Response:
"""
        
        response = self.generate_response(prompt)
        
        # Parse JSON response
        try:
            recommendation = json.loads(response)
        except:
            # Fallback parsing
            recommendation = {
                "rating": "Hold",
                "confidence": "Low",
                "rationale": [response[:500]],
                "target_investor": "Unable to parse recommendation",
                "risks_to_monitor": [],
                "timeframe": "Unknown",
                "action_items": []
            }
        
        return recommendation
    
    def compare_investments(
        self,
        doc_ids: List[str],
        criteria: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple investment opportunities.
        
        Args:
            doc_ids: List of document IDs to compare
            criteria: Optional list of comparison criteria
        
        Returns:
            Comparative analysis
        """
        logger.info(f"Comparing {len(doc_ids)} investments")
        
        # Load decision analyses for each document
        analyses = {}
        for doc_id in doc_ids:
            analysis = self._load_analysis(doc_id, "decision_analysis")
            if analysis:
                analyses[doc_id] = analysis
        
        if not analyses:
            return {"error": "No decision analyses found for comparison"}
        
        # Generate comparison
        comparison_text = self._format_comparison(analyses)
        
        prompt = f"""
Compare the following investment opportunities:

{comparison_text}

Provide:
1. Relative ranking (best to worst)
2. Comparison summary for each investment
3. Which investment is best for different investor profiles
4. Key differentiating factors

Comparison Analysis:
"""
        
        comparison = self.generate_response(prompt)
        
        return {
            "investments_compared": list(analyses.keys()),
            "comparison": comparison,
            "individual_ratings": {
                doc_id: data.get('recommendation', {}).get('rating')
                for doc_id, data in analyses.items()
            }
        }
    
    def _format_comparison(self, analyses: Dict[str, Dict]) -> str:
        """Format analyses for comparison."""
        formatted = []
        
        for doc_id, analysis in analyses.items():
            rec = analysis.get('recommendation', {})
            formatted.append(f"""
Investment: {doc_id}
Rating: {rec.get('rating', 'N/A')}
Confidence: {rec.get('confidence', 'N/A')}
Key Rationale: {', '.join(rec.get('rationale', [])[:3])}
""")
        
        return "\n".join(formatted)


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate investment decision")
    parser.add_argument("--doc-id", required=True, help="Document ID")
    parser.add_argument("--risk-tolerance", default="moderate",
                       choices=["conservative", "moderate", "aggressive"],
                       help="Risk tolerance")
    parser.add_argument("--horizon", default="medium_term",
                       choices=["short_term", "medium_term", "long_term"],
                       help="Investment horizon")
    
    args = parser.parse_args()
    
    agent = DecisionAgent()
    result = agent.analyze(
        args.doc_id,
        risk_tolerance=args.risk_tolerance,
        investment_horizon=args.horizon
    )
    
    print("\n" + "="*80)
    print("INVESTMENT RECOMMENDATION")
    print("="*80)
    
    rec = result['recommendation']
    
    print(f"\n🎯 RATING: {rec.get('rating', 'N/A')}")
    print(f"📊 CONFIDENCE: {rec.get('confidence', 'N/A')}")
    
    print("\n💡 KEY RATIONALE:")
    for point in rec.get('rationale', []):
        print(f"  • {point}")
    
    print(f"\n👤 TARGET INVESTOR: {rec.get('target_investor', 'N/A')}")
    
    print("\n⚠️  RISKS TO MONITOR:")
    for risk in rec.get('risks_to_monitor', []):
        print(f"  • {risk}")
    
    print(f"\n⏱️  TIMEFRAME: {rec.get('timeframe', 'N/A')}")
    
    print("\n✅ ACTION ITEMS:")
    for action in rec.get('action_items', []):
        print(f"  • {action}")


if __name__ == "__main__":
    main()
