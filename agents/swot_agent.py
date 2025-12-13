"""
SWOT Agent

Performs SWOT (Strengths, Weaknesses, Opportunities, Threats) analysis
on SEC filings with evidence-based citations.
"""

from typing import Dict, Any, List
from .base_agent import BaseAgent
from utils import get_logger

logger = get_logger("swot_agent")


class SWOTAgent(BaseAgent):
    """Agent for SWOT analysis."""
    
    def __init__(self):
        """Initialize SWOT agent."""
        super().__init__(
            agent_name="SWOT Agent",
            temperature=0.2  # Low temperature for structured analysis
        )
    
    def analyze(self, doc_id: str) -> Dict[str, Any]:
        """
        Perform SWOT analysis on the filing.
        
        Args:
            doc_id: Document identifier
        
        Returns:
            Dictionary containing SWOT analysis
        """
        logger.info(f"Performing SWOT analysis for {doc_id}")
        
        # Create vector store
        vector_store = self.create_vector_store(doc_id)
        
        # Analyze each SWOT component
        strengths = self._analyze_strengths(doc_id, vector_store)
        weaknesses = self._analyze_weaknesses(doc_id, vector_store)
        opportunities = self._analyze_opportunities(doc_id, vector_store)
        threats = self._analyze_threats(doc_id, vector_store)
        
        # Generate overall assessment
        swot_summary = self._generate_swot_summary(
            strengths, weaknesses, opportunities, threats
        )
        
        # Compile results
        result = {
            "doc_id": doc_id,
            "agent": self.agent_name,
            "swot_analysis": {
                "strengths": strengths,
                "weaknesses": weaknesses,
                "opportunities": opportunities,
                "threats": threats
            },
            "summary": swot_summary,
            "metadata": {
                "total_items": sum([
                    len(strengths['items']),
                    len(weaknesses['items']),
                    len(opportunities['items']),
                    len(threats['items'])
                ])
            }
        }
        
        # Save analysis
        self.save_analysis(doc_id, result, "swot_analysis")
        
        logger.info(f"✅ SWOT analysis complete for {doc_id}")
        return result
    
    def _analyze_strengths(self, doc_id: str, vector_store) -> Dict[str, Any]:
        """Analyze company strengths."""
        query = """
        What are the company's key strengths, competitive advantages, and positive attributes?
        Look for: market position, brand value, unique capabilities, financial strength,
        innovation, customer base, intellectual property, operational efficiency.
        """
        
        relevant_chunks = self.retrieve_relevant_chunks(
            doc_id=doc_id,
            query=query,
            k=5,
            vector_store=vector_store
        )
        
        context = "\n\n".join([c['text'][:800] for c in relevant_chunks])
        
        prompt = f"""
Analyze the following excerpts from a SEC filing and identify the company's STRENGTHS.

Context:
{context}

Please identify 3-5 key strengths. For each strength:
1. State the strength clearly
2. Provide supporting evidence from the text
3. Rate the significance (High/Medium/Low)

Format as JSON:
{{
  "items": [
    {{
      "strength": "Clear statement",
      "evidence": "Supporting quote or paraphrase",
      "significance": "High/Medium/Low",
      "category": "e.g., Market Position, Financial, Innovation, etc."
    }}
  ]
}}

JSON Response:
"""
        
        response = self.generate_response(prompt)
        
        # Parse JSON response (with fallback)
        try:
            import json
            strengths = json.loads(response)
        except:
            strengths = {
                "items": [{
                    "strength": "Unable to parse structured response",
                    "evidence": response[:500],
                    "significance": "Unknown",
                    "category": "General"
                }]
            }
        
        return strengths
    
    def _analyze_weaknesses(self, doc_id: str, vector_store) -> Dict[str, Any]:
        """Analyze company weaknesses."""
        query = """
        What are the company's weaknesses, limitations, or areas of concern?
        Look for: operational challenges, financial constraints, competitive disadvantages,
        regulatory issues, dependency risks, resource limitations, market vulnerabilities.
        """
        
        relevant_chunks = self.retrieve_relevant_chunks(
            doc_id=doc_id,
            query=query,
            k=5,
            vector_store=vector_store
        )
        
        context = "\n\n".join([c['text'][:800] for c in relevant_chunks])
        
        prompt = f"""
Analyze the following excerpts from a SEC filing and identify the company's WEAKNESSES.

Context:
{context}

Please identify 3-5 key weaknesses or limitations. For each weakness:
1. State the weakness clearly
2. Provide supporting evidence from the text
3. Rate the severity (High/Medium/Low)

Format as JSON:
{{
  "items": [
    {{
      "weakness": "Clear statement",
      "evidence": "Supporting quote or paraphrase",
      "severity": "High/Medium/Low",
      "category": "e.g., Financial, Operational, Competitive, etc."
    }}
  ]
}}

JSON Response:
"""
        
        response = self.generate_response(prompt)
        
        try:
            import json
            weaknesses = json.loads(response)
        except:
            weaknesses = {
                "items": [{
                    "weakness": "Unable to parse structured response",
                    "evidence": response[:500],
                    "severity": "Unknown",
                    "category": "General"
                }]
            }
        
        return weaknesses
    
    def _analyze_opportunities(self, doc_id: str, vector_store) -> Dict[str, Any]:
        """Analyze market opportunities."""
        query = """
        What opportunities exist for the company to grow and improve?
        Look for: market expansion, new products/services, emerging technologies,
        strategic partnerships, regulatory changes, customer trends, innovation potential.
        """
        
        relevant_chunks = self.retrieve_relevant_chunks(
            doc_id=doc_id,
            query=query,
            k=5,
            vector_store=vector_store
        )
        
        context = "\n\n".join([c['text'][:800] for c in relevant_chunks])
        
        prompt = f"""
Analyze the following excerpts from a SEC filing and identify OPPORTUNITIES for the company.

Context:
{context}

Please identify 3-5 key opportunities. For each opportunity:
1. State the opportunity clearly
2. Provide supporting evidence from the text
3. Rate the potential impact (High/Medium/Low)

Format as JSON:
{{
  "items": [
    {{
      "opportunity": "Clear statement",
      "evidence": "Supporting quote or paraphrase",
      "potential": "High/Medium/Low",
      "category": "e.g., Market Expansion, Innovation, Partnership, etc."
    }}
  ]
}}

JSON Response:
"""
        
        response = self.generate_response(prompt)
        
        try:
            import json
            opportunities = json.loads(response)
        except:
            opportunities = {
                "items": [{
                    "opportunity": "Unable to parse structured response",
                    "evidence": response[:500],
                    "potential": "Unknown",
                    "category": "General"
                }]
            }
        
        return opportunities
    
    def _analyze_threats(self, doc_id: str, vector_store) -> Dict[str, Any]:
        """Analyze threats and risks."""
        query = """
        What threats and risks does the company face?
        Look for: competitive threats, market risks, regulatory risks, economic factors,
        technological disruption, supply chain vulnerabilities, legal issues, cybersecurity.
        """
        
        relevant_chunks = self.retrieve_relevant_chunks(
            doc_id=doc_id,
            query=query,
            k=5,
            vector_store=vector_store
        )
        
        context = "\n\n".join([c['text'][:800] for c in relevant_chunks])
        
        prompt = f"""
Analyze the following excerpts from a SEC filing and identify THREATS and risks to the company.

Context:
{context}

Please identify 3-5 key threats or risks. For each threat:
1. State the threat clearly
2. Provide supporting evidence from the text
3. Rate the severity (High/Medium/Low)

Format as JSON:
{{
  "items": [
    {{
      "threat": "Clear statement",
      "evidence": "Supporting quote or paraphrase",
      "severity": "High/Medium/Low",
      "category": "e.g., Competitive, Regulatory, Economic, etc."
    }}
  ]
}}

JSON Response:
"""
        
        response = self.generate_response(prompt)
        
        try:
            import json
            threats = json.loads(response)
        except:
            threats = {
                "items": [{
                    "threat": "Unable to parse structured response",
                    "evidence": response[:500],
                    "severity": "Unknown",
                    "category": "General"
                }]
            }
        
        return threats
    
    def _generate_swot_summary(
        self,
        strengths: Dict,
        weaknesses: Dict,
        opportunities: Dict,
        threats: Dict
    ) -> str:
        """Generate overall SWOT summary."""
        
        prompt = f"""
Based on the following SWOT analysis, provide a comprehensive summary (2-3 paragraphs):

STRENGTHS:
{self._format_items(strengths.get('items', []))}

WEAKNESSES:
{self._format_items(weaknesses.get('items', []))}

OPPORTUNITIES:
{self._format_items(opportunities.get('items', []))}

THREATS:
{self._format_items(threats.get('items', []))}

Summary should:
1. Highlight the most critical insights from each SWOT category
2. Identify strategic implications
3. Suggest how strengths can address weaknesses
4. Recommend how to leverage opportunities while mitigating threats

Summary:
"""
        
        return self.generate_response(prompt)
    
    def _format_items(self, items: List[Dict]) -> str:
        """Format SWOT items for summary prompt."""
        if not items:
            return "None identified"
        
        formatted = []
        for item in items:
            # Get the main content (first key that ends with specific words)
            content_key = None
            for key in ['strength', 'weakness', 'opportunity', 'threat']:
                if key in item:
                    content_key = key
                    break
            
            if content_key:
                formatted.append(f"- {item[content_key]}")
        
        return "\n".join(formatted) if formatted else "None identified"


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Perform SWOT analysis")
    parser.add_argument("--doc-id", required=True, help="Document ID")
    
    args = parser.parse_args()
    
    agent = SWOTAgent()
    result = agent.analyze(args.doc_id)
    
    print("\n" + "="*80)
    print("SWOT ANALYSIS")
    print("="*80)
    
    swot = result['swot_analysis']
    
    print("\n📈 STRENGTHS:")
    for item in swot['strengths'].get('items', []):
        print(f"  • {item.get('strength', 'N/A')}")
    
    print("\n📉 WEAKNESSES:")
    for item in swot['weaknesses'].get('items', []):
        print(f"  • {item.get('weakness', 'N/A')}")
    
    print("\n🚀 OPPORTUNITIES:")
    for item in swot['opportunities'].get('items', []):
        print(f"  • {item.get('opportunity', 'N/A')}")
    
    print("\n⚠️  THREATS:")
    for item in swot['threats'].get('items', []):
        print(f"  • {item.get('threat', 'N/A')}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(result['summary'])


if __name__ == "__main__":
    main()
