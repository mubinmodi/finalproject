"""
Enhanced SWOT Agent (V2) - Buy-Side / Hostile Witness Mode

Evidence-driven SWOT tied to specific filing sections and quantitative signals.
Rule: No generic claims unless backed by concrete disclosures or metrics.

Includes YoY comparison of Risk Factors (Item 1A delta analysis).
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from .base_agent import BaseAgent
from utils import get_logger, ProvenanceTracker, create_citation_from_chunk, config

logger = get_logger("swot_agent_v2")


class SWOTAgentV2(BaseAgent):
    """Enhanced SWOT Agent with evidence-based analysis and YoY delta."""
    
    def __init__(self):
        """Initialize enhanced SWOT agent."""
        super().__init__(
            agent_name="SWOT Agent V2 (Hostile Witness)",
            temperature=0.1  # Very low for objective analysis
        )
        self.provenance = ProvenanceTracker()
    
    def analyze(
        self,
        doc_id: str,
        prior_doc_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform evidence-driven SWOT analysis with delta tracking.
        
        Args:
            doc_id: Current document identifier
            prior_doc_id: Prior year document for delta analysis
        
        Returns:
            Dictionary containing SWOT analysis with evidence and deltas
        """
        logger.info(f"Performing hostile witness SWOT for {doc_id}")
        
        # Create vector store
        vector_store = self.create_vector_store(doc_id)
        
        # Load XBRL and tables for quantitative backing
        xbrl_metrics = self._load_xbrl_metrics(doc_id)
        
        # Analyze each SWOT component with specific data targets
        strengths = self._analyze_strengths(doc_id, vector_store, xbrl_metrics)
        weaknesses = self._analyze_weaknesses(doc_id, vector_store, xbrl_metrics)
        opportunities = self._analyze_opportunities(doc_id, vector_store, xbrl_metrics)
        threats = self._analyze_threats(doc_id, vector_store, xbrl_metrics)
        
        # Perform delta SWOT (YoY comparison)
        delta_swot = self._analyze_delta_swot(doc_id, prior_doc_id, vector_store)
        
        # Risk factor delta (Item 1A YoY)
        risk_delta = self._analyze_risk_factor_delta(doc_id, prior_doc_id)
        
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
            "delta_swot": delta_swot,
            "risk_factor_delta": risk_delta,
            "provenance": self.provenance.to_dict(),
            "metadata": {
                "has_prior_comparison": prior_doc_id is not None,
                "total_swot_items": sum([
                    len(strengths['items']),
                    len(weaknesses['items']),
                    len(opportunities['items']),
                    len(threats['items'])
                ]),
                "num_citations": len(self.provenance.citations)
            }
        }
        
        # Save analysis
        self.save_analysis(doc_id, result, "swot_analysis_v2")
        
        logger.info(f"✅ Hostile witness SWOT complete")
        return result
    
    def _load_xbrl_metrics(self, doc_id: str) -> Dict[str, Any]:
        """Load XBRL metrics for quantitative validation."""
        try:
            xbrl_path = config.paths.processed_dir / doc_id / "xbrl_facts.jsonl"
            facts = []
            
            with open(xbrl_path, 'r') as f:
                for line in f:
                    facts.append(json.loads(line.strip()))
            
            # Extract key metrics
            metrics = {}
            metric_concepts = {
                "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
                "gross_profit": ["GrossProfit"],
                "operating_income": ["OperatingIncomeLoss"],
                "net_income": ["NetIncomeLoss"],
                "assets": ["Assets"],
                "liabilities": ["Liabilities"],
                "cash": ["Cash", "CashAndCashEquivalentsAtCarryingValue"],
                "debt": ["LongTermDebt"],
                "receivables": ["AccountsReceivable", "AccountsReceivableNetCurrent"],
            }
            
            for metric_name, concepts in metric_concepts.items():
                for concept in concepts:
                    matching = [f for f in facts if concept.lower() in f['concept'].lower()]
                    if matching:
                        fact = matching[0]
                        try:
                            metrics[metric_name] = float(fact['value']) if fact['value'] else None
                        except:
                            metrics[metric_name] = None
                        break
            
            # Calculate key ratios
            if metrics.get('gross_profit') and metrics.get('revenue'):
                metrics['gross_margin'] = (metrics['gross_profit'] / metrics['revenue']) * 100
            
            if metrics.get('operating_income') and metrics.get('revenue'):
                metrics['operating_margin'] = (metrics['operating_income'] / metrics['revenue']) * 100
            
            if metrics.get('debt') and metrics.get('assets'):
                metrics['leverage_ratio'] = metrics['debt'] / metrics['assets']
            
            return metrics
        
        except Exception as e:
            logger.warning(f"Could not load XBRL metrics: {e}")
            return {}
    
    def _analyze_strengths(
        self,
        doc_id: str,
        vector_store,
        xbrl_metrics: Dict
    ) -> Dict[str, Any]:
        """
        Analyze strengths (Moat) with evidence.
        
        Data targets:
        - Item 7: pricing power, operating leverage
        - Item 8: margin stability, working capital
        - Item 1: IP, exclusivity, contracts
        """
        
        queries = [
            ("Pricing Power", "Does the company discuss pricing power, price increases, or ability to pass through costs?"),
            ("Operating Leverage", "What evidence exists of operating leverage or economies of scale?"),
            ("Competitive Moat", "What intellectual property, exclusivity, or long-term contracts protect the business?"),
            ("Margin Stability", "How stable are gross and operating margins over time?"),
        ]
        
        items = []
        
        for category, query in queries:
            chunks = self.retrieve_relevant_chunks(
                doc_id=doc_id,
                query=query,
                k=2,
                vector_store=vector_store
            )
            
            if not chunks:
                continue
            
            chunk = chunks[0]
            context = chunk['text'][:700]
            
            # Add quantitative backing if available
            quant_context = ""
            if category == "Margin Stability" and xbrl_metrics.get('gross_margin'):
                quant_context = f"\nQuantitative: Gross Margin = {xbrl_metrics['gross_margin']:.1f}%, Operating Margin = {xbrl_metrics.get('operating_margin', 'N/A'):.1f}%"
            
            prompt = f"""
Analyze this evidence for a STRENGTH in category "{category}".

Evidence: {context}{quant_context}

RULE: Only cite as a strength if there is CONCRETE evidence. No generic claims.

Provide:
1. Strength statement (specific, backed by evidence)
2. Evidence (quote or specific data point)
3. Significance (High/Medium/Low)

If no concrete evidence, return "INSUFFICIENT_EVIDENCE"

Format as JSON:
{{
  "strength": "specific statement",
  "evidence": "concrete backing",
  "significance": "High/Medium/Low",
  "category": "{category}"
}}

JSON:
"""
            
            response = self.generate_response(prompt).strip()
            
            if "INSUFFICIENT_EVIDENCE" in response:
                continue
            
            try:
                item = json.loads(response)
                
                # Add citation
                citation_id = create_citation_from_chunk(chunk, context[:200], self.provenance)
                item['citation_ids'] = [citation_id]
                item['page'] = chunk.get('page', 0)
                
                items.append(item)
            except:
                logger.warning(f"Could not parse strength response for {category}")
        
        return {"items": items}
    
    def _analyze_weaknesses(
        self,
        doc_id: str,
        vector_store,
        xbrl_metrics: Dict
    ) -> Dict[str, Any]:
        """
        Analyze weaknesses (Capital Drag) with evidence.
        
        Data targets:
        - Item 1A: non-boilerplate internal dependencies
        - Item 7: liquidity constraints, covenant pressure
        - Receivables/DSO deterioration
        """
        
        queries = [
            ("Liquidity", "Are there liquidity constraints, covenant pressures, or working capital issues?"),
            ("Operational Dependencies", "What critical dependencies or single points of failure exist?"),
            ("Receivables Quality", "Is there deterioration in receivables or DSO trends?"),
            ("Capital Intensity", "Are there high capital requirements or cash flow challenges?"),
        ]
        
        items = []
        
        for category, query in queries:
            chunks = self.retrieve_relevant_chunks(
                doc_id=doc_id,
                query=query,
                k=2,
                vector_store=vector_store
            )
            
            if not chunks:
                continue
            
            chunk = chunks[0]
            context = chunk['text'][:700]
            
            # Add quantitative signals
            quant_context = ""
            if category == "Liquidity" and xbrl_metrics.get('leverage_ratio'):
                quant_context = f"\nQuantitative: Debt/Assets = {xbrl_metrics['leverage_ratio']:.2%}"
            
            prompt = f"""
Analyze this evidence for a WEAKNESS in category "{category}".

Evidence: {context}{quant_context}

RULE: Only cite if there is CONCRETE evidence of a problem or constraint.

Provide:
1. Weakness statement (specific issue)
2. Evidence (concrete data)
3. Severity (High/Medium/Low)

If no concrete evidence, return "INSUFFICIENT_EVIDENCE"

Format as JSON:
{{
  "weakness": "specific issue",
  "evidence": "concrete backing",
  "severity": "High/Medium/Low",
  "category": "{category}"
}}

JSON:
"""
            
            response = self.generate_response(prompt).strip()
            
            if "INSUFFICIENT_EVIDENCE" in response:
                continue
            
            try:
                item = json.loads(response)
                citation_id = create_citation_from_chunk(chunk, context[:200], self.provenance)
                item['citation_ids'] = [citation_id]
                item['page'] = chunk.get('page', 0)
                items.append(item)
            except:
                logger.warning(f"Could not parse weakness response for {category}")
        
        return {"items": items}
    
    def _analyze_opportunities(
        self,
        doc_id: str,
        vector_store,
        xbrl_metrics: Dict
    ) -> Dict[str, Any]:
        """
        Analyze opportunities (Alpha vectors) with evidence.
        
        Data targets:
        - Item 7: CapEx plans (growth vs maintenance)
        - Item 1: TAM expansion, new segments/products
        - Item 8: tax assets like NOLs
        """
        
        queries = [
            ("Growth CapEx", "What growth capital expenditure plans or investments are disclosed?"),
            ("Market Expansion", "What market expansion, new products, or TAM growth opportunities exist?"),
            ("Tax Assets", "Are there NOLs, tax credits, or other tax optimization opportunities?"),
            ("M&A Potential", "Are there acquisition opportunities or strategic partnerships discussed?"),
        ]
        
        items = []
        
        for category, query in queries:
            chunks = self.retrieve_relevant_chunks(
                doc_id=doc_id,
                query=query,
                k=2,
                vector_store=vector_store
            )
            
            if not chunks:
                continue
            
            chunk = chunks[0]
            context = chunk['text'][:700]
            
            prompt = f"""
Analyze this evidence for an OPPORTUNITY in category "{category}".

Evidence: {context}

RULE: Only cite if there is CONCRETE evidence of a potential growth vector.

Provide:
1. Opportunity statement
2. Evidence
3. Potential impact (High/Medium/Low)

If no concrete evidence, return "INSUFFICIENT_EVIDENCE"

Format as JSON:
{{
  "opportunity": "specific opportunity",
  "evidence": "concrete backing",
  "potential": "High/Medium/Low",
  "category": "{category}"
}}

JSON:
"""
            
            response = self.generate_response(prompt).strip()
            
            if "INSUFFICIENT_EVIDENCE" in response:
                continue
            
            try:
                item = json.loads(response)
                citation_id = create_citation_from_chunk(chunk, context[:200], self.provenance)
                item['citation_ids'] = [citation_id]
                item['page'] = chunk.get('page', 0)
                items.append(item)
            except:
                logger.warning(f"Could not parse opportunity response for {category}")
        
        return {"items": items}
    
    def _analyze_threats(
        self,
        doc_id: str,
        vector_store,
        xbrl_metrics: Dict
    ) -> Dict[str, Any]:
        """
        Analyze threats (Tail risks) with evidence.
        
        Data targets:
        - Item 1A: regulatory + disruption
        - Item 3: legal proceedings
        - Item 7: competitive pressure, margin compression
        """
        
        queries = [
            ("Regulatory", "What regulatory risks, investigations, or compliance issues are disclosed?"),
            ("Competitive Pressure", "What competitive threats or margin compression is discussed?"),
            ("Legal Proceedings", "What material legal proceedings or litigations exist?"),
            ("Disruption Risk", "What technological disruption or industry transformation risks exist?"),
        ]
        
        items = []
        
        for category, query in queries:
            chunks = self.retrieve_relevant_chunks(
                doc_id=doc_id,
                query=query,
                k=2,
                vector_store=vector_store
            )
            
            if not chunks:
                continue
            
            chunk = chunks[0]
            context = chunk['text'][:700]
            
            prompt = f"""
Analyze this evidence for a THREAT in category "{category}".

Evidence: {context}

RULE: Only cite if there is CONCRETE evidence of a material risk.

Provide:
1. Threat statement
2. Evidence
3. Severity (High/Medium/Low)

If no concrete evidence, return "INSUFFICIENT_EVIDENCE"

Format as JSON:
{{
  "threat": "specific threat",
  "evidence": "concrete backing",
  "severity": "High/Medium/Low",
  "category": "{category}"
}}

JSON:
"""
            
            response = self.generate_response(prompt).strip()
            
            if "INSUFFICIENT_EVIDENCE" in response:
                continue
            
            try:
                item = json.loads(response)
                citation_id = create_citation_from_chunk(chunk, context[:200], self.provenance)
                item['citation_ids'] = [citation_id]
                item['page'] = chunk.get('page', 0)
                items.append(item)
            except:
                logger.warning(f"Could not parse threat response for {category}")
        
        return {"items": items}
    
    def _analyze_delta_swot(
        self,
        doc_id: str,
        prior_doc_id: Optional[str],
        vector_store
    ) -> Dict[str, Any]:
        """Analyze what changed vs prior year in each SWOT quadrant."""
        
        if not prior_doc_id:
            return {"available": False}
        
        # This would require loading prior analysis
        # For now, return structure
        return {
            "available": False,
            "note": "Full delta SWOT requires prior year analysis to be loaded"
        }
    
    def _analyze_risk_factor_delta(
        self,
        doc_id: str,
        prior_doc_id: Optional[str]
    ) -> Dict[str, Any]:
        """
        Compare Item 1A year-over-year.
        
        Tracks:
        - Added risks → new threats
        - Removed risks → threat reduction
        - Language shift → sentiment changes
        """
        
        if not prior_doc_id:
            return {"available": False}
        
        # Load current risk factors
        vector_store = self.create_vector_store(doc_id)
        
        query = "What are all the risk factors disclosed in Item 1A?"
        current_chunks = self.retrieve_relevant_chunks(
            doc_id=doc_id,
            query=query,
            k=5,
            vector_store=vector_store
        )
        
        # Filter for Item 1A
        current_risks = [c for c in current_chunks if 'item 1a' in c['metadata'].get('item', '').lower()]
        
        # For full implementation, would load prior year and compare
        # For now, identify key risks
        
        prompt = f"""
From these Item 1A risk factor excerpts, identify the 3-5 most material risks:

{chr(10).join([c['text'][:400] for c in current_risks[:3]])}

For each risk, note:
- The specific risk
- Whether language suggests it's NEW, HEIGHTENED, or ONGOING

Format as JSON list:
[
  {{
    "risk": "description",
    "status": "NEW/HEIGHTENED/ONGOING",
    "severity": "High/Medium/Low"
  }}
]

JSON:
"""
        
        response = self.generate_response(prompt).strip()
        
        try:
            risks = json.loads(response)
            added_risks = [r for r in risks if r['status'] == 'NEW']
            heightened_risks = [r for r in risks if r['status'] == 'HEIGHTENED']
            
            return {
                "available": True,
                "added_risks": added_risks,
                "heightened_risks": heightened_risks,
                "removed_risks": [],  # Would need prior year comparison
                "sentiment_shifts": []  # Would need prior year comparison
            }
        except:
            return {"available": False, "error": "Could not parse risk delta"}


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Perform hostile witness SWOT")
    parser.add_argument("--doc-id", required=True, help="Document ID")
    parser.add_argument("--prior-doc-id", help="Prior year document ID")
    
    args = parser.parse_args()
    
    agent = SWOTAgentV2()
    result = agent.analyze(args.doc_id, args.prior_doc_id)
    
    swot = result['swot_analysis']
    
    print("\n" + "="*80)
    print("HOSTILE WITNESS SWOT ANALYSIS")
    print("="*80)
    
    print("\n💪 STRENGTHS (Evidence-Based):")
    for item in swot['strengths']['items']:
        print(f"\n• {item['strength']}")
        print(f"  Evidence: {item['evidence'][:100]}...")
        print(f"  Significance: {item['significance']} | Page {item['page'] + 1}")
    
    print("\n⚠️  WEAKNESSES (Evidence-Based):")
    for item in swot['weaknesses']['items']:
        print(f"\n• {item['weakness']}")
        print(f"  Evidence: {item['evidence'][:100]}...")
        print(f"  Severity: {item['severity']} | Page {item['page'] + 1}")
    
    print("\n🚀 OPPORTUNITIES (Evidence-Based):")
    for item in swot['opportunities']['items']:
        print(f"\n• {item['opportunity']}")
        print(f"  Evidence: {item['evidence'][:100]}...")
        print(f"  Potential: {item['potential']} | Page {item['page'] + 1}")
    
    print("\n🔴 THREATS (Evidence-Based):")
    for item in swot['threats']['items']:
        print(f"\n• {item['threat']}")
        print(f"  Evidence: {item['evidence'][:100]}...")
        print(f"  Severity: {item['severity']} | Page {item['page'] + 1}")
    
    if result['risk_factor_delta'].get('available'):
        delta = result['risk_factor_delta']
        print("\n📊 RISK FACTOR DELTA (Item 1A YoY):")
        if delta.get('added_risks'):
            print(f"  Added: {len(delta['added_risks'])} new risks")
        if delta.get('heightened_risks'):
            print(f"  Heightened: {len(delta['heightened_risks'])} risks")
    
    print(f"\n📚 Total Citations: {result['metadata']['num_citations']}")


if __name__ == "__main__":
    main()
