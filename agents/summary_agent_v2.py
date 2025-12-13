"""
Enhanced Summary Agent (V2) - Executive Brief

Focus on:
- Key findings (5-10 bullets)
- Direction of company (1 paragraph)
- What changed vs last year (delta analysis)
- Source citations with provenance
"""

import json
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent
from utils import get_logger, ProvenanceTracker, create_citation_from_chunk

logger = get_logger("summary_agent_v2")


class SummaryAgentV2(BaseAgent):
    """Enhanced Summary Agent with delta analysis and provenance."""
    
    def __init__(self):
        """Initialize enhanced summary agent."""
        super().__init__(
            agent_name="Summary Agent V2",
            temperature=0.2  # Lower temperature for more focused summaries
        )
        self.provenance = ProvenanceTracker()
    
    def analyze(
        self,
        doc_id: str,
        prior_doc_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate executive brief with delta analysis.
        
        Args:
            doc_id: Current document identifier
            prior_doc_id: Prior year document identifier for delta analysis
        
        Returns:
            Dictionary containing executive brief with provenance
        """
        logger.info(f"Generating executive brief for {doc_id}")
        
        # Create vector store
        vector_store = self.create_vector_store(doc_id)
        
        # Extract key findings from priority sections
        key_findings = self._extract_key_findings(doc_id, vector_store)
        
        # Extract company direction
        direction = self._extract_company_direction(doc_id, vector_store)
        
        # Extract delta vs prior year
        delta = self._extract_delta_analysis(doc_id, vector_store, prior_doc_id)
        
        # Extract financial highlights
        financial_highlights = self._extract_financial_highlights(doc_id)
        
        # Extract new risk factors
        new_risks = self._extract_new_risks(doc_id, prior_doc_id)
        
        # Compile results
        result = {
            "doc_id": doc_id,
            "agent": self.agent_name,
            "executive_brief": {
                "key_findings": key_findings,
                "company_direction": direction,
                "delta_analysis": delta,
                "financial_highlights": financial_highlights,
                "new_risks": new_risks
            },
            "provenance": self.provenance.to_dict(),
            "metadata": {
                "has_prior_comparison": prior_doc_id is not None,
                "num_findings": len(key_findings),
                "num_citations": len(self.provenance.citations)
            }
        }
        
        # Save analysis
        self.save_analysis(doc_id, result, "summary_analysis_v2")
        
        logger.info(f"✅ Executive brief generated with {len(key_findings)} findings")
        return result
    
    def _extract_key_findings(
        self,
        doc_id: str,
        vector_store
    ) -> List[Dict[str, Any]]:
        """Extract 5-10 key findings from priority sections."""
        
        # Priority queries for key findings
        queries = [
            ("Item 1: Business", "What are the key business strategies, market positioning, and competitive advantages?"),
            ("Item 7: Operations", "What were the most significant operational and financial results this period?"),
            ("Item 7: Outlook", "What is management's outlook and future guidance?"),
            ("Item 1A: Risks", "What are the most material new or changed risk factors?"),
        ]
        
        findings = []
        
        for section_label, query in queries:
            chunks = self.retrieve_relevant_chunks(
                doc_id=doc_id,
                query=query,
                k=2,
                vector_store=vector_store
            )
            
            if chunks:
                # Get most relevant chunk
                chunk = chunks[0]
                context = chunk['text'][:800]
                
                prompt = f"""
Extract ONE key finding from this {section_label} excerpt. Be specific and quantitative when possible.

Context: {context}

Format as a single bullet point (one sentence) that highlights what matters most.

Finding:
"""
                
                finding_text = self.generate_response(prompt).strip()
                
                # Create citation
                citation_id = create_citation_from_chunk(
                    chunk=chunk,
                    text_excerpt=context[:200],
                    tracker=self.provenance
                )
                
                findings.append({
                    "finding": finding_text,
                    "section": section_label,
                    "citation_ids": [citation_id],
                    "page": chunk['metadata']['page']
                })
        
        # Limit to top 10
        return findings[:10]
    
    def _extract_company_direction(
        self,
        doc_id: str,
        vector_store
    ) -> Dict[str, Any]:
        """Extract company direction (1 paragraph)."""
        
        query = "What is the company's strategic direction, key initiatives, and management priorities?"
        
        chunks = self.retrieve_relevant_chunks(
            doc_id=doc_id,
            query=query,
            k=3,
            vector_store=vector_store
        )
        
        context = "\n\n".join([c['text'][:600] for c in chunks])
        
        prompt = f"""
Based on the following excerpts, write ONE paragraph (4-5 sentences) describing the company's strategic direction:

{context}

Focus on:
- Core strategic priorities
- Key growth initiatives
- Market positioning
- Capital allocation strategy

Direction (1 paragraph):
"""
        
        direction_text = self.generate_response(prompt).strip()
        
        # Create citations
        citation_ids = [
            create_citation_from_chunk(chunk, chunk['text'][:200], self.provenance)
            for chunk in chunks
        ]
        
        return {
            "text": direction_text,
            "citation_ids": citation_ids
        }
    
    def _extract_delta_analysis(
        self,
        doc_id: str,
        vector_store,
        prior_doc_id: Optional[str]
    ) -> Dict[str, Any]:
        """Extract what changed vs last year."""
        
        if not prior_doc_id:
            return {
                "available": False,
                "note": "Prior year filing not provided for comparison"
            }
        
        # Load prior year chunks for comparison
        try:
            prior_chunks = self.load_document(prior_doc_id)
        except:
            return {
                "available": False,
                "note": "Prior year filing not processed"
            }
        
        # Query for changes
        query = "What significant changes, new initiatives, or strategic shifts occurred this year?"
        
        current_chunks = self.retrieve_relevant_chunks(
            doc_id=doc_id,
            query=query,
            k=3,
            vector_store=vector_store
        )
        
        context = "\n\n".join([c['text'][:500] for c in current_chunks])
        
        prompt = f"""
Based on the following context, identify key changes vs prior year:

{context}

List 3-5 specific changes in bullet format:
- New business lines or products
- Strategic pivots or discontinuations
- Major acquisitions or divestitures
- Organizational changes
- Market expansion/contraction

Changes:
"""
        
        changes_text = self.generate_response(prompt).strip()
        
        # Parse into list
        changes = [line.strip('- ').strip() for line in changes_text.split('\n') if line.strip().startswith('-')]
        
        citation_ids = [
            create_citation_from_chunk(chunk, chunk['text'][:200], self.provenance)
            for chunk in current_chunks
        ]
        
        return {
            "available": True,
            "changes": changes,
            "citation_ids": citation_ids
        }
    
    def _extract_financial_highlights(self, doc_id: str) -> Dict[str, Any]:
        """Extract financial statement headlines."""
        
        # Load XBRL facts
        try:
            xbrl_path = config.paths.processed_dir / doc_id / "xbrl_facts.jsonl"
            facts = []
            
            with open(xbrl_path, 'r') as f:
                for line in f:
                    facts.append(json.loads(line.strip()))
            
            # Extract key metrics
            metrics_map = {
                "Revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
                "Operating Income": ["OperatingIncomeLoss"],
                "Free Cash Flow": ["FreeCashFlow", "NetCashProvidedByUsedInOperatingActivities"]
            }
            
            highlights = []
            
            for metric_name, concepts in metrics_map.items():
                for concept in concepts:
                    matching = [f for f in facts if concept.lower() in f['concept'].lower()]
                    if matching:
                        fact = matching[0]
                        value = fact.get('value')
                        units = fact.get('units', '')
                        
                        if value:
                            highlights.append({
                                "metric": metric_name,
                                "value": value,
                                "units": units,
                                "source": "XBRL"
                            })
                        break
            
            return {
                "available": True,
                "metrics": highlights
            }
        
        except Exception as e:
            logger.warning(f"Could not extract financial highlights: {e}")
            return {
                "available": False,
                "metrics": []
            }
    
    def _extract_new_risks(
        self,
        doc_id: str,
        prior_doc_id: Optional[str]
    ) -> Dict[str, Any]:
        """Extract selected highlights from Risk Factors (what's new)."""
        
        vector_store = self.create_vector_store(doc_id)
        
        query = "What are the most significant risk factors disclosed?"
        
        chunks = self.retrieve_relevant_chunks(
            doc_id=doc_id,
            query=query,
            k=3,
            vector_store=vector_store
        )
        
        # Filter for Item 1A
        risk_chunks = [c for c in chunks if c['metadata'].get('item', '').startswith('Item 1A')]
        
        if not risk_chunks:
            risk_chunks = chunks[:2]  # Fallback
        
        prompt = f"""
From these risk factor excerpts, identify 2-3 most material risks:

{chr(10).join([c['text'][:400] for c in risk_chunks])}

Focus on:
- Non-boilerplate, company-specific risks
- Quantified or specific risks
- New or heightened risks

Risks (bullet format):
"""
        
        risks_text = self.generate_response(prompt).strip()
        risks = [line.strip('- ').strip() for line in risks_text.split('\n') if line.strip().startswith('-')]
        
        citation_ids = [
            create_citation_from_chunk(chunk, chunk['text'][:200], self.provenance)
            for chunk in risk_chunks
        ]
        
        return {
            "risks": risks,
            "citation_ids": citation_ids,
            "has_yoy_comparison": prior_doc_id is not None
        }


def main():
    """Example usage."""
    import argparse
    from utils import config
    
    parser = argparse.ArgumentParser(description="Generate executive brief")
    parser.add_argument("--doc-id", required=True, help="Document ID")
    parser.add_argument("--prior-doc-id", help="Prior year document ID for delta analysis")
    
    args = parser.parse_args()
    
    agent = SummaryAgentV2()
    result = agent.analyze(args.doc_id, args.prior_doc_id)
    
    brief = result['executive_brief']
    
    print("\n" + "="*80)
    print("EXECUTIVE BRIEF")
    print("="*80)
    
    print("\n📋 KEY FINDINGS:")
    for i, finding in enumerate(brief['key_findings'], 1):
        print(f"\n{i}. {finding['finding']}")
        print(f"   Source: {finding['section']}, Page {finding['page'] + 1}")
    
    print("\n🎯 COMPANY DIRECTION:")
    print(brief['company_direction']['text'])
    
    if brief['delta_analysis'].get('available'):
        print("\n📊 WHAT CHANGED VS LAST YEAR:")
        for change in brief['delta_analysis']['changes']:
            print(f"  • {change}")
    
    if brief['financial_highlights'].get('available'):
        print("\n💰 FINANCIAL HIGHLIGHTS:")
        for metric in brief['financial_highlights']['metrics']:
            print(f"  • {metric['metric']}: {metric['value']} {metric['units']}")
    
    print("\n⚠️  KEY RISKS:")
    for risk in brief['new_risks']['risks']:
        print(f"  • {risk}")
    
    print(f"\n📚 Total Citations: {result['metadata']['num_citations']}")


if __name__ == "__main__":
    main()
