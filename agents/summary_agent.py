"""
Summary Agent

Generates executive summaries of SEC filings, highlighting:
- Key business developments
- Management discussion insights
- Major financial highlights
- Notable events and changes
"""

from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent
from utils import get_logger

logger = get_logger("summary_agent")


class SummaryAgent(BaseAgent):
    """Agent for generating executive summaries."""
    
    def __init__(self):
        """Initialize summary agent."""
        super().__init__(
            agent_name="Summary Agent",
            temperature=0.3  # Lower temperature for more focused summaries
        )
    
    def analyze(
        self,
        doc_id: str,
        max_length: int = 1000,
        focus_areas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate executive summary of the filing.
        
        Args:
            doc_id: Document identifier
            max_length: Maximum summary length in words
            focus_areas: Optional list of areas to focus on
        
        Returns:
            Dictionary containing summary and metadata
        """
        logger.info(f"Generating summary for {doc_id}")
        
        # Load document
        chunks = self.load_document(doc_id)
        
        # Identify key sections to summarize
        key_sections = [
            "Item 1",  # Business
            "Item 1A",  # Risk Factors
            "Item 7",  # Management's Discussion and Analysis
            "Item 8",  # Financial Statements
        ]
        
        # Create vector store for retrieval
        vector_store = self.create_vector_store(doc_id)
        
        # Generate summaries for each section
        section_summaries = {}
        
        for section in key_sections:
            section_chunks = [c for c in chunks if c.get('item', '').startswith(section)]
            
            if section_chunks:
                # Get most relevant chunks for this section
                relevant_chunks = self.retrieve_relevant_chunks(
                    doc_id=doc_id,
                    query=f"Summarize key points from {section}",
                    k=3,
                    vector_store=vector_store
                )
                
                # Generate section summary
                context = "\n\n".join([c['text'][:1000] for c in relevant_chunks])
                
                prompt = f"""
Please provide a concise summary (2-3 paragraphs) of the following content from {section}:

{context}

Focus on:
- Key business activities and strategies
- Important changes or developments
- Significant financial information
- Material risks or opportunities

Summary:
"""
                
                summary = self.generate_response(prompt)
                section_summaries[section] = summary
        
        # Generate overall executive summary
        overall_prompt = f"""
Based on the following section summaries from a SEC filing, create a comprehensive executive summary (3-5 paragraphs):

{self._format_section_summaries(section_summaries)}

The executive summary should:
1. Highlight the most important business developments
2. Summarize key financial performance indicators
3. Identify major risks and opportunities
4. Provide actionable insights for investors

Executive Summary:
"""
        
        executive_summary = self.generate_response(overall_prompt)
        
        # Compile results
        result = {
            "doc_id": doc_id,
            "agent": self.agent_name,
            "executive_summary": executive_summary,
            "section_summaries": section_summaries,
            "metadata": {
                "num_sections_analyzed": len(section_summaries),
                "total_chunks": len(chunks)
            }
        }
        
        # Save analysis
        self.save_analysis(doc_id, result, "summary_analysis")
        
        logger.info(f"✅ Summary generated for {doc_id}")
        return result
    
    def _format_section_summaries(self, summaries: Dict[str, str]) -> str:
        """Format section summaries for the overall summary prompt."""
        formatted = []
        for section, summary in summaries.items():
            formatted.append(f"### {section}\n{summary}\n")
        return "\n".join(formatted)
    
    def get_key_highlights(
        self,
        doc_id: str,
        num_highlights: int = 5
    ) -> List[str]:
        """
        Extract key highlights as bullet points.
        
        Args:
            doc_id: Document identifier
            num_highlights: Number of highlights to extract
        
        Returns:
            List of key highlights
        """
        # Create vector store
        vector_store = self.create_vector_store(doc_id)
        
        # Query for highlights
        queries = [
            "What are the most significant business developments?",
            "What are the key financial results?",
            "What are the major strategic initiatives?",
            "What are the most important risk factors?"
        ]
        
        highlights = []
        
        for query in queries:
            chunks = self.retrieve_relevant_chunks(
                doc_id=doc_id,
                query=query,
                k=2,
                vector_store=vector_store
            )
            
            if chunks:
                # Extract key point from most relevant chunk
                context = chunks[0]['text'][:500]
                
                prompt = f"""
Extract ONE key highlight (one sentence) from the following text that answers: {query}

Text: {context}

Highlight:
"""
                
                highlight = self.generate_response(prompt).strip()
                if highlight and len(highlight) < 200:
                    highlights.append(highlight)
        
        return highlights[:num_highlights]


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate filing summary")
    parser.add_argument("--doc-id", required=True, help="Document ID")
    
    args = parser.parse_args()
    
    agent = SummaryAgent()
    result = agent.analyze(args.doc_id)
    
    print("\n" + "="*80)
    print("EXECUTIVE SUMMARY")
    print("="*80)
    print(result['executive_summary'])
    print("\n" + "="*80)
    
    # Get key highlights
    highlights = agent.get_key_highlights(args.doc_id)
    print("\nKEY HIGHLIGHTS:")
    for i, highlight in enumerate(highlights, 1):
        print(f"{i}. {highlight}")


if __name__ == "__main__":
    main()
