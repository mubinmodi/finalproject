"""
Provenance tracking for citations and source attribution.

Every piece of extracted information should be traceable to its source
with page numbers, bounding boxes, and extraction methods.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Citation for a piece of information."""
    
    text: str = Field(description="The cited text or data")
    page: int = Field(ge=0, description="Page number (0-indexed)")
    section: Optional[str] = Field(None, description="Section/Item label (e.g., 'Item 7')")
    bbox: Optional[Dict[str, float]] = Field(None, description="Bounding box coordinates")
    extraction_method: str = Field(description="Method used (pdfplumber, OCR, XBRL, camelot)")
    source_path: str = Field(description="Path to source file")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Extraction confidence")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ProvenanceTracker:
    """Track and manage citations throughout analysis."""
    
    def __init__(self):
        """Initialize provenance tracker."""
        self.citations: List[Citation] = []
    
    def add_citation(
        self,
        text: str,
        page: int,
        extraction_method: str,
        source_path: str,
        section: Optional[str] = None,
        bbox: Optional[Dict[str, float]] = None,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add a citation and return its ID.
        
        Returns:
            Citation ID (index in citations list)
        """
        citation = Citation(
            text=text,
            page=page,
            section=section,
            bbox=bbox,
            extraction_method=extraction_method,
            source_path=source_path,
            confidence=confidence,
            metadata=metadata
        )
        
        self.citations.append(citation)
        return len(self.citations) - 1
    
    def get_citation(self, citation_id: int) -> Optional[Citation]:
        """Get citation by ID."""
        if 0 <= citation_id < len(self.citations):
            return self.citations[citation_id]
        return None
    
    def get_citations_by_section(self, section: str) -> List[Citation]:
        """Get all citations from a specific section."""
        return [c for c in self.citations if c.section == section]
    
    def get_citations_by_page(self, page: int) -> List[Citation]:
        """Get all citations from a specific page."""
        return [c for c in self.citations if c.page == page]
    
    def format_citation(self, citation_id: int, style: str = "short") -> str:
        """
        Format citation for display.
        
        Args:
            citation_id: Citation ID
            style: 'short', 'medium', or 'full'
        
        Returns:
            Formatted citation string
        """
        citation = self.get_citation(citation_id)
        if not citation:
            return "[Citation not found]"
        
        if style == "short":
            section_str = f"{citation.section}, " if citation.section else ""
            return f"[{section_str}p.{citation.page + 1}]"
        
        elif style == "medium":
            section_str = f"{citation.section}, " if citation.section else ""
            return f"[{section_str}Page {citation.page + 1}, via {citation.extraction_method}]"
        
        else:  # full
            parts = []
            if citation.section:
                parts.append(f"Section: {citation.section}")
            parts.append(f"Page: {citation.page + 1}")
            parts.append(f"Method: {citation.extraction_method}")
            if citation.confidence:
                parts.append(f"Confidence: {citation.confidence:.2%}")
            return "[" + ", ".join(parts) + "]"
    
    def to_dict(self) -> List[Dict]:
        """Export all citations as list of dictionaries."""
        return [c.dict() for c in self.citations]
    
    def from_dict(self, citations_data: List[Dict]):
        """Import citations from list of dictionaries."""
        self.citations = [Citation(**c) for c in citations_data]


class AnalysisWithProvenance(BaseModel):
    """Base model for analysis results with provenance."""
    
    content: str = Field(description="Main analysis content")
    citations: List[int] = Field(default_factory=list, description="Citation IDs")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Overall confidence")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    def add_citation_ref(self, citation_id: int):
        """Add reference to a citation."""
        if citation_id not in self.citations:
            self.citations.append(citation_id)
    
    def format_with_citations(self, tracker: ProvenanceTracker, style: str = "short") -> str:
        """Format content with inline citations."""
        result = self.content
        
        if self.citations:
            citation_strs = [tracker.format_citation(cid, style) for cid in self.citations]
            result += " " + " ".join(citation_strs)
        
        return result


def extract_section_from_chunk(chunk: Dict) -> Optional[str]:
    """
    Extract section/item label from chunk metadata.
    
    Common patterns:
    - Item 1
    - Item 1A
    - Item 7
    - Part I
    - Part II
    """
    # Check item field
    if chunk.get('item'):
        return chunk['item']
    
    # Check section field
    if chunk.get('section'):
        return chunk['section']
    
    # Try to extract from text
    text = chunk.get('text', '')[:200]
    
    import re
    patterns = [
        r'(?i)^item\s+(\d+[a-z]?)',
        r'(?i)^part\s+([IVX]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0).strip()
    
    return None


def create_citation_from_chunk(
    chunk: Dict,
    text_excerpt: str,
    tracker: ProvenanceTracker
) -> int:
    """
    Create citation from a chunk and add to tracker.
    
    Args:
        chunk: Chunk dictionary with metadata
        text_excerpt: Specific text being cited
        tracker: ProvenanceTracker instance
    
    Returns:
        Citation ID
    """
    section = extract_section_from_chunk(chunk)
    
    return tracker.add_citation(
        text=text_excerpt,
        page=chunk.get('page', 0),
        section=section,
        bbox=chunk.get('bbox'),
        extraction_method=chunk.get('extractor', 'unknown'),
        source_path=chunk.get('source_path', ''),
        metadata=chunk.get('metadata')
    )
