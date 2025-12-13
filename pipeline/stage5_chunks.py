"""
Stage 5: Canonical Document Store (RAG-ready)

Creates sectioned Markdown documents and JSONL chunks with full provenance
for RAG retrieval and agent analysis.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from utils import config, get_logger, Chunk, BoundingBox

logger = get_logger("stage5_chunks")


class ChunkingStage:
    """Create RAG-ready document chunks with provenance."""
    
    # Common SEC filing section patterns
    ITEM_PATTERNS = [
        r'(?i)^item\s+(\d+[a-z]?)\.\s*(.+?)$',
        r'(?i)^part\s+([IVX]+)\s*[—-]\s*item\s+(\d+[a-z]?)\.\s*(.+?)$',
    ]
    
    def __init__(self):
        """Initialize chunking stage."""
        logger.info("Chunking stage initialized")
    
    def process(self, doc_id: str) -> Tuple[str, List[Dict]]:
        """
        Create markdown document and chunks from processed data.
        
        Args:
            doc_id: Document identifier
        
        Returns:
            Tuple of (markdown_content, chunks)
        """
        logger.info(f"Creating chunks for {doc_id}")
        
        # Load processed data
        manifest = self._load_manifest(doc_id)
        text_blocks = self._load_text_blocks(doc_id)
        tables = self._load_tables(doc_id)
        xbrl_facts = self._load_xbrl_facts(doc_id)
        
        # Create markdown document
        markdown = self._create_markdown(doc_id, manifest, text_blocks, tables, xbrl_facts)
        
        # Create chunks
        chunks = self._create_chunks(doc_id, text_blocks, tables)
        
        # Save outputs
        self._save_markdown(doc_id, markdown)
        self._save_chunks(doc_id, chunks)
        
        logger.info(f"✅ Created {len(chunks)} chunks")
        return markdown, chunks
    
    def _load_manifest(self, doc_id: str) -> Dict:
        """Load document manifest."""
        manifest_path = config.paths.raw_dir / doc_id / "manifest.json"
        
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        with open(manifest_path, 'r') as f:
            return json.load(f)
    
    def _load_text_blocks(self, doc_id: str) -> List[Dict]:
        """Load text blocks."""
        blocks_path = config.paths.processed_dir / doc_id / "text_blocks.jsonl"
        
        if not blocks_path.exists():
            logger.warning(f"No text blocks found for {doc_id}")
            return []
        
        blocks = []
        with open(blocks_path, 'r') as f:
            for line in f:
                blocks.append(json.loads(line.strip()))
        
        return blocks
    
    def _load_tables(self, doc_id: str) -> List[Dict]:
        """Load table metadata."""
        tables_path = config.paths.processed_dir / doc_id / "tables_index.jsonl"
        
        if not tables_path.exists():
            logger.warning(f"No tables found for {doc_id}")
            return []
        
        tables = []
        with open(tables_path, 'r') as f:
            for line in f:
                tables.append(json.loads(line.strip()))
        
        return tables
    
    def _load_xbrl_facts(self, doc_id: str) -> List[Dict]:
        """Load XBRL facts."""
        xbrl_path = config.paths.processed_dir / doc_id / "xbrl_facts.jsonl"
        
        if not xbrl_path.exists():
            logger.warning(f"No XBRL facts found for {doc_id}")
            return []
        
        facts = []
        with open(xbrl_path, 'r') as f:
            for line in f:
                facts.append(json.loads(line.strip()))
        
        return facts
    
    def _create_markdown(
        self,
        doc_id: str,
        manifest: Dict,
        text_blocks: List[Dict],
        tables: List[Dict],
        xbrl_facts: List[Dict]
    ) -> str:
        """
        Create a structured markdown document.
        
        Args:
            doc_id: Document identifier
            manifest: Document manifest
            text_blocks: List of text blocks
            tables: List of table metadata
            xbrl_facts: List of XBRL facts
        
        Returns:
            Markdown content
        """
        lines = []
        
        # Header
        lines.append(f"# {manifest['ticker']} - {manifest['form_type']}")
        lines.append(f"\n**Filing Date:** {manifest['filing_date']}")
        lines.append(f"**Accession Number:** {manifest['accession_number']}")
        lines.append(f"\n---\n")
        
        # Group text blocks by page and section
        current_page = -1
        current_section = None
        
        for block in sorted(text_blocks, key=lambda b: (b['page'], b['block_id'])):
            page = block['page']
            text = block['text'].strip()
            
            if not text:
                continue
            
            # Detect page boundaries
            if page != current_page:
                if current_page >= 0:
                    lines.append("\n")
                lines.append(f"## Page {page + 1}\n")
                current_page = page
            
            # Try to detect section headers (Items)
            section = self._detect_section(text)
            if section and section != current_section:
                lines.append(f"\n### {section}\n")
                current_section = section
            
            # Add text block
            lines.append(text)
            lines.append("\n")
        
        # Add tables section
        if tables:
            lines.append("\n---\n")
            lines.append("## Financial Tables\n")
            
            for table in tables:
                lines.append(f"\n### Table {table['table_id']} (Page {table['page'] + 1})")
                if table.get('caption'):
                    lines.append(f"**Caption:** {table['caption']}")
                lines.append(f"**Method:** {table['method']} (Quality: {table['quality_score']:.1f})")
                lines.append(f"**Dimensions:** {table['num_rows']} rows × {table['num_cols']} columns")
                lines.append(f"**CSV:** `{table['csv_path']}`\n")
        
        # Add XBRL summary
        if xbrl_facts:
            lines.append("\n---\n")
            lines.append("## XBRL Key Facts\n")
            
            # Get unique concepts
            concepts = {}
            for fact in xbrl_facts:
                concept = fact['concept']
                if concept not in concepts:
                    concepts[concept] = fact
            
            # Show top 20 concepts
            for idx, (concept, fact) in enumerate(list(concepts.items())[:20]):
                value_str = str(fact['value'])
                if fact.get('units'):
                    value_str += f" {fact['units']}"
                lines.append(f"- **{concept}:** {value_str}")
                
                if idx >= 19:
                    lines.append(f"\n*... and {len(concepts) - 20} more concepts*")
                    break
        
        return "\n".join(lines)
    
    def _detect_section(self, text: str) -> Optional[str]:
        """
        Detect SEC filing section headers (Items).
        
        Args:
            text: Text to analyze
        
        Returns:
            Section name or None
        """
        # Check first 200 characters for section headers
        header_text = text[:200].strip()
        
        for pattern in self.ITEM_PATTERNS:
            match = re.match(pattern, header_text, re.MULTILINE)
            if match:
                return header_text.split('\n')[0].strip()
        
        return None
    
    def _create_chunks(
        self,
        doc_id: str,
        text_blocks: List[Dict],
        tables: List[Dict]
    ) -> List[Dict]:
        """
        Create RAG-ready chunks from text blocks and tables.
        
        Args:
            doc_id: Document identifier
            text_blocks: List of text blocks
            tables: List of table metadata
        
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        chunk_id_counter = 0
        
        # Create chunks from text blocks
        for block in text_blocks:
            text = block['text'].strip()
            
            if len(text) < config.chunking.min_chunk_size:
                continue
            
            # Split long blocks into smaller chunks
            if len(text) > config.chunking.chunk_size:
                sub_chunks = self._split_text(text, config.chunking.chunk_size, config.chunking.chunk_overlap)
                
                for sub_text in sub_chunks:
                    chunk = Chunk(
                        chunk_id=f"{doc_id}_chunk{chunk_id_counter}",
                        doc_id=doc_id,
                        item=self._detect_section(text),
                        section=None,
                        page=block['page'],
                        bbox=BoundingBox(**block['bbox']) if block.get('bbox') else None,
                        text=sub_text,
                        extractor=block['extractor'],
                        source_path=str(config.paths.raw_dir / doc_id / "filing.pdf"),
                        metadata={
                            'block_id': block['block_id'],
                            'quality_score': block.get('quality_score')
                        }
                    )
                    
                    chunks.append(chunk.dict())
                    chunk_id_counter += 1
            else:
                chunk = Chunk(
                    chunk_id=f"{doc_id}_chunk{chunk_id_counter}",
                    doc_id=doc_id,
                    item=self._detect_section(text),
                    section=None,
                    page=block['page'],
                    bbox=BoundingBox(**block['bbox']) if block.get('bbox') else None,
                    text=text,
                    extractor=block['extractor'],
                    source_path=str(config.paths.raw_dir / doc_id / "filing.pdf"),
                    metadata={
                        'block_id': block['block_id'],
                        'quality_score': block.get('quality_score')
                    }
                )
                
                chunks.append(chunk.dict())
                chunk_id_counter += 1
        
        # Create chunks for tables (reference to CSV)
        for table in tables:
            # Create a text description of the table
            table_text = f"Financial Table: {table['table_id']}\n"
            if table.get('caption'):
                table_text += f"Caption: {table['caption']}\n"
            table_text += f"Dimensions: {table['num_rows']} rows × {table['num_cols']} columns\n"
            table_text += f"Extraction method: {table['method']} (quality: {table['quality_score']:.1f})\n"
            table_text += f"CSV data available at: {table['csv_path']}"
            
            chunk = Chunk(
                chunk_id=f"{doc_id}_chunk{chunk_id_counter}",
                doc_id=doc_id,
                item=None,
                section="Financial Tables",
                page=table['page'],
                bbox=BoundingBox(**table['bbox']) if table.get('bbox') else None,
                text=table_text,
                extractor="table_metadata",
                source_path=table['csv_path'],
                metadata={
                    'table_id': table['table_id'],
                    'num_rows': table['num_rows'],
                    'num_cols': table['num_cols'],
                    'method': table['method'],
                    'quality_score': table['quality_score']
                }
            )
            
            chunks.append(chunk.dict())
            chunk_id_counter += 1
        
        return chunks
    
    def _split_text(
        self,
        text: str,
        chunk_size: int,
        overlap: int
    ) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to split
            chunk_size: Maximum chunk size in characters
            overlap: Overlap between chunks in characters
        
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end within last 100 characters
                last_period = text[max(end - 100, start):end].rfind('. ')
                if last_period >= 0:
                    end = max(end - 100, start) + last_period + 2
            
            chunks.append(text[start:end].strip())
            start = end - overlap
        
        return chunks
    
    def _save_markdown(self, doc_id: str, markdown: str):
        """Save markdown document."""
        output_dir = config.paths.final_dir / doc_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "filing.md"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown)
        
        logger.info(f"Saved markdown to {output_path}")
    
    def _save_chunks(self, doc_id: str, chunks: List[Dict]):
        """Save chunks to JSONL file."""
        output_dir = config.paths.final_dir / doc_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "chunks.jsonl"
        
        with open(output_path, 'w') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + '\n')
        
        logger.info(f"Saved {len(chunks)} chunks to {output_path}")


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create document chunks")
    parser.add_argument("--doc-id", required=True, help="Document ID")
    
    args = parser.parse_args()
    
    stage = ChunkingStage()
    markdown, chunks = stage.process(args.doc_id)
    
    logger.info(f"✅ Chunking complete:")
    logger.info(f"  - Total chunks: {len(chunks)}")
    logger.info(f"  - Markdown size: {len(markdown)} characters")
    
    # Chunk statistics
    text_chunks = [c for c in chunks if c['extractor'] != 'table_metadata']
    table_chunks = [c for c in chunks if c['extractor'] == 'table_metadata']
    
    logger.info(f"  - Text chunks: {len(text_chunks)}")
    logger.info(f"  - Table chunks: {len(table_chunks)}")
    
    # Average chunk size
    if text_chunks:
        avg_size = sum(len(c['text']) for c in text_chunks) / len(text_chunks)
        logger.info(f"  - Average chunk size: {avg_size:.0f} characters")


if __name__ == "__main__":
    main()
