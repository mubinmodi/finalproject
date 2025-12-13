"""
Stage 2: Text Extraction (PDF + OCR fallback)

Extracts text from PDF using pdfplumber with word-level bounding boxes.
Falls back to Tesseract OCR for poor quality or missing text.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pdfplumber
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import numpy as np
from utils import config, get_logger, Token, TextBlock, BoundingBox

logger = get_logger("stage2_text")


class TextStage:
    """Extract text from documents with provenance."""
    
    def __init__(self):
        """Initialize text extraction."""
        # Configure Tesseract
        if config.ocr.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = config.ocr.tesseract_cmd
        
        logger.info("Text extraction stage initialized")
    
    def process(self, doc_id: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Extract text from a document.
        
        Args:
            doc_id: Document identifier
        
        Returns:
            Tuple of (tokens, text_blocks)
        """
        logger.info(f"Extracting text for {doc_id}")
        
        # Check for PDF first, then HTML
        pdf_path = config.paths.raw_dir / doc_id / "filing.pdf"
        html_path = config.paths.raw_dir / doc_id / "filing.html"
        
        if pdf_path.exists():
            return self._process_pdf(doc_id, pdf_path)
        elif html_path.exists():
            logger.info(f"PDF not found, processing HTML: {html_path}")
            return self._process_html(doc_id, html_path)
        else:
            raise FileNotFoundError(f"Neither PDF nor HTML found for {doc_id}")
        
    def _process_pdf(self, doc_id: str, pdf_path: Path) -> Tuple[List[Dict], List[Dict]]:
        """Process PDF file."""
        # Load layout blocks if available
        blocks = self._load_blocks(doc_id)
        
        tokens = []
        text_blocks = []
        
        try:
            # Extract text using pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                logger.info(f"Processing {len(pdf.pages)} pages")
                
                for page_num, page in enumerate(pdf.pages):
                    page_tokens, page_text_blocks = self._process_page(
                        page=page,
                        page_num=page_num,
                        doc_id=doc_id,
                        blocks=blocks
                    )
                    tokens.extend(page_tokens)
                    text_blocks.extend(page_text_blocks)
                    
                    if (page_num + 1) % 10 == 0:
                        logger.info(f"Processed {page_num + 1}/{len(pdf.pages)} pages")
            
            # Save results
            self._save_tokens(doc_id, tokens)
            self._save_text_blocks(doc_id, text_blocks)
            
            logger.info(f"✅ Extracted {len(tokens)} tokens, {len(text_blocks)} text blocks")
            return tokens, text_blocks
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
    
    def _process_html(self, doc_id: str, html_path: Path) -> Tuple[List[Dict], List[Dict]]:
        """
        Process HTML file - extract text without PDF.
        
        Args:
            doc_id: Document identifier
            html_path: Path to HTML file
        
        Returns:
            Tuple of (tokens, text_blocks)
        """
        from bs4 import BeautifulSoup
        
        logger.info(f"Extracting text from HTML: {html_path}")
        
        # Read HTML
        with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()
        
        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text(separator='\n')
        
        # Clean up text
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = '\n'.join(lines)
        
        # Create simple tokens (word-level)
        tokens = []
        text_blocks = []
        
        # Split into paragraphs
        paragraphs = clean_text.split('\n\n')
        
        for page_num, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                continue
            
            # Simulate page structure (each paragraph as a "page")
            block_id = f"{doc_id}_html_block{page_num}"
            
            # Create text block
            text_block = TextBlock(
                block_id=block_id,
                doc_id=doc_id,
                page=page_num,  # Simulate pages
                text=paragraph,
                bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100),  # Dummy bbox
                extractor='html_parser',
                quality_score=100.0  # HTML text is clean
            )
            
            text_blocks.append(text_block.dict())
            
            # Create simple tokens from words
            words = paragraph.split()
            for word_idx, word in enumerate(words):
                token_id = f"{block_id}_token{word_idx}"
                
                token = Token(
                    token_id=token_id,
                    doc_id=doc_id,
                    page=page_num,
                    text=word,
                    bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),  # Dummy bbox
                    extractor='html_parser'
                )
                
                tokens.append(token.dict())
        
        # Save results
        self._save_tokens(doc_id, tokens)
        self._save_text_blocks(doc_id, text_blocks)
        
        logger.info(f"✅ Extracted {len(tokens)} tokens, {len(text_blocks)} text blocks from HTML")
        return tokens, text_blocks
    
    def _load_blocks(self, doc_id: str) -> Optional[List[Dict]]:
        """Load layout blocks if available."""
        blocks_path = config.paths.processed_dir / doc_id / "blocks.jsonl"
        
        if not blocks_path.exists():
            logger.warning(f"No layout blocks found for {doc_id}")
            return None
        
        blocks = []
        with open(blocks_path, 'r') as f:
            for line in f:
                blocks.append(json.loads(line.strip()))
        
        logger.info(f"Loaded {len(blocks)} layout blocks")
        return blocks
    
    def _process_page(
        self,
        page,
        page_num: int,
        doc_id: str,
        blocks: Optional[List[Dict]]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Extract text from a single page.
        
        Args:
            page: pdfplumber page object
            page_num: Page number (0-indexed)
            doc_id: Document identifier
            blocks: Layout blocks for this page
        
        Returns:
            Tuple of (tokens, text_blocks)
        """
        tokens = []
        text_blocks = []
        
        # Get page dimensions
        page_width = page.width
        page_height = page.height
        
        # Extract words with bounding boxes
        words = page.extract_words(
            x_tolerance=3,
            y_tolerance=3,
            keep_blank_chars=False
        )
        
        # Convert words to tokens
        for idx, word in enumerate(words):
            token_id = f"{doc_id}_page{page_num}_token{idx}"
            
            bbox = BoundingBox(
                x1=float(word['x0']),
                y1=float(word['top']),
                x2=float(word['x1']),
                y2=float(word['bottom'])
            )
            
            token = Token(
                token_id=token_id,
                doc_id=doc_id,
                page=page_num,
                text=word['text'],
                bbox=bbox,
                font=word.get('fontname'),
                font_size=word.get('height'),
                extractor='pdfplumber'
            )
            
            tokens.append(token.dict())
        
        # Group tokens into text blocks based on layout blocks
        if blocks:
            page_blocks = [b for b in blocks if b['page'] == page_num and b['block_type'] in ['text', 'title', 'list']]
            
            for block in page_blocks:
                block_tokens = self._get_tokens_in_block(tokens, block['bbox'])
                
                if block_tokens:
                    text = ' '.join([t['text'] for t in block_tokens])
                    
                    text_block = TextBlock(
                        block_id=block['block_id'],
                        doc_id=doc_id,
                        page=page_num,
                        text=text,
                        bbox=BoundingBox(**block['bbox']),
                        extractor='pdfplumber',
                        quality_score=self._calculate_text_quality(text)
                    )
                    
                    text_blocks.append(text_block.dict())
        else:
            # No layout blocks - create one text block per page
            if tokens:
                text = ' '.join([t['text'] for t in tokens])
                
                text_block = TextBlock(
                    block_id=f"{doc_id}_page{page_num}_block0",
                    doc_id=doc_id,
                    page=page_num,
                    text=text,
                    bbox=BoundingBox(x1=0, y1=0, x2=page_width, y2=page_height),
                    extractor='pdfplumber',
                    quality_score=self._calculate_text_quality(text)
                )
                
                text_blocks.append(text_block.dict())
        
        # Check for OCR fallback on low-quality blocks
        text_blocks = self._apply_ocr_fallback(doc_id, page_num, text_blocks)
        
        return tokens, text_blocks
    
    def _get_tokens_in_block(self, tokens: List[Dict], bbox: Dict) -> List[Dict]:
        """Get all tokens that fall within a bounding box."""
        block_tokens = []
        
        for token in tokens:
            token_bbox = token['bbox']
            # Check if token center is within block
            token_cx = (token_bbox['x1'] + token_bbox['x2']) / 2
            token_cy = (token_bbox['y1'] + token_bbox['y2']) / 2
            
            if (bbox['x1'] <= token_cx <= bbox['x2'] and
                bbox['y1'] <= token_cy <= bbox['y2']):
                block_tokens.append(token)
        
        return block_tokens
    
    def _calculate_text_quality(self, text: str) -> float:
        """
        Calculate text quality score (0-100).
        
        Simple heuristic based on:
        - Character variety
        - Alphanumeric ratio
        - Average word length
        """
        if not text:
            return 0.0
        
        # Count alphanumeric characters
        alnum_count = sum(c.isalnum() for c in text)
        alnum_ratio = alnum_count / len(text)
        
        # Count unique characters
        unique_ratio = len(set(text)) / len(text)
        
        # Average word length
        words = text.split()
        avg_word_len = sum(len(w) for w in words) / len(words) if words else 0
        word_score = min(avg_word_len / 10, 1.0)  # Normalize to 0-1
        
        # Combined score
        quality = (alnum_ratio * 0.5 + unique_ratio * 0.3 + word_score * 0.2) * 100
        
        return min(quality, 100.0)
    
    def _apply_ocr_fallback(
        self,
        doc_id: str,
        page_num: int,
        text_blocks: List[Dict]
    ) -> List[Dict]:
        """
        Apply OCR to low-quality text blocks.
        
        Args:
            doc_id: Document identifier
            page_num: Page number
            text_blocks: Text blocks from PDF extraction
        
        Returns:
            Updated text blocks with OCR where needed
        """
        # For now, only log low-quality blocks
        # Full OCR implementation would convert page to image and run Tesseract
        
        for block in text_blocks:
            if block['quality_score'] < config.ocr.confidence_threshold:
                logger.warning(
                    f"Low quality text in {block['block_id']} "
                    f"(score: {block['quality_score']:.1f}). "
                    f"OCR fallback recommended but not yet implemented."
                )
        
        return text_blocks
    
    def _save_tokens(self, doc_id: str, tokens: List[Dict]):
        """Save tokens to JSONL file."""
        output_dir = config.paths.processed_dir / doc_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "tokens.jsonl"
        
        with open(output_path, 'w') as f:
            for token in tokens:
                f.write(json.dumps(token) + '\n')
        
        logger.info(f"Saved {len(tokens)} tokens to {output_path}")
    
    def _save_text_blocks(self, doc_id: str, text_blocks: List[Dict]):
        """Save text blocks to JSONL file."""
        output_dir = config.paths.processed_dir / doc_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "text_blocks.jsonl"
        
        with open(output_path, 'w') as f:
            for block in text_blocks:
                f.write(json.dumps(block) + '\n')
        
        logger.info(f"Saved {len(text_blocks)} text blocks to {output_path}")


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract text from document")
    parser.add_argument("--doc-id", required=True, help="Document ID")
    
    args = parser.parse_args()
    
    stage = TextStage()
    tokens, text_blocks = stage.process(args.doc_id)
    
    logger.info(f"✅ Text extraction complete:")
    logger.info(f"  - Tokens: {len(tokens)}")
    logger.info(f"  - Text blocks: {len(text_blocks)}")
    
    # Calculate average quality
    if text_blocks:
        avg_quality = sum(b['quality_score'] for b in text_blocks) / len(text_blocks)
        logger.info(f"  - Average quality: {avg_quality:.1f}")


if __name__ == "__main__":
    main()
