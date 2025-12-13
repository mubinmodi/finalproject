"""
Stage 1: Layout Detection + Block Routing

Uses LayoutParser with Detectron2 to detect document blocks
(text, title, table, figure, list) with bounding boxes.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import layoutparser as lp
from pdf2image import convert_from_path
import numpy as np
from utils import config, get_logger, Block, BoundingBox

logger = get_logger("stage1_layout")

# Fix model cache directory to avoid permission issues
cache_dir = config.paths.data_dir / ".model_cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ['TORCH_HOME'] = str(cache_dir)
os.environ['DETECTRON2_DATASETS'] = str(cache_dir)
logger.info(f"Model cache directory set to: {cache_dir}")


class LayoutStage:
    """Detect document layout and identify blocks."""
    
    def __init__(self):
        """Initialize layout detection model."""
        logger.info(f"Loading layout model: {config.layout.model_name}")
        
        try:
            self.model = lp.Detectron2LayoutModel(
                config.layout.model_name,
                extra_config=[
                    "MODEL.ROI_HEADS.SCORE_THRESH_TEST",
                    config.layout.confidence_threshold
                ],
                label_map=config.layout.label_map
            )
            logger.info("Layout model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading layout model: {e}")
            raise
    
    def process(self, doc_id: str, use_html: bool = False) -> List[Dict]:
        """
        Process a document and detect layout blocks.
        
        Args:
            doc_id: Document identifier
            use_html: If True, try to convert HTML to PDF first
        
        Returns:
            List of detected blocks
        """
        logger.info(f"Processing layout for {doc_id}")
        
        doc_dir = config.paths.raw_dir / doc_id
        
        # Find source file (PDF or HTML)
        pdf_path = doc_dir / "filing.pdf"
        html_path = doc_dir / "filing.html"
        
        if not pdf_path.exists() and not html_path.exists():
            raise FileNotFoundError(f"No PDF or HTML found for {doc_id}")
        
        # If only HTML exists, we need to convert it
        if not pdf_path.exists() and html_path.exists():
            logger.warning(f"PDF not found, HTML conversion not implemented yet")
            logger.warning(f"Skipping layout detection for {doc_id}")
            return []
        
        # Process PDF pages
        blocks = []
        
        try:
            # Convert PDF pages to images
            images = convert_from_path(str(pdf_path), dpi=150)
            logger.info(f"Processing {len(images)} pages")
            
            for page_num, image in enumerate(images):
                page_blocks = self._process_page(
                    image=np.array(image),
                    doc_id=doc_id,
                    page_num=page_num
                )
                blocks.extend(page_blocks)
                
                if (page_num + 1) % 10 == 0:
                    logger.info(f"Processed {page_num + 1}/{len(images)} pages")
            
            # Save blocks
            self._save_blocks(doc_id, blocks)
            
            logger.info(f"✅ Detected {len(blocks)} blocks across {len(images)} pages")
            return blocks
            
        except Exception as e:
            logger.error(f"Error processing layout: {e}")
            raise
    
    def _process_page(
        self,
        image: np.ndarray,
        doc_id: str,
        page_num: int
    ) -> List[Dict]:
        """
        Process a single page and detect blocks.
        
        Args:
            image: Page image as numpy array
            doc_id: Document identifier
            page_num: Page number (0-indexed)
        
        Returns:
            List of detected blocks for this page
        """
        # Run layout detection
        layout = self.model.detect(image)
        
        blocks = []
        for idx, block in enumerate(layout):
            block_id = f"{doc_id}_page{page_num}_block{idx}"
            
            # Extract bounding box
            bbox = BoundingBox(
                x1=float(block.coordinates[0]),
                y1=float(block.coordinates[1]),
                x2=float(block.coordinates[2]),
                y2=float(block.coordinates[3])
            )
            
            # Create Block object
            block_obj = Block(
                block_id=block_id,
                page=page_num,
                block_type=block.type,
                bbox=bbox,
                confidence=float(block.score) if hasattr(block, 'score') else None
            )
            
            blocks.append(block_obj.dict())
        
        return blocks
    
    def _save_blocks(self, doc_id: str, blocks: List[Dict]):
        """
        Save detected blocks to JSONL file.
        
        Args:
            doc_id: Document identifier
            blocks: List of block dictionaries
        """
        output_dir = config.paths.processed_dir / doc_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "blocks.jsonl"
        
        with open(output_path, 'w') as f:
            for block in blocks:
                f.write(json.dumps(block) + '\n')
        
        logger.info(f"Saved {len(blocks)} blocks to {output_path}")
    
    def load_blocks(self, doc_id: str) -> List[Dict]:
        """
        Load blocks from a processed document.
        
        Args:
            doc_id: Document identifier
        
        Returns:
            List of block dictionaries
        """
        blocks_path = config.paths.processed_dir / doc_id / "blocks.jsonl"
        
        if not blocks_path.exists():
            raise FileNotFoundError(f"Blocks file not found: {blocks_path}")
        
        blocks = []
        with open(blocks_path, 'r') as f:
            for line in f:
                blocks.append(json.loads(line.strip()))
        
        return blocks
    
    def get_blocks_by_type(self, doc_id: str, block_type: str) -> List[Dict]:
        """
        Get all blocks of a specific type.
        
        Args:
            doc_id: Document identifier
            block_type: Block type ('text', 'table', 'title', 'figure', 'list')
        
        Returns:
            List of matching blocks
        """
        all_blocks = self.load_blocks(doc_id)
        return [b for b in all_blocks if b['block_type'] == block_type]


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect document layout")
    parser.add_argument("--doc-id", required=True, help="Document ID")
    
    args = parser.parse_args()
    
    stage = LayoutStage()
    blocks = stage.process(args.doc_id)
    
    # Print summary
    block_types = {}
    for block in blocks:
        btype = block['block_type']
        block_types[btype] = block_types.get(btype, 0) + 1
    
    logger.info(f"✅ Layout detection complete:")
    for btype, count in sorted(block_types.items()):
        logger.info(f"  - {btype}: {count}")


if __name__ == "__main__":
    main()
