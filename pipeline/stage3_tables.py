"""
Stage 3: Table Extraction (CSV + provenance)

Extracts tables from PDFs using multiple methods (Camelot lattice/stream, pdfplumber)
and saves them as CSV with quality metrics and provenance.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import camelot
import pdfplumber
from utils import config, get_logger, TableMetadata, BoundingBox

logger = get_logger("stage3_tables")


class TableStage:
    """Extract tables from documents."""
    
    def __init__(self):
        """Initialize table extraction."""
        logger.info("Table extraction stage initialized")
    
    def process(self, doc_id: str) -> List[Dict]:
        """
        Extract tables from a document.
        
        Args:
            doc_id: Document identifier
        
        Returns:
            List of table metadata dictionaries
        """
        logger.info(f"Extracting tables for {doc_id}")
        
        pdf_path = config.paths.raw_dir / doc_id / "filing.pdf"
        html_path = config.paths.raw_dir / doc_id / "filing.html"
        
        # Check if we have a PDF
        if not pdf_path.exists():
            if html_path.exists():
                logger.warning(f"Table extraction requires PDF, but only HTML found. Skipping table extraction.")
                # Create empty tables index
                self._save_tables_index(doc_id, [])
                return []
            else:
                raise FileNotFoundError(f"Neither PDF nor HTML found for {doc_id}")
        
        # Load layout blocks to find table regions
        blocks = self._load_blocks(doc_id)
        
        tables_metadata = []
        
        try:
            # Method 1: Use layout blocks to guide extraction
            if blocks:
                table_blocks = [b for b in blocks if b['block_type'] == 'table']
                logger.info(f"Found {len(table_blocks)} table blocks from layout detection")
                
                for block in table_blocks:
                    metadata = self._extract_table_from_block(
                        pdf_path=pdf_path,
                        doc_id=doc_id,
                        block=block
                    )
                    if metadata:
                        tables_metadata.append(metadata)
            
            # Method 2: Fallback to full-page Camelot extraction
            else:
                logger.info("No layout blocks found, using full-page extraction")
                tables_metadata = self._extract_tables_full_page(
                    pdf_path=pdf_path,
                    doc_id=doc_id
                )
            
            # Save tables metadata
            self._save_tables_index(doc_id, tables_metadata)
            
            logger.info(f"✅ Extracted {len(tables_metadata)} tables")
            return tables_metadata
            
        except Exception as e:
            logger.error(f"Error extracting tables: {e}")
            raise
    
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
        
        return blocks
    
    def _extract_table_from_block(
        self,
        pdf_path: Path,
        doc_id: str,
        block: Dict
    ) -> Optional[Dict]:
        """
        Extract a table from a specific block region.
        
        Args:
            pdf_path: Path to PDF file
            doc_id: Document identifier
            block: Layout block dictionary
        
        Returns:
            Table metadata dictionary or None if extraction failed
        """
        page_num = block['page']
        bbox = block['bbox']
        
        # Try Camelot lattice first (for bordered tables)
        table_data, method, quality = self._try_camelot_extraction(
            pdf_path=pdf_path,
            page_num=page_num + 1,  # Camelot uses 1-indexed pages
            bbox=bbox,
            flavor='lattice'
        )
        
        # If lattice failed, try stream (for borderless tables)
        if table_data is None or quality < config.table.min_accuracy:
            logger.debug(f"Lattice failed or low quality, trying stream for {block['block_id']}")
            table_data, method, quality = self._try_camelot_extraction(
                pdf_path=pdf_path,
                page_num=page_num + 1,
                bbox=bbox,
                flavor='stream'
            )
        
        # Fallback to pdfplumber
        if table_data is None or quality < config.table.min_accuracy:
            logger.debug(f"Camelot failed, trying pdfplumber for {block['block_id']}")
            table_data, method, quality = self._try_pdfplumber_extraction(
                pdf_path=pdf_path,
                page_num=page_num,
                bbox=bbox
            )
        
        # If all methods failed, skip this table
        if table_data is None:
            logger.warning(f"Failed to extract table {block['block_id']}")
            return None
        
        # Save table as CSV
        table_id = f"{block['block_id']}_table"
        csv_path = self._save_table_csv(doc_id, table_id, table_data)
        
        # Create metadata
        metadata = TableMetadata(
            table_id=table_id,
            doc_id=doc_id,
            page=page_num,
            bbox=BoundingBox(**bbox),
            caption=None,  # TODO: Extract caption from nearby text
            method=method,
            quality_score=quality,
            csv_path=str(csv_path),
            num_rows=len(table_data),
            num_cols=len(table_data[0]) if table_data else 0
        )
        
        logger.debug(f"Extracted table {table_id} using {method} (quality: {quality:.1f})")
        return metadata.dict()
    
    def _try_camelot_extraction(
        self,
        pdf_path: Path,
        page_num: int,
        bbox: Dict,
        flavor: str
    ) -> Tuple[Optional[List[List[str]]], str, float]:
        """
        Try extracting table using Camelot.
        
        Args:
            pdf_path: Path to PDF
            page_num: Page number (1-indexed for Camelot)
            bbox: Bounding box dictionary
            flavor: 'lattice' or 'stream'
        
        Returns:
            Tuple of (table_data, method, quality_score)
        """
        try:
            # Convert bbox to Camelot format: "x1,y1,x2,y2"
            table_areas = [f"{bbox['x1']},{bbox['y1']},{bbox['x2']},{bbox['y2']}"]
            
            tables = camelot.read_pdf(
                str(pdf_path),
                pages=str(page_num),
                flavor=flavor,
                table_areas=table_areas,
                strip_text='\n',
                edge_tol=config.table.edge_tol,
                row_tol=config.table.row_tol,
                col_tol=config.table.col_tol
            )
            
            if len(tables) > 0:
                table = tables[0]
                df = table.df
                
                # Convert to list of lists
                table_data = df.values.tolist()
                
                # Quality score (Camelot's accuracy)
                quality = float(table.accuracy)
                
                method = f"camelot_{flavor}"
                return table_data, method, quality
            
            return None, f"camelot_{flavor}", 0.0
            
        except Exception as e:
            logger.debug(f"Camelot {flavor} extraction failed: {e}")
            return None, f"camelot_{flavor}", 0.0
    
    def _try_pdfplumber_extraction(
        self,
        pdf_path: Path,
        page_num: int,
        bbox: Dict
    ) -> Tuple[Optional[List[List[str]]], str, float]:
        """
        Try extracting table using pdfplumber.
        
        Args:
            pdf_path: Path to PDF
            page_num: Page number (0-indexed)
            bbox: Bounding box dictionary
        
        Returns:
            Tuple of (table_data, method, quality_score)
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_num >= len(pdf.pages):
                    return None, "pdfplumber", 0.0
                
                page = pdf.pages[page_num]
                
                # Crop page to bbox region
                cropped_page = page.crop((
                    bbox['x1'],
                    bbox['y1'],
                    bbox['x2'],
                    bbox['y2']
                ))
                
                # Extract table
                table = cropped_page.extract_table()
                
                if table:
                    # Calculate simple quality score based on completeness
                    total_cells = len(table) * len(table[0]) if table[0] else 0
                    filled_cells = sum(1 for row in table for cell in row if cell and cell.strip())
                    quality = (filled_cells / total_cells * 100) if total_cells > 0 else 0.0
                    
                    return table, "pdfplumber", quality
                
                return None, "pdfplumber", 0.0
                
        except Exception as e:
            logger.debug(f"pdfplumber extraction failed: {e}")
            return None, "pdfplumber", 0.0
    
    def _extract_tables_full_page(
        self,
        pdf_path: Path,
        doc_id: str
    ) -> List[Dict]:
        """
        Extract tables from entire PDF (no layout blocks).
        
        Args:
            pdf_path: Path to PDF
            doc_id: Document identifier
        
        Returns:
            List of table metadata
        """
        tables_metadata = []
        
        try:
            # Use Camelot on all pages
            for flavor in ['lattice', 'stream']:
                logger.info(f"Trying Camelot {flavor} on all pages")
                
                try:
                    tables = camelot.read_pdf(
                        str(pdf_path),
                        pages='all',
                        flavor=flavor,
                        strip_text='\n'
                    )
                    
                    for idx, table in enumerate(tables):
                        if table.accuracy < config.table.min_accuracy:
                            continue
                        
                        # Get table data
                        df = table.df
                        table_data = df.values.tolist()
                        
                        # Create table ID
                        table_id = f"{doc_id}_page{table.page-1}_table{idx}"
                        
                        # Save CSV
                        csv_path = self._save_table_csv(doc_id, table_id, table_data)
                        
                        # Get bounding box (Camelot provides this)
                        bbox_coords = table._bbox
                        bbox = BoundingBox(
                            x1=bbox_coords[0],
                            y1=bbox_coords[1],
                            x2=bbox_coords[2],
                            y2=bbox_coords[3]
                        )
                        
                        # Create metadata
                        metadata = TableMetadata(
                            table_id=table_id,
                            doc_id=doc_id,
                            page=table.page - 1,  # Convert to 0-indexed
                            bbox=bbox,
                            caption=None,
                            method=f"camelot_{flavor}",
                            quality_score=float(table.accuracy),
                            csv_path=str(csv_path),
                            num_rows=len(table_data),
                            num_cols=len(table_data[0]) if table_data else 0
                        )
                        
                        tables_metadata.append(metadata.dict())
                    
                except Exception as e:
                    logger.warning(f"Camelot {flavor} failed: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Full page table extraction failed: {e}")
        
        return tables_metadata
    
    def _save_table_csv(
        self,
        doc_id: str,
        table_id: str,
        table_data: List[List[str]]
    ) -> Path:
        """
        Save table data as CSV.
        
        Args:
            doc_id: Document identifier
            table_id: Table identifier
            table_data: Table data as list of lists
        
        Returns:
            Path to saved CSV file
        """
        tables_dir = config.paths.processed_dir / doc_id / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = tables_dir / f"{table_id}.csv"
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(table_data)
        
        return csv_path
    
    def _save_tables_index(self, doc_id: str, tables_metadata: List[Dict]):
        """Save tables index to JSONL file."""
        output_dir = config.paths.processed_dir / doc_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "tables_index.jsonl"
        
        with open(output_path, 'w') as f:
            for metadata in tables_metadata:
                f.write(json.dumps(metadata) + '\n')
        
        logger.info(f"Saved tables index to {output_path}")


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract tables from document")
    parser.add_argument("--doc-id", required=True, help="Document ID")
    
    args = parser.parse_args()
    
    stage = TableStage()
    tables = stage.process(args.doc_id)
    
    logger.info(f"✅ Table extraction complete:")
    logger.info(f"  - Total tables: {len(tables)}")
    
    # Group by method
    methods = {}
    for table in tables:
        method = table['method']
        methods[method] = methods.get(method, 0) + 1
    
    for method, count in sorted(methods.items()):
        logger.info(f"  - {method}: {count}")
    
    # Average quality
    if tables:
        avg_quality = sum(t['quality_score'] for t in tables) / len(tables)
        logger.info(f"  - Average quality: {avg_quality:.1f}")


if __name__ == "__main__":
    main()
