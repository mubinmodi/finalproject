"""Validation utilities for pipeline outputs."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, validator


class BoundingBox(BaseModel):
    """Bounding box coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float
    
    @validator("x2")
    def x2_greater_than_x1(cls, v, values):
        if "x1" in values and v <= values["x1"]:
            raise ValueError("x2 must be greater than x1")
        return v
    
    @validator("y2")
    def y2_greater_than_y1(cls, v, values):
        if "y1" in values and v <= values["y1"]:
            raise ValueError("y2 must be greater than y1")
        return v


class Block(BaseModel):
    """Layout block from Stage 1."""
    block_id: str
    page: int = Field(ge=0)
    block_type: str
    bbox: BoundingBox
    confidence: Optional[float] = Field(None, ge=0, le=1)


class Token(BaseModel):
    """Text token with provenance from Stage 2."""
    token_id: str
    doc_id: str
    page: int = Field(ge=0)
    text: str
    bbox: BoundingBox
    font: Optional[str] = None
    font_size: Optional[float] = None
    extractor: str  # 'pdfplumber' or 'tesseract'


class TextBlock(BaseModel):
    """Aggregated text block from Stage 2."""
    block_id: str
    doc_id: str
    page: int = Field(ge=0)
    text: str
    bbox: BoundingBox
    extractor: str
    quality_score: Optional[float] = Field(None, ge=0, le=100)


class TableMetadata(BaseModel):
    """Table metadata from Stage 3."""
    table_id: str
    doc_id: str
    page: int = Field(ge=0)
    bbox: BoundingBox
    caption: Optional[str] = None
    method: str  # 'camelot_lattice', 'camelot_stream', 'pdfplumber'
    quality_score: Optional[float] = Field(None, ge=0, le=100)
    csv_path: str
    num_rows: Optional[int] = Field(None, ge=0)
    num_cols: Optional[int] = Field(None, ge=0)


class XBRLFact(BaseModel):
    """XBRL fact from Stage 4."""
    fact_id: str
    doc_id: str
    concept: str
    context: str
    period: str
    value: Any
    units: Optional[str] = None
    decimals: Optional[int] = None
    dimensions: Optional[Dict[str, str]] = None


class Chunk(BaseModel):
    """RAG chunk from Stage 5."""
    chunk_id: str
    doc_id: str
    item: Optional[str] = None
    section: Optional[str] = None
    page: int = Field(ge=0)
    bbox: Optional[BoundingBox] = None
    text: str = Field(min_length=1)
    extractor: str
    source_path: str
    metadata: Optional[Dict[str, Any]] = None


class Manifest(BaseModel):
    """Document manifest from Stage 0."""
    doc_id: str
    ticker: str
    form_type: str
    filing_date: str
    accession_number: str
    pdf_path: Optional[str] = None
    html_path: Optional[str] = None
    xbrl_dir: Optional[str] = None
    download_timestamp: str


def validate_jsonl_file(file_path: Path, model: BaseModel) -> List[Dict]:
    """
    Validate a JSONL file against a Pydantic model.
    
    Args:
        file_path: Path to JSONL file
        model: Pydantic model to validate against
    
    Returns:
        List of validated records
    
    Raises:
        ValueError: If validation fails
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    validated_records = []
    errors = []
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                validated = model(**data)
                validated_records.append(validated.dict())
            except Exception as e:
                errors.append(f"Line {line_num}: {str(e)}")
    
    if errors:
        raise ValueError(f"Validation errors in {file_path}:\n" + "\n".join(errors[:10]))
    
    return validated_records


def validate_pipeline_output(doc_id: str, stage: int, config) -> bool:
    """
    Validate outputs for a specific pipeline stage.
    
    Args:
        doc_id: Document identifier
        stage: Pipeline stage number (0-5)
        config: Configuration object
    
    Returns:
        True if validation passes
    
    Raises:
        ValueError: If validation fails
    """
    if stage == 0:
        # Stage 0: Check manifest exists
        manifest_path = config.paths.raw_dir / doc_id / "manifest.json"
        if not manifest_path.exists():
            raise ValueError(f"Manifest not found: {manifest_path}")
        with open(manifest_path) as f:
            Manifest(**json.load(f))
    
    elif stage == 1:
        # Stage 1: Validate blocks
        blocks_path = config.paths.processed_dir / doc_id / "blocks.jsonl"
        validate_jsonl_file(blocks_path, Block)
    
    elif stage == 2:
        # Stage 2: Validate tokens and text blocks
        tokens_path = config.paths.processed_dir / doc_id / "tokens.jsonl"
        text_blocks_path = config.paths.processed_dir / doc_id / "text_blocks.jsonl"
        validate_jsonl_file(tokens_path, Token)
        validate_jsonl_file(text_blocks_path, TextBlock)
    
    elif stage == 3:
        # Stage 3: Validate tables index
        tables_index_path = config.paths.processed_dir / doc_id / "tables_index.jsonl"
        validate_jsonl_file(tables_index_path, TableMetadata)
    
    elif stage == 4:
        # Stage 4: Validate XBRL facts
        xbrl_path = config.paths.processed_dir / doc_id / "xbrl_facts.jsonl"
        validate_jsonl_file(xbrl_path, XBRLFact)
    
    elif stage == 5:
        # Stage 5: Validate chunks
        chunks_path = config.paths.final_dir / doc_id / "chunks.jsonl"
        validate_jsonl_file(chunks_path, Chunk)
    
    return True
