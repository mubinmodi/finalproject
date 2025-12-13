"""Configuration management for the SEC filings pipeline."""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class PathConfig(BaseModel):
    """File system paths configuration."""
    
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")
    raw_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data" / "raw")
    processed_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data" / "processed")
    final_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data" / "final")
    logs_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "logs")
    
    def ensure_dirs(self):
        """Create all necessary directories."""
        for path in [self.data_dir, self.raw_dir, self.processed_dir, 
                     self.final_dir, self.logs_dir]:
            path.mkdir(parents=True, exist_ok=True)


class DownloadConfig(BaseModel):
    """SEC filing download configuration."""
    
    user_agent: str = Field(
        default=os.getenv("SEC_USER_AGENT", "Anonymous anonymous@example.com"),
        description="SEC requires a user agent string"
    )
    download_dir: str = "sec-edgar-filings"
    rate_limit_sleep: float = 0.1  # SEC rate limit: 10 requests/second


class OCRConfig(BaseModel):
    """OCR configuration."""
    
    tesseract_cmd: Optional[str] = Field(
        default=os.getenv("TESSERACT_CMD"),
        description="Path to tesseract executable"
    )
    lang: str = "eng"
    confidence_threshold: float = 60.0  # Minimum OCR confidence to use text


class LayoutConfig(BaseModel):
    """Layout detection configuration."""
    
    model_name: str = "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"
    confidence_threshold: float = 0.5
    device: str = "cpu"  # or "cuda" if GPU available
    label_map: dict = Field(default_factory=lambda: {
        0: "text",
        1: "title",
        2: "list",
        3: "table",
        4: "figure"
    })


class TableConfig(BaseModel):
    """Table extraction configuration."""
    
    camelot_flavor: list = Field(default_factory=lambda: ["lattice", "stream"])
    min_accuracy: float = 50.0  # Minimum Camelot accuracy score
    edge_tol: int = 50  # Camelot edge tolerance
    row_tol: int = 2
    col_tol: int = 2


class XBRLConfig(BaseModel):
    """XBRL parsing configuration."""
    
    validate: bool = True
    include_dimensions: bool = True
    include_footnotes: bool = True


class ChunkingConfig(BaseModel):
    """Document chunking configuration for RAG."""
    
    chunk_size: int = 1000  # Characters per chunk
    chunk_overlap: int = 200  # Overlap between chunks
    min_chunk_size: int = 100  # Minimum viable chunk size


class AgentConfig(BaseModel):
    """LLM agent configuration."""
    
    openai_api_key: str = Field(default=os.getenv("OPENAI_API_KEY", ""))
    model: str = "gpt-4-turbo-preview"
    temperature: float = 0.1
    max_tokens: int = 4000
    embedding_model: str = "text-embedding-3-small"
    vector_db_path: str = "./chroma_db"


class LoggingConfig(BaseModel):
    """Logging configuration."""
    
    level: str = os.getenv("LOG_LEVEL", "INFO")
    format: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
    rotation: str = "10 MB"
    retention: str = "30 days"


class Config(BaseModel):
    """Main configuration class."""
    
    paths: PathConfig = Field(default_factory=PathConfig)
    download: DownloadConfig = Field(default_factory=DownloadConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    layout: LayoutConfig = Field(default_factory=LayoutConfig)
    table: TableConfig = Field(default_factory=TableConfig)
    xbrl: XBRLConfig = Field(default_factory=XBRLConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    def initialize(self):
        """Initialize the configuration (create directories, setup logging, etc.)."""
        self.paths.ensure_dirs()
    
    class Config:
        arbitrary_types_allowed = True


# Global configuration instance
config = Config()
config.initialize()
