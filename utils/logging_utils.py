"""Logging utilities for the SEC filings pipeline."""

import sys
from pathlib import Path
from loguru import logger
from .config import config


def setup_logger(name: str = "pipeline", log_file: str = None):
    """
    Setup logger with file and console output.
    
    Args:
        name: Logger name (used in log messages)
        log_file: Optional log file path. If None, uses default from config.
    
    Returns:
        Configured logger instance
    """
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stderr,
        format=config.logging.format,
        level=config.logging.level,
        colorize=True
    )
    
    # Add file handler
    if log_file is None:
        log_file = config.paths.logs_dir / f"{name}.log"
    else:
        log_file = Path(log_file)
    
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_file,
        format=config.logging.format,
        level=config.logging.level,
        rotation=config.logging.rotation,
        retention=config.logging.retention,
        compression="zip"
    )
    
    return logger


def get_logger(name: str = "pipeline"):
    """Get or create a logger instance."""
    return setup_logger(name)
