#!/usr/bin/env python3
"""
Download and process a single additional year for comparison.
"""
import os
import sys
import subprocess
from pathlib import Path
from loguru import logger

# Setup logging
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")


def download_filing(ticker: str, year: int):
    """Download a single year's filing."""
    logger.info(f"📥 Downloading {ticker} 10-K for {year}...")
    
    try:
        from sec_edgar_downloader import Downloader
        
        user_agent = os.getenv("SEC_USER_AGENT", "Project Green Lattern greenlattern@example.com")
        email = user_agent.split()[-1]
        
        dl = Downloader("Project Green Lattern", email, "data/raw")
        
        # Download 1 filing
        dl.get("10-K", ticker, limit=1, download_details=True)
        
        logger.success(f"✅ Downloaded {ticker} filings")
        return True
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def get_unprocessed_filing(ticker: str):
    """Find an unprocessed filing (2024 specifically)."""
    raw_dir = Path("data/raw/sec-edgar-filings") / ticker / "10-K"
    final_dir = Path("data/final")
    
    if not raw_dir.exists():
        return None
    
    # Find all filing directories
    filings = []
    for filing_dir in raw_dir.iterdir():
        if filing_dir.is_dir() and filing_dir.name.startswith("0"):
            filings.append(filing_dir.name)
    
    if not filings:
        return None
    
    # Sort (they're in format 0000320193-24-000123)
    filings.sort()
    
    # Check which ones are not processed
    for filing in filings:
        doc_id = f"{ticker}_10-K_{filing}"
        
        # Check if already processed (has summary.json in final)
        summary_file = final_dir / doc_id / "summary.json"
        if not summary_file.exists():
            logger.info(f"Found unprocessed filing: {doc_id}")
            return doc_id
    
    logger.warning("All filings already processed")
    return None


def process_filing(doc_id: str):
    """Run full pipeline for a single filing."""
    logger.info(f"🔄 Processing {doc_id} through pipeline...")
    
    # Get project root
    project_root = Path(__file__).parent.absolute()
    
    # Run all stages
    stages = [
        ("Stage 1: Layout", f"{sys.executable} pipeline/stage1_layout.py --doc-id {doc_id}"),
        ("Stage 2: Text", f"{sys.executable} pipeline/stage2_text.py --doc-id {doc_id}"),
        ("Stage 3: Tables", f"{sys.executable} pipeline/stage3_tables.py --doc-id {doc_id}"),
        ("Stage 4: XBRL", f"{sys.executable} pipeline/stage4_xbrl.py --doc-id {doc_id}"),
        ("Stage 5: Chunks", f"{sys.executable} pipeline/stage5_chunks.py --doc-id {doc_id}"),
    ]
    
    # Set environment with PYTHONPATH
    env = os.environ.copy()
    env['PYTHONPATH'] = str(project_root)
    
    for stage_name, cmd in stages:
        logger.info(f"  Running {stage_name}...")
        result = subprocess.run(
            cmd.split(), 
            capture_output=True, 
            text=True,
            cwd=project_root,
            env=env
        )
        
        if result.returncode != 0:
            logger.error(f"  ❌ {stage_name} failed: {result.stderr}")
            return False
        
        logger.success(f"  ✅ {stage_name} complete")
    
    logger.success(f"✅ Pipeline complete for {doc_id}")
    return True


def run_agents(doc_id: str):
    """Run all agents for a filing."""
    logger.info(f"🤖 Running agents for {doc_id}...")
    
    # Get project root
    project_root = Path(__file__).parent.absolute()
    
    # Set environment with PYTHONPATH
    env = os.environ.copy()
    env['PYTHONPATH'] = str(project_root)
    
    cmd = [sys.executable, "run_agents_v2.py", "--doc-id", doc_id]
    result = subprocess.run(
        cmd, 
        capture_output=True, 
        text=True,
        cwd=project_root,
        env=env
    )
    
    if result.returncode != 0:
        logger.error(f"Agents failed: {result.stderr}")
        return False
    
    logger.success(f"✅ All agents complete for {doc_id}")
    return True


def main():
    ticker = "AAPL"
    
    logger.info("=" * 80)
    logger.info("SINGLE YEAR DOWNLOAD & PROCESSING")
    logger.info("=" * 80)
    
    # Step 1: Find unprocessed filing
    doc_id = get_unprocessed_filing(ticker)
    if not doc_id:
        logger.error("Could not find downloaded filing")
        return
    
    # Step 3: Process through pipeline
    if not process_filing(doc_id):
        logger.error("Pipeline failed")
        return
    
    # Step 4: Run agents
    if not run_agents(doc_id):
        logger.error("Agents failed")
        return
    
    logger.info("=" * 80)
    logger.info("✅ PROCESSING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Processed: {doc_id}")
    logger.info("")
    logger.info("Next steps:")
    logger.info(f"1. Run comparison: python run_comparison.py --ticker {ticker}")
    logger.info("2. View in Streamlit: streamlit run streamlit_app.py")


if __name__ == "__main__":
    main()
