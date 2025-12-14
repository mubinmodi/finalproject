#!/usr/bin/env python3
"""
Download and process multiple years of 10-K filings for comparison.

Usage:
    python download_multi_year.py --ticker AAPL --years 2
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from loguru import logger
from sec_edgar_downloader import Downloader

def download_years(ticker: str, num_years: int = 2, filing_type: str = "10-K"):
    """Download multiple years of filings."""
    
    logger.info(f"📥 Downloading last {num_years} years of {filing_type} filings for {ticker}")
    
    try:
        # Get user agent from environment or use default
        user_agent = os.getenv("SEC_USER_AGENT", "Anonymous anonymous@example.com")
        email = user_agent.split()[-1]  # Extract email from "Name email@example.com"
        
        # Use sec-edgar-downloader library
        dl = Downloader("Project Green Lattern", email, "data/raw")
        
        # Download filings
        logger.info(f"Downloading {num_years} filings of type {filing_type} for {ticker}...")
        num_downloaded = dl.get(
            filing_type, 
            ticker, 
            limit=num_years,
            download_details=True  # Download XBRL and other attachments
        )
        
        logger.success(f"✅ Downloaded {num_downloaded} {filing_type} filings for {ticker}")
        return True
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.error("Hint: Make sure you have internet connection and SEC EDGAR is accessible")
        return False


def process_filing(doc_id: str):
    """Run full pipeline for a single filing."""
    
    logger.info(f"🔄 Processing {doc_id} through pipeline...")
    
    # Get project root directory
    project_root = Path(__file__).parent.absolute()
    
    # Run all stages with proper PYTHONPATH
    stages = [
        ("Stage 1: Layout", f"{sys.executable} pipeline/stage1_layout.py"),
        ("Stage 2: Text", f"{sys.executable} pipeline/stage2_text.py"),
        ("Stage 3: Tables", f"{sys.executable} pipeline/stage3_tables.py"),
        ("Stage 4: XBRL", f"{sys.executable} pipeline/stage4_xbrl.py"),
        ("Stage 5: Chunks", f"{sys.executable} pipeline/stage5_chunks.py"),
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


def find_downloaded_filings(ticker: str, filing_type: str = "10-K"):
    """Find all downloaded filings for a ticker."""
    
    raw_dir = Path("data/raw/sec-edgar-filings") / ticker / filing_type
    
    if not raw_dir.exists():
        logger.error(f"No filings found in {raw_dir}")
        return []
    
    # Find all filing directories
    filings = []
    for filing_dir in sorted(raw_dir.iterdir()):
        if filing_dir.is_dir():
            # Extract accession number from directory name
            accession = filing_dir.name
            doc_id = f"{ticker}_{filing_type}_{accession}"
            filings.append((doc_id, filing_dir))
    
    return filings


def main():
    parser = argparse.ArgumentParser(description="Download and process multiple years of SEC filings")
    parser.add_argument("--ticker", required=True, help="Stock ticker (e.g., AAPL)")
    parser.add_argument("--years", type=int, default=2, help="Number of years to download (default: 2)")
    parser.add_argument("--filing-type", default="10-K", help="Filing type (default: 10-K)")
    parser.add_argument("--skip-download", action="store_true", help="Skip download, just process existing")
    parser.add_argument("--skip-pipeline", action="store_true", help="Skip pipeline, just run agents")
    parser.add_argument("--skip-agents", action="store_true", help="Skip agents, just run pipeline")
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info(f"MULTI-YEAR FILING ANALYSIS")
    logger.info(f"Ticker: {args.ticker} | Years: {args.years} | Type: {args.filing_type}")
    logger.info("="*80)
    
    # Step 1: Download filings
    if not args.skip_download:
        success = download_years(args.ticker, args.years, args.filing_type)
        if not success:
            logger.error("Download failed. Exiting.")
            sys.exit(1)
    
    # Step 2: Find all downloaded filings
    filings = find_downloaded_filings(args.ticker, args.filing_type)
    
    if not filings:
        logger.error("No filings found. Did download succeed?")
        sys.exit(1)
    
    logger.info(f"📁 Found {len(filings)} filings to process:")
    for doc_id, _ in filings:
        logger.info(f"  - {doc_id}")
    
    # Step 3: Process each filing through pipeline
    if not args.skip_pipeline:
        logger.info("\n" + "="*80)
        logger.info("PIPELINE PROCESSING")
        logger.info("="*80)
        
        for doc_id, filing_dir in filings:
            success = process_filing(doc_id)
            if not success:
                logger.warning(f"Pipeline failed for {doc_id}, continuing...")
    
    # Step 4: Run agents on each filing
    if not args.skip_agents:
        logger.info("\n" + "="*80)
        logger.info("AGENT ANALYSIS")
        logger.info("="*80)
        
        for doc_id, _ in filings:
            success = run_agents(doc_id)
            if not success:
                logger.warning(f"Agents failed for {doc_id}, continuing...")
    
    # Step 5: Summary
    logger.info("\n" + "="*80)
    logger.info("✅ MULTI-YEAR ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"Processed {len(filings)} filings for {args.ticker}")
    logger.info("\nNext steps:")
    logger.info("1. Run comparison agent: python run_comparison.py --ticker AAPL")
    logger.info("2. View in Streamlit: streamlit run streamlit_app.py")


if __name__ == "__main__":
    main()
