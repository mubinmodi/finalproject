#!/usr/bin/env python3
"""
Example: Compare Multiple Companies

Downloads and analyzes filings from multiple companies,
then compares their investment potential.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import DownloadStage
from agents import DecisionAgent
from utils import get_logger

logger = get_logger("compare_example")


def main():
    """Compare multiple companies."""
    
    # Companies to compare
    COMPANIES = [
        ("AAPL", "Apple Inc."),
        ("MSFT", "Microsoft"),
        ("GOOGL", "Alphabet Inc.")
    ]
    
    FORM_TYPE = "10-K"
    
    logger.info("="*80)
    logger.info("EXAMPLE: Multi-Company Comparison")
    logger.info("="*80)
    
    # Download filings (in a real scenario, you'd process these too)
    logger.info("\n📥 Downloading filings...")
    downloader = DownloadStage()
    
    doc_ids = []
    for ticker, name in COMPANIES:
        logger.info(f"  Downloading {name} ({ticker})...")
        try:
            ids = downloader.download(ticker=ticker, form_type=FORM_TYPE, limit=1)
            if ids:
                doc_ids.extend(ids)
                logger.info(f"  ✓ {name}: {ids[0]}")
        except Exception as e:
            logger.error(f"  ✗ Failed to download {ticker}: {e}")
    
    if len(doc_ids) < 2:
        logger.error("Need at least 2 filings to compare")
        return
    
    logger.info(f"\n✓ Downloaded {len(doc_ids)} filings")
    
    # Note: In a real scenario, you would process each filing through
    # the pipeline and agents before comparison. For this example,
    # we'll assume they're already processed.
    
    logger.info("\n🔍 To complete this comparison:")
    logger.info("1. Process each filing through the pipeline:")
    for doc_id in doc_ids:
        logger.info(f"   python run_pipeline.py --doc-id {doc_id}")
    
    logger.info("\n2. Run agents on each filing:")
    for doc_id in doc_ids:
        logger.info(f"   python run_agents.py --doc-id {doc_id}")
    
    logger.info("\n3. Then run comparison:")
    logger.info(f"   (Compare functionality would use DecisionAgent.compare_investments)")
    
    # Example comparison (if analyses exist)
    logger.info("\n📊 Comparison Preview:")
    logger.info("  This would show:")
    logger.info("  - Relative investment ratings")
    logger.info("  - Risk vs. return profiles")
    logger.info("  - Best fit for different investor types")
    logger.info("  - Key differentiating factors")
    
    print("\n" + "="*80)
    print("✅ Example setup complete!")
    print("\nNext steps:")
    print("1. Process each filing with run_pipeline.py")
    print("2. Analyze each filing with run_agents.py")
    print("3. Results will be saved in data/final/")
    print("="*80)


if __name__ == "__main__":
    main()
