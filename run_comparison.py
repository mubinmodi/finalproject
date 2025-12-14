#!/usr/bin/env python3
"""
Run comparison analysis on multiple years of SEC filings.

Usage:
    python run_comparison.py --ticker AAPL
    python run_comparison.py --doc-ids AAPL_10-K_2023 AAPL_10-K_2024 AAPL_10-K_2025
"""

import argparse
import sys
from pathlib import Path
from agents.comparison_agent import ComparisonAgent
from utils import get_logger

logger = get_logger("run_comparison")


def find_doc_ids_for_ticker(ticker: str, filing_type: str = "10-K") -> list:
    """Find all processed doc IDs for a ticker."""
    
    final_dir = Path("data/final")
    
    if not final_dir.exists():
        logger.error(f"No analysis directory found: {final_dir}")
        return []
    
    # Find all directories matching ticker_filing-type pattern
    doc_ids = []
    for doc_dir in final_dir.iterdir():
        if doc_dir.is_dir() and doc_dir.name.startswith(f"{ticker}_{filing_type}_"):
            # Check if analysis files exist
            if (doc_dir / "metrics_analysis_v2.json").exists():
                doc_ids.append(doc_dir.name)
    
    return sorted(doc_ids, reverse=True)  # Newest first


def main():
    parser = argparse.ArgumentParser(description="Compare multiple years of SEC filings")
    parser.add_argument("--ticker", help="Stock ticker (e.g., AAPL)")
    parser.add_argument("--doc-ids", nargs="+", help="Specific doc IDs to compare")
    parser.add_argument("--filing-type", default="10-K", help="Filing type (default: 10-K)")
    
    args = parser.parse_args()
    
    # Determine doc IDs to compare
    if args.doc_ids:
        doc_ids = args.doc_ids
    elif args.ticker:
        doc_ids = find_doc_ids_for_ticker(args.ticker, args.filing_type)
    else:
        logger.error("Must provide either --ticker or --doc-ids")
        sys.exit(1)
    
    if len(doc_ids) < 2:
        logger.error(f"Need at least 2 filings to compare. Found: {len(doc_ids)}")
        if doc_ids:
            logger.info(f"Available: {doc_ids}")
        sys.exit(1)
    
    logger.info("="*80)
    logger.info("MULTI-YEAR COMPARISON ANALYSIS")
    logger.info("="*80)
    logger.info(f"Comparing {len(doc_ids)} filings:")
    for doc_id in doc_ids:
        logger.info(f"  • {doc_id}")
    
    # Run comparison agent
    agent = ComparisonAgent()
    result = agent.analyze(doc_ids)
    
    # Display results
    logger.info("\n" + "="*80)
    logger.info("📊 COMPARISON RESULTS")
    logger.info("="*80)
    
    # Financial trends
    if result['financial_trends']['yoy_changes']:
        logger.info("\n💰 Financial Trends:")
        for trend in result['financial_trends']['yoy_changes']:
            logger.info(f"  {trend['period']}:")
            logger.info(f"    Revenue Growth: {trend['revenue_growth']:+.1f}%")
            logger.info(f"    Net Income Growth: {trend['net_income_growth']:+.1f}%")
            logger.info(f"    Gross Margin Change: {trend['gross_margin_change']:+.1f}pp")
    
    # Key insights
    logger.info("\n🔍 Key Insights:")
    for insight in result['key_insights']:
        logger.info(f"  • {insight}")
    
    logger.info("\n" + "="*80)
    logger.info("✅ COMPARISON COMPLETE")
    logger.info("="*80)
    logger.info("\n💡 View results in Streamlit:")
    logger.info("   streamlit run streamlit_app.py")
    logger.info("   (Go to 'Comparison' tab)")


if __name__ == "__main__":
    main()
