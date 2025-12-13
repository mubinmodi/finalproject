#!/usr/bin/env python3
"""
Simple Example: Process a Single Filing

This script demonstrates the complete workflow:
1. Download a filing
2. Process through pipeline
3. Run all agents
4. Display results
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import DownloadStage, LayoutStage, TextStage, TableStage, XBRLStage, ChunkingStage
from agents import SummaryAgent, SWOTAgent, MetricsAgent, DecisionAgent
from utils import get_logger

logger = get_logger("example")


def main():
    """Run complete example."""
    
    # Configuration
    TICKER = "AAPL"
    FORM_TYPE = "10-K"
    LIMIT = 1
    
    logger.info("="*80)
    logger.info("SIMPLE EXAMPLE: Complete SEC Filing Analysis")
    logger.info("="*80)
    
    # Step 1: Download
    logger.info("\n📥 Step 1: Downloading filing...")
    downloader = DownloadStage()
    doc_ids = downloader.download(ticker=TICKER, form_type=FORM_TYPE, limit=LIMIT)
    
    if not doc_ids:
        logger.error("No filings downloaded")
        return
    
    doc_id = doc_ids[0]
    logger.info(f"✓ Downloaded: {doc_id}")
    
    # Step 2: Process Pipeline
    logger.info("\n🔄 Step 2: Processing pipeline...")
    
    try:
        # Layout detection
        logger.info("  - Detecting layout...")
        layout = LayoutStage()
        layout.process(doc_id)
        logger.info("  ✓ Layout detection complete")
    except Exception as e:
        logger.warning(f"  ⚠ Layout detection failed: {e}")
    
    try:
        # Text extraction
        logger.info("  - Extracting text...")
        text = TextStage()
        text.process(doc_id)
        logger.info("  ✓ Text extraction complete")
    except Exception as e:
        logger.error(f"  ✗ Text extraction failed: {e}")
        return
    
    try:
        # Table extraction
        logger.info("  - Extracting tables...")
        tables = TableStage()
        tables.process(doc_id)
        logger.info("  ✓ Table extraction complete")
    except Exception as e:
        logger.warning(f"  ⚠ Table extraction failed: {e}")
    
    try:
        # XBRL extraction
        logger.info("  - Extracting XBRL...")
        xbrl = XBRLStage()
        xbrl.process(doc_id)
        logger.info("  ✓ XBRL extraction complete")
    except Exception as e:
        logger.warning(f"  ⚠ XBRL extraction failed: {e}")
    
    try:
        # Chunking
        logger.info("  - Creating chunks...")
        chunker = ChunkingStage()
        chunker.process(doc_id)
        logger.info("  ✓ Chunking complete")
    except Exception as e:
        logger.error(f"  ✗ Chunking failed: {e}")
        return
    
    # Step 3: Run Agents
    logger.info("\n🤖 Step 3: Running analysis agents...")
    
    try:
        # Summary Agent
        logger.info("  - Running Summary Agent...")
        summary_agent = SummaryAgent()
        summary_result = summary_agent.analyze(doc_id)
        logger.info("  ✓ Summary complete")
    except Exception as e:
        logger.error(f"  ✗ Summary Agent failed: {e}")
        summary_result = None
    
    try:
        # SWOT Agent
        logger.info("  - Running SWOT Agent...")
        swot_agent = SWOTAgent()
        swot_result = swot_agent.analyze(doc_id)
        logger.info("  ✓ SWOT analysis complete")
    except Exception as e:
        logger.error(f"  ✗ SWOT Agent failed: {e}")
        swot_result = None
    
    try:
        # Metrics Agent
        logger.info("  - Running Metrics Agent...")
        metrics_agent = MetricsAgent()
        metrics_result = metrics_agent.analyze(doc_id)
        logger.info("  ✓ Metrics analysis complete")
    except Exception as e:
        logger.error(f"  ✗ Metrics Agent failed: {e}")
        metrics_result = None
    
    try:
        # Decision Agent
        logger.info("  - Running Decision Agent...")
        decision_agent = DecisionAgent()
        decision_result = decision_agent.analyze(doc_id)
        logger.info("  ✓ Investment decision complete")
    except Exception as e:
        logger.error(f"  ✗ Decision Agent failed: {e}")
        decision_result = None
    
    # Step 4: Display Results
    logger.info("\n📊 Step 4: Results Summary")
    logger.info("="*80)
    
    if summary_result:
        print("\n📝 EXECUTIVE SUMMARY (excerpt):")
        summary_text = summary_result.get('executive_summary', '')
        print(summary_text[:500] + "..." if len(summary_text) > 500 else summary_text)
    
    if swot_result:
        print("\n🎯 SWOT ANALYSIS:")
        swot = swot_result.get('swot_analysis', {})
        print(f"  Strengths: {len(swot.get('strengths', {}).get('items', []))}")
        print(f"  Weaknesses: {len(swot.get('weaknesses', {}).get('items', []))}")
        print(f"  Opportunities: {len(swot.get('opportunities', {}).get('items', []))}")
        print(f"  Threats: {len(swot.get('threats', {}).get('items', []))}")
    
    if metrics_result:
        print("\n💰 KEY METRICS:")
        metrics = metrics_result.get('metrics', {}).get('xbrl_metrics', {})
        for key in ['revenue', 'net_income', 'assets']:
            if key in metrics:
                value = metrics[key].get('value')
                units = metrics[key].get('units', '')
                if value:
                    print(f"  {key.title()}: {value:,} {units}")
    
    if decision_result:
        print("\n🎯 INVESTMENT RECOMMENDATION:")
        rec = decision_result.get('recommendation', {})
        print(f"  Rating: {rec.get('rating', 'N/A')}")
        print(f"  Confidence: {rec.get('confidence', 'N/A')}")
        if rec.get('rationale'):
            print(f"  Top Rationale: {rec['rationale'][0]}")
    
    print("\n" + "="*80)
    print("✅ Example complete!")
    print(f"\nDetailed results saved in: data/final/{doc_id}/")
    print("="*80)


if __name__ == "__main__":
    main()
