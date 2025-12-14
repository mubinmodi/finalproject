#!/usr/bin/env python3
"""
Quick processing of 2024 filing - skip Stage 1 (layout), process rest.
"""
import os
import sys
import subprocess
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")


def process_filing_quick(doc_id: str):
    """Process filing through stages 2-5 (skip Stage 1)."""
    logger.info(f"🔄 Quick processing {doc_id} (skipping Stage 1)...")
    
    project_root = Path(__file__).parent.absolute()
    
    # Skip Stage 1, run 2-5
    stages = [
        ("Stage 2: Text", f"{sys.executable} pipeline/stage2_text.py --doc-id {doc_id}"),
        ("Stage 3: Tables", f"{sys.executable} pipeline/stage3_tables.py --doc-id {doc_id}"),
        ("Stage 4: XBRL", f"{sys.executable} pipeline/stage4_xbrl.py --doc-id {doc_id}"),
        ("Stage 5: Chunks", f"{sys.executable} pipeline/stage5_chunks.py --doc-id {doc_id}"),
    ]
    
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
    """Run all agents."""
    logger.info(f"🤖 Running agents for {doc_id}...")
    
    project_root = Path(__file__).parent.absolute()
    env = os.environ.copy()
    env['PYTHONPATH'] = str(project_root)
    
    cmd = [sys.executable, "run_agents_v2.py", "--doc-id", doc_id]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root, env=env)
    
    if result.returncode != 0:
        logger.error(f"Agents failed: {result.stderr}")
        return False
    
    logger.success(f"✅ All agents complete for {doc_id}")
    return True


def run_comparison(ticker: str):
    """Run comparison analysis."""
    logger.info(f"📊 Running comparison for {ticker}...")
    
    project_root = Path(__file__).parent.absolute()
    env = os.environ.copy()
    env['PYTHONPATH'] = str(project_root)
    
    cmd = [sys.executable, "run_comparison.py", "--ticker", ticker]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root, env=env)
    
    if result.returncode != 0:
        logger.error(f"Comparison failed: {result.stderr}")
        return False
    
    logger.success(f"✅ Comparison complete for {ticker}")
    return True


def main():
    doc_id_2024 = "AAPL_10-K_0000320193-24-000123"
    ticker = "AAPL"
    
    logger.info("=" * 80)
    logger.info("QUICK 2024 PROCESSING & COMPARISON")
    logger.info("=" * 80)
    
    # Process 2024
    if not process_filing_quick(doc_id_2024):
        logger.error("Processing failed")
        return
    
    # Run agents on 2024
    if not run_agents(doc_id_2024):
        logger.error("Agents failed")
        return
    
    # Run comparison
    if not run_comparison(ticker):
        logger.error("Comparison failed")
        return
    
    logger.info("=" * 80)
    logger.info("✅ COMPLETE - 2024 vs 2025 COMPARISON READY")
    logger.info("=" * 80)
    logger.info("")
    logger.info("View results:")
    logger.info("  streamlit run streamlit_app.py")
    logger.info("")
    logger.info("📊 Comparison tab will show:")
    logger.info("  • Financial trends (Revenue, Margins)")
    logger.info("  • Year-over-year growth rates")
    logger.info("  • SWOT evolution")
    logger.info("  • Risk factor changes")


if __name__ == "__main__":
    main()
