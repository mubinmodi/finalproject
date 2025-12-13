#!/usr/bin/env python3
"""
Main Pipeline Runner

Executes the complete SEC filings processing pipeline from download to chunking.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from pipeline import (
    DownloadStage,
    LayoutStage,
    TextStage,
    TableStage,
    XBRLStage,
    ChunkingStage
)
from utils import config, get_logger, validate_pipeline_output

logger = get_logger("pipeline_runner")


class PipelineRunner:
    """Main pipeline orchestrator."""
    
    def __init__(self):
        """Initialize pipeline runner."""
        self.stages = {
            0: ("Download", DownloadStage),
            1: ("Layout Detection", LayoutStage),
            2: ("Text Extraction", TextStage),
            3: ("Table Extraction", TableStage),
            4: ("XBRL Extraction", XBRLStage),
            5: ("Chunking", ChunkingStage)
        }
        logger.info("Pipeline runner initialized")
    
    def run(
        self,
        ticker: Optional[str] = None,
        form_type: str = "10-K",
        limit: int = 1,
        doc_id: Optional[str] = None,
        stages: Optional[List[int]] = None,
        validate: bool = True
    ) -> List[str]:
        """
        Run the complete pipeline or specific stages.
        
        Args:
            ticker: Stock ticker (for download stage)
            form_type: Filing form type
            limit: Number of filings to download
            doc_id: Specific document ID (skip download)
            stages: List of stage numbers to run (None = all)
            validate: Validate outputs after each stage
        
        Returns:
            List of processed document IDs
        """
        logger.info("="*80)
        logger.info("SEC FILINGS PIPELINE")
        logger.info("="*80)
        
        doc_ids = []
        
        # Stage 0: Download
        if stages is None or 0 in stages:
            if not ticker and not doc_id:
                logger.error("Either --ticker or --doc-id must be provided")
                sys.exit(1)
            
            if ticker:
                logger.info(f"\n{'='*80}")
                logger.info(f"Stage 0: Download")
                logger.info(f"{'='*80}")
                
                downloader = DownloadStage()
                doc_ids = downloader.download(
                    ticker=ticker,
                    form_type=form_type,
                    limit=limit
                )
                
                if validate:
                    for doc_id_item in doc_ids:
                        validate_pipeline_output(doc_id_item, 0, config)
                
                logger.info(f"✅ Downloaded {len(doc_ids)} filings")
            else:
                doc_ids = [doc_id]
        else:
            if not doc_id:
                logger.error("--doc-id required when skipping download stage")
                sys.exit(1)
            doc_ids = [doc_id]
        
        # Process each document through remaining stages
        for doc_id_item in doc_ids:
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing: {doc_id_item}")
            logger.info(f"{'='*80}")
            
            # Stage 1: Layout Detection
            if stages is None or 1 in stages:
                self._run_stage(1, doc_id_item, validate)
            
            # Stage 2: Text Extraction
            if stages is None or 2 in stages:
                self._run_stage(2, doc_id_item, validate)
            
            # Stage 3: Table Extraction
            if stages is None or 3 in stages:
                self._run_stage(3, doc_id_item, validate)
            
            # Stage 4: XBRL Extraction
            if stages is None or 4 in stages:
                self._run_stage(4, doc_id_item, validate)
            
            # Stage 5: Chunking
            if stages is None or 5 in stages:
                self._run_stage(5, doc_id_item, validate)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ Pipeline Complete")
        logger.info(f"{'='*80}")
        logger.info(f"Processed {len(doc_ids)} documents")
        
        return doc_ids
    
    def _run_stage(self, stage_num: int, doc_id: str, validate: bool):
        """Run a specific pipeline stage."""
        stage_name, stage_class = self.stages[stage_num]
        
        logger.info(f"\n--- Stage {stage_num}: {stage_name} ---")
        
        try:
            stage = stage_class()
            stage.process(doc_id)
            
            if validate:
                validate_pipeline_output(doc_id, stage_num, config)
                logger.info(f"✓ Validation passed")
        
        except Exception as e:
            logger.error(f"❌ Stage {stage_num} failed: {e}")
            raise


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run SEC filings processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download and process 1 AAPL 10-K filing
  python run_pipeline.py --ticker AAPL --form-type 10-K --limit 1
  
  # Download multiple filings
  python run_pipeline.py --ticker MSFT --form-type 10-Q --limit 4
  
  # Process existing document (skip download)
  python run_pipeline.py --doc-id AAPL_10-K_0001234567890
  
  # Run specific stages only
  python run_pipeline.py --doc-id AAPL_10-K_0001234567890 --stages 1,2,3
  
  # Skip validation (faster)
  python run_pipeline.py --ticker AAPL --no-validate
"""
    )
    
    # Download options
    download_group = parser.add_argument_group("download options")
    download_group.add_argument(
        "--ticker",
        help="Stock ticker symbol (e.g., AAPL, MSFT)"
    )
    download_group.add_argument(
        "--form-type",
        default="10-K",
        help="Filing form type (default: 10-K)"
    )
    download_group.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Number of filings to download (default: 1)"
    )
    download_group.add_argument(
        "--after",
        help="Download filings after date (YYYY-MM-DD)"
    )
    download_group.add_argument(
        "--before",
        help="Download filings before date (YYYY-MM-DD)"
    )
    
    # Processing options
    process_group = parser.add_argument_group("processing options")
    process_group.add_argument(
        "--doc-id",
        help="Process specific document ID (skip download)"
    )
    process_group.add_argument(
        "--stages",
        help="Comma-separated stage numbers to run (e.g., 1,2,3)"
    )
    process_group.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation after each stage"
    )
    
    args = parser.parse_args()
    
    # Parse stages
    stages = None
    if args.stages:
        try:
            stages = [int(s.strip()) for s in args.stages.split(',')]
        except ValueError:
            logger.error("Invalid stages format. Use: --stages 1,2,3")
            sys.exit(1)
    
    # Run pipeline
    runner = PipelineRunner()
    
    try:
        doc_ids = runner.run(
            ticker=args.ticker,
            form_type=args.form_type,
            limit=args.limit,
            doc_id=args.doc_id,
            stages=stages,
            validate=not args.no_validate
        )
        
        # Print summary
        print("\n" + "="*80)
        print("📊 PIPELINE SUMMARY")
        print("="*80)
        print(f"\nProcessed Documents:")
        for doc_id in doc_ids:
            print(f"  ✓ {doc_id}")
        
        print(f"\nOutputs:")
        print(f"  - Raw data: {config.paths.raw_dir}")
        print(f"  - Processed data: {config.paths.processed_dir}")
        print(f"  - Final chunks: {config.paths.final_dir}")
        
        print(f"\n✅ Pipeline completed successfully!")
        print(f"\nNext steps:")
        print(f"  1. Run agents: python run_agents.py --doc-id {doc_ids[0]}")
        print(f"  2. View markdown: cat {config.paths.final_dir}/{doc_ids[0]}/filing.md")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
