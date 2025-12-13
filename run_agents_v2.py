#!/usr/bin/env python3
"""
Enhanced Multi-Agent Analysis Runner (V2)

Runs enhanced agents with provenance tracking and citation support.
"""

import argparse
import sys
import json
from pathlib import Path

from agents import SummaryAgentV2, SWOTAgentV2, MetricsAgentV2, DecisionAgentV2
from utils import config, get_logger

logger = get_logger("agents_runner_v2")


def main():
    """Command-line interface for V2 agents."""
    parser = argparse.ArgumentParser(
        description="Run enhanced analysis agents with provenance tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all V2 agents
  python run_agents_v2.py --doc-id AAPL_10-K_0001234567890
  
  # Run with prior year comparison
  python run_agents_v2.py --doc-id AAPL_10-K_2023 --prior-doc-id AAPL_10-K_2022
  
  # Run specific agents
  python run_agents_v2.py --doc-id AAPL_10-K_0001234567890 --agents summary,metrics
  
  # Custom investor profile
  python run_agents_v2.py --doc-id AAPL_10-K_0001234567890 \\
    --risk-tolerance aggressive \\
    --horizon long_term
"""
    )
    
    parser.add_argument(
        "--doc-id",
        required=True,
        help="Document ID to analyze"
    )
    
    parser.add_argument(
        "--prior-doc-id",
        help="Prior year document ID for delta analysis"
    )
    
    parser.add_argument(
        "--agents",
        help="Comma-separated agent names (summary,swot,metrics,decision)"
    )
    
    parser.add_argument(
        "--risk-tolerance",
        default="moderate",
        choices=["conservative", "moderate", "aggressive"],
        help="Risk tolerance for decision agent (default: moderate)"
    )
    
    parser.add_argument(
        "--horizon",
        dest="investment_horizon",
        default="medium_term",
        choices=["short_term", "medium_term", "long_term"],
        help="Investment horizon (default: medium_term)"
    )
    
    args = parser.parse_args()
    
    # Determine which agents to run
    if args.agents:
        agents_list = [a.strip() for a in args.agents.split(',')]
    else:
        agents_list = ['summary', 'swot', 'metrics', 'decision']
    
    logger.info("="*80)
    logger.info("ENHANCED MULTI-AGENT ANALYSIS (V2)")
    logger.info("="*80)
    logger.info(f"Document: {args.doc_id}")
    if args.prior_doc_id:
        logger.info(f"Prior Year: {args.prior_doc_id}")
    
    results = {}
    
    # Run Summary Agent
    if 'summary' in agents_list:
        logger.info("\n" + "="*80)
        logger.info("SUMMARY AGENT V2 (Executive Brief)")
        logger.info("="*80)
        
        try:
            agent = SummaryAgentV2()
            result = agent.analyze(args.doc_id, args.prior_doc_id)
            results['summary'] = result
            
            # Display preview
            brief = result['executive_brief']
            print("\n📋 KEY FINDINGS:")
            for i, finding in enumerate(brief['key_findings'][:3], 1):
                print(f"  {i}. {finding['finding']}")
            
            print(f"\n📚 Citations: {result['metadata']['num_citations']}")
            logger.info("✅ Summary Agent complete")
            
        except Exception as e:
            logger.error(f"❌ Summary Agent failed: {e}")
    
    # Run SWOT Agent
    if 'swot' in agents_list:
        logger.info("\n" + "="*80)
        logger.info("SWOT AGENT V2 (Hostile Witness Mode)")
        logger.info("="*80)
        
        try:
            agent = SWOTAgentV2()
            result = agent.analyze(args.doc_id, args.prior_doc_id)
            results['swot'] = result
            
            # Display preview
            swot = result['swot_analysis']
            print("\n💪 STRENGTHS:")
            for item in swot['strengths']['items'][:2]:
                print(f"  • {item['strength'][:80]}...")
            
            print("\n⚠️  WEAKNESSES:")
            for item in swot['weaknesses']['items'][:2]:
                print(f"  • {item['weakness'][:80]}...")
            
            print(f"\n📚 Citations: {result['metadata']['num_citations']}")
            logger.info("✅ SWOT Agent complete")
            
        except Exception as e:
            logger.error(f"❌ SWOT Agent failed: {e}")
    
    # Run Metrics Agent
    if 'metrics' in agents_list:
        logger.info("\n" + "="*80)
        logger.info("METRICS AGENT V2 (Quant Engine)")
        logger.info("="*80)
        
        try:
            agent = MetricsAgentV2()
            result = agent.analyze(args.doc_id)
            results['metrics'] = result
            
            # Display preview
            metrics = result['metrics']
            
            print("\n💰 PROFITABILITY:")
            for name, data in list(metrics.get('profitability', {}).items())[:3]:
                print(f"  {name.replace('_', ' ').title()}: {data['value']:.2f}{data['units']}")
            
            print("\n🏦 LIQUIDITY:")
            for name, data in list(metrics.get('liquidity', {}).items())[:2]:
                print(f"  {name.replace('_', ' ').title()}: {data['value']:.2f}{data['units']}")
            
            logger.info("✅ Metrics Agent complete")
            
        except Exception as e:
            logger.error(f"❌ Metrics Agent failed: {e}")
    
    # Run Decision Agent
    if 'decision' in agents_list:
        logger.info("\n" + "="*80)
        logger.info("DECISION AGENT V2 (Investment Memo)")
        logger.info("="*80)
        
        try:
            agent = DecisionAgentV2()
            result = agent.analyze(
                doc_id=args.doc_id,
                risk_tolerance=args.risk_tolerance,
                investment_horizon=args.investment_horizon
            )
            results['decision'] = result
            
            # Display preview
            memo = result['investment_memo']
            rec = memo['recommendation']
            
            print(f"\n🎯 RECOMMENDATION: {rec['rating']}")
            print(f"📊 Confidence: {rec['confidence']}")
            print(f"💯 Composite Score: {rec['composite_score']}/100")
            
            print("\n🚩 RED FLAGS:")
            for flag in memo['red_flags'][:3]:
                print(f"  • {flag}")
            
            logger.info("✅ Decision Agent complete")
            
        except Exception as e:
            logger.error(f"❌ Decision Agent failed: {e}")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("✅ ALL AGENTS COMPLETE")
    logger.info("="*80)
    
    print(f"\n📁 Detailed results saved in: {config.paths.final_dir / args.doc_id}/")
    print(f"\n💡 View in UI:")
    print(f"   streamlit run streamlit_app.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
