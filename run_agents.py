#!/usr/bin/env python3
"""
Multi-Agent Analysis Runner

Runs investment analysis agents on processed SEC filings.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Optional

from agents import SummaryAgent, SWOTAgent, MetricsAgent, DecisionAgent
from utils import config, get_logger

logger = get_logger("agents_runner")


class AgentsRunner:
    """Multi-agent analysis orchestrator."""
    
    def __init__(self):
        """Initialize agents runner."""
        self.agents = {
            'summary': SummaryAgent,
            'swot': SWOTAgent,
            'metrics': MetricsAgent,
            'decision': DecisionAgent
        }
        logger.info("Agents runner initialized")
    
    def run(
        self,
        doc_id: str,
        agents: Optional[List[str]] = None,
        risk_tolerance: str = "moderate",
        investment_horizon: str = "medium_term"
    ) -> dict:
        """
        Run analysis agents on a document.
        
        Args:
            doc_id: Document identifier
            agents: List of agent names to run (None = all)
            risk_tolerance: Risk tolerance for decision agent
            investment_horizon: Investment horizon for decision agent
        
        Returns:
            Dictionary of agent results
        """
        logger.info("="*80)
        logger.info("MULTI-AGENT INVESTMENT ANALYSIS")
        logger.info("="*80)
        logger.info(f"Document: {doc_id}")
        
        # Check if document exists
        chunks_path = config.paths.final_dir / doc_id / "chunks.jsonl"
        if not chunks_path.exists():
            logger.error(f"Document not found: {doc_id}")
            logger.error("Please run the pipeline first: python run_pipeline.py")
            sys.exit(1)
        
        # Determine which agents to run
        agents_to_run = agents if agents else list(self.agents.keys())
        
        results = {}
        
        # Run agents in order
        agent_order = ['summary', 'swot', 'metrics', 'decision']
        
        for agent_name in agent_order:
            if agent_name not in agents_to_run:
                continue
            
            logger.info(f"\n{'='*80}")
            logger.info(f"Running: {agent_name.upper()} Agent")
            logger.info(f"{'='*80}")
            
            try:
                agent_class = self.agents[agent_name]
                agent = agent_class()
                
                # Run agent with appropriate parameters
                if agent_name == 'decision':
                    result = agent.analyze(
                        doc_id=doc_id,
                        risk_tolerance=risk_tolerance,
                        investment_horizon=investment_horizon
                    )
                else:
                    result = agent.analyze(doc_id=doc_id)
                
                results[agent_name] = result
                logger.info(f"✅ {agent_name.upper()} Agent complete")
                
            except Exception as e:
                logger.error(f"❌ {agent_name.upper()} Agent failed: {e}")
                results[agent_name] = {"error": str(e)}
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ All Agents Complete")
        logger.info(f"{'='*80}")
        
        return results
    
    def print_summary(self, results: dict):
        """Print a summary of agent results."""
        print("\n" + "="*80)
        print("📊 ANALYSIS SUMMARY")
        print("="*80)
        
        # Summary Agent
        if 'summary' in results:
            print("\n📝 EXECUTIVE SUMMARY")
            print("-"*80)
            summary = results['summary'].get('executive_summary', 'N/A')
            print(summary[:500] + "..." if len(summary) > 500 else summary)
        
        # SWOT Agent
        if 'swot' in results:
            print("\n🎯 SWOT HIGHLIGHTS")
            print("-"*80)
            swot = results['swot'].get('swot_analysis', {})
            
            strengths = swot.get('strengths', {}).get('items', [])
            if strengths:
                print(f"  Strengths: {len(strengths)} identified")
            
            weaknesses = swot.get('weaknesses', {}).get('items', [])
            if weaknesses:
                print(f"  Weaknesses: {len(weaknesses)} identified")
            
            opportunities = swot.get('opportunities', {}).get('items', [])
            if opportunities:
                print(f"  Opportunities: {len(opportunities)} identified")
            
            threats = swot.get('threats', {}).get('items', [])
            if threats:
                print(f"  Threats: {len(threats)} identified")
        
        # Metrics Agent
        if 'metrics' in results:
            print("\n💰 KEY FINANCIAL METRICS")
            print("-"*80)
            metrics = results['metrics'].get('metrics', {})
            xbrl = metrics.get('xbrl_metrics', {})
            
            key_metrics = ['revenue', 'net_income', 'assets', 'equity']
            for metric in key_metrics:
                if metric in xbrl:
                    value = xbrl[metric].get('value')
                    units = xbrl[metric].get('units', '')
                    if value:
                        print(f"  {metric.replace('_', ' ').title()}: {value:,} {units}")
            
            ratios = metrics.get('calculated_ratios', {})
            if ratios:
                print("\n  Key Ratios:")
                for ratio_name, ratio_value in list(ratios.items())[:5]:
                    print(f"    {ratio_name.replace('_', ' ').title()}: {ratio_value:.2f}%")
        
        # Decision Agent
        if 'decision' in results:
            print("\n🎯 INVESTMENT RECOMMENDATION")
            print("-"*80)
            rec = results['decision'].get('recommendation', {})
            
            rating = rec.get('rating', 'N/A')
            confidence = rec.get('confidence', 'N/A')
            
            print(f"  Rating: {rating}")
            print(f"  Confidence: {confidence}")
            
            rationale = rec.get('rationale', [])
            if rationale:
                print("\n  Key Rationale:")
                for point in rationale[:3]:
                    print(f"    • {point}")
        
        print("\n" + "="*80)


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run multi-agent investment analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all agents
  python run_agents.py --doc-id AAPL_10-K_0001234567890
  
  # Run specific agents
  python run_agents.py --doc-id AAPL_10-K_0001234567890 --agents summary,metrics
  
  # Custom investor profile
  python run_agents.py --doc-id AAPL_10-K_0001234567890 \\
    --risk-tolerance aggressive \\
    --horizon long_term
  
  # Save detailed output
  python run_agents.py --doc-id AAPL_10-K_0001234567890 --output results.json
"""
    )
    
    parser.add_argument(
        "--doc-id",
        required=True,
        help="Document ID to analyze"
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
    
    parser.add_argument(
        "--output",
        help="Save combined results to JSON file"
    )
    
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Don't print summary to console"
    )
    
    args = parser.parse_args()
    
    # Parse agents
    agents_list = None
    if args.agents:
        agents_list = [a.strip() for a in args.agents.split(',')]
        valid_agents = ['summary', 'swot', 'metrics', 'decision']
        for agent in agents_list:
            if agent not in valid_agents:
                logger.error(f"Invalid agent: {agent}")
                logger.error(f"Valid agents: {', '.join(valid_agents)}")
                sys.exit(1)
    
    # Run agents
    runner = AgentsRunner()
    
    try:
        results = runner.run(
            doc_id=args.doc_id,
            agents=agents_list,
            risk_tolerance=args.risk_tolerance,
            investment_horizon=args.investment_horizon
        )
        
        # Print summary
        if not args.no_summary:
            runner.print_summary(results)
        
        # Save to file
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to: {output_path}")
        
        # Print file locations
        print(f"\n📁 Detailed results saved to:")
        for agent_name in results.keys():
            if agent_name != 'error':
                file_path = config.paths.final_dir / args.doc_id / f"{agent_name}_analysis.json"
                if file_path.exists():
                    print(f"  - {file_path}")
        
        print(f"\n✅ Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
