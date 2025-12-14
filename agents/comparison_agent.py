"""
Comparison Agent - Multi-Year Analysis

Compares SEC filings across multiple years to identify:
- Financial trends (revenue growth, margin changes)
- SWOT evolution (new/removed strengths/weaknesses)
- Risk factor changes (Item 1A delta)
- Strategic shifts
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from .base_agent import BaseAgent
from utils import get_logger, ProvenanceTracker

logger = get_logger("comparison_agent")


class ComparisonAgent(BaseAgent):
    """Compare multiple years of SEC filings."""
    
    def __init__(self):
        super().__init__(agent_name="Comparison Agent (YoY Trends)")
    
    def analyze(self, doc_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple filings.
        
        Args:
            doc_ids: List of document IDs to compare (e.g., ["AAPL_10-K_2023", "AAPL_10-K_2024", "AAPL_10-K_2025"])
        
        Returns:
            Comparison analysis with YoY changes
        """
        
        if len(doc_ids) < 2:
            raise ValueError("Need at least 2 documents to compare")
        
        logger.info(f"Comparing {len(doc_ids)} filings: {doc_ids}")
        
        # Load all analyses
        analyses = self._load_all_analyses(doc_ids)
        
        # Sort by year (newest first)
        analyses = sorted(analyses, key=lambda x: x['doc_id'], reverse=True)
        
        comparison = {
            "doc_ids": doc_ids,
            "num_years": len(doc_ids),
            "financial_trends": self._compare_financials(analyses),
            "swot_evolution": self._compare_swot(analyses),
            "risk_changes": self._compare_risks(analyses),
            "key_insights": self._generate_insights(analyses),
            "metadata": {
                "agent": self.agent_name,
                "analyzed_at": datetime.now().isoformat(),
                "num_docs": len(doc_ids)
            }
        }
        
        # Save comparison (use ticker as doc_id for comparison)
        ticker = doc_ids[0].split('_')[0]
        self.save_analysis(ticker, comparison, f"comparison_{ticker}")
        
        logger.success(f"✅ Comparison complete for {len(doc_ids)} years")
        
        return comparison
    
    def _load_all_analyses(self, doc_ids: List[str]) -> List[Dict]:
        """Load all analysis files for given doc IDs."""
        
        analyses = []
        
        for doc_id in doc_ids:
            doc_dir = Path(f"data/final/{doc_id}")
            
            if not doc_dir.exists():
                logger.warning(f"No analysis found for {doc_id}")
                continue
            
            analysis = {
                "doc_id": doc_id,
                "summary": self._load_json(doc_dir / "summary_analysis_v2.json"),
                "swot": self._load_json(doc_dir / "swot_analysis_v2.json"),
                "metrics": self._load_json(doc_dir / "metrics_analysis_v2.json"),
                "decision": self._load_json(doc_dir / "decision_analysis_v2.json"),
            }
            
            analyses.append(analysis)
        
        return analyses
    
    def _load_json(self, path: Path) -> Optional[Dict]:
        """Load JSON file, return None if not found."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return None
    
    def _compare_financials(self, analyses: List[Dict]) -> Dict[str, Any]:
        """Compare financial metrics across years."""
        
        logger.info("Comparing financial metrics...")
        
        metrics_timeline = []
        
        for analysis in analyses:
            if not analysis.get('metrics'):
                continue
            
            m = analysis['metrics']['metrics']
            
            year_metrics = {
                "doc_id": analysis['doc_id'],
                "revenue": m['income_statement'].get('revenue', {}).get('value', 0),
                "net_income": m['income_statement'].get('net_income', {}).get('value', 0),
                "gross_margin": m['profitability'].get('gross_margin', {}).get('value', 0),
                "operating_margin": m['profitability'].get('operating_margin', {}).get('value', 0),
                "net_margin": m['profitability'].get('net_margin', {}).get('value', 0),
            }
            
            metrics_timeline.append(year_metrics)
        
        # Calculate YoY changes
        trends = []
        for i in range(len(metrics_timeline) - 1):
            current = metrics_timeline[i]
            prior = metrics_timeline[i + 1]
            
            trends.append({
                "period": f"{prior['doc_id']} → {current['doc_id']}",
                "revenue_growth": self._calc_growth(prior['revenue'], current['revenue']),
                "net_income_growth": self._calc_growth(prior['net_income'], current['net_income']),
                "gross_margin_change": current['gross_margin'] - prior['gross_margin'],
                "operating_margin_change": current['operating_margin'] - prior['operating_margin'],
                "net_margin_change": current['net_margin'] - prior['net_margin'],
            })
        
        return {
            "timeline": metrics_timeline,
            "yoy_changes": trends,
            "summary": self._summarize_financial_trends(trends)
        }
    
    def _calc_growth(self, prior: float, current: float) -> float:
        """Calculate growth rate."""
        if prior == 0:
            return 0.0
        return ((current - prior) / prior) * 100
    
    def _summarize_financial_trends(self, trends: List[Dict]) -> Dict[str, str]:
        """Generate summary of financial trends."""
        
        if not trends:
            return {"overall": "No data available"}
        
        latest = trends[0]
        
        summary = {}
        
        # Revenue trend
        if latest['revenue_growth'] > 5:
            summary['revenue'] = f"Strong growth ({latest['revenue_growth']:.1f}%)"
        elif latest['revenue_growth'] > 0:
            summary['revenue'] = f"Modest growth ({latest['revenue_growth']:.1f}%)"
        else:
            summary['revenue'] = f"Declining ({latest['revenue_growth']:.1f}%)"
        
        # Margin trend
        if latest['gross_margin_change'] > 1:
            summary['margins'] = f"Expanding ({latest['gross_margin_change']:.1f}pp)"
        elif latest['gross_margin_change'] < -1:
            summary['margins'] = f"Contracting ({latest['gross_margin_change']:.1f}pp)"
        else:
            summary['margins'] = "Stable"
        
        return summary
    
    def _compare_swot(self, analyses: List[Dict]) -> Dict[str, Any]:
        """Compare SWOT analysis across years."""
        
        logger.info("Comparing SWOT evolution...")
        
        swot_timeline = []
        
        for analysis in analyses:
            if not analysis.get('swot'):
                continue
            
            swot = analysis['swot']['swot_analysis']
            
            year_swot = {
                "doc_id": analysis['doc_id'],
                "num_strengths": len(swot['strengths']['items']),
                "num_weaknesses": len(swot['weaknesses']['items']),
                "num_opportunities": len(swot['opportunities']['items']),
                "num_threats": len(swot['threats']['items']),
                "strengths": [s['strength'] for s in swot['strengths']['items']],
                "weaknesses": [w['weakness'] for w in swot['weaknesses']['items']],
                "opportunities": [o['opportunity'] for o in swot['opportunities']['items']],
                "threats": [t['threat'] for t in swot['threats']['items']],
            }
            
            swot_timeline.append(year_swot)
        
        # Identify new/removed items
        changes = []
        for i in range(len(swot_timeline) - 1):
            current = swot_timeline[i]
            prior = swot_timeline[i + 1]
            
            changes.append({
                "period": f"{prior['doc_id']} → {current['doc_id']}",
                "added_strengths": current['num_strengths'] - prior['num_strengths'],
                "added_weaknesses": current['num_weaknesses'] - prior['num_weaknesses'],
                "added_opportunities": current['num_opportunities'] - prior['num_opportunities'],
                "added_threats": current['num_threats'] - prior['num_threats'],
            })
        
        return {
            "timeline": swot_timeline,
            "changes": changes
        }
    
    def _compare_risks(self, analyses: List[Dict]) -> Dict[str, Any]:
        """Compare risk factors (Item 1A) across years."""
        
        logger.info("Comparing risk factor evolution...")
        
        risk_timeline = []
        
        for analysis in analyses:
            if not analysis.get('swot') or not analysis['swot'].get('risk_factor_delta'):
                continue
            
            delta = analysis['swot']['risk_factor_delta']
            
            year_risks = {
                "doc_id": analysis['doc_id'],
                "new_risks": len(delta.get('new_risks', [])),
                "heightened_risks": len(delta.get('heightened_risks', [])),
                "new_risk_items": [r['risk'] for r in delta.get('new_risks', [])],
                "heightened_risk_items": [r['risk'] for r in delta.get('heightened_risks', [])],
            }
            
            risk_timeline.append(year_risks)
        
        return {
            "timeline": risk_timeline,
            "summary": self._summarize_risk_trends(risk_timeline)
        }
    
    def _summarize_risk_trends(self, timeline: List[Dict]) -> str:
        """Generate summary of risk trends."""
        
        if not timeline:
            return "No risk data available"
        
        total_new = sum(r['new_risks'] for r in timeline)
        total_heightened = sum(r['heightened_risks'] for r in timeline)
        
        if total_new + total_heightened == 0:
            return "Risk profile stable"
        elif total_new > total_heightened:
            return f"Risk profile expanding ({total_new} new risks)"
        else:
            return f"Existing risks intensifying ({total_heightened} heightened)"
    
    def _generate_insights(self, analyses: List[Dict]) -> List[str]:
        """Generate key insights from comparison."""
        
        logger.info("Generating comparison insights...")
        
        insights = []
        
        # Financial insight
        if len(analyses) >= 2:
            latest = analyses[0]
            prior = analyses[1]
            
            if latest.get('metrics') and prior.get('metrics'):
                latest_revenue = latest['metrics']['metrics']['income_statement']['revenue']['value']
                prior_revenue = prior['metrics']['metrics']['income_statement']['revenue']['value']
                growth = self._calc_growth(prior_revenue, latest_revenue)
                
                insights.append(f"Revenue grew {growth:.1f}% YoY to ${latest_revenue:,.0f}M")
        
        # SWOT insight
        if len(analyses) >= 2:
            latest = analyses[0]
            prior = analyses[1]
            
            if latest.get('swot') and prior.get('swot'):
                latest_threats = len(latest['swot']['swot_analysis']['threats']['items'])
                prior_threats = len(prior['swot']['swot_analysis']['threats']['items'])
                
                if latest_threats > prior_threats:
                    insights.append(f"Threat profile increased by {latest_threats - prior_threats} items")
                elif latest_threats < prior_threats:
                    insights.append(f"Threat profile decreased by {prior_threats - latest_threats} items")
        
        # Decision insight
        if len(analyses) >= 2:
            latest = analyses[0]
            prior = analyses[1]
            
            if latest.get('decision') and prior.get('decision'):
                latest_rec = latest['decision']['investment_memo']['recommendation']['rating']
                prior_rec = prior['decision']['investment_memo']['recommendation']['rating']
                
                if latest_rec != prior_rec:
                    insights.append(f"Investment recommendation changed from {prior_rec} to {latest_rec}")
        
        return insights if insights else ["No significant changes detected"]
