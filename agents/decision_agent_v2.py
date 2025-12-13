"""
Enhanced Decision Agent (V2) - Investment Memo Generator

Generates decision-grade investment memos with:
- Quality scores (profitability + FCF conversion + margin stability)
- Balance sheet risk assessment
- Earnings quality checks (red flags)
- Narrative consistency validation
- Clear investment recommendation
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from .base_agent import BaseAgent
from utils import get_logger, ProvenanceTracker, config

logger = get_logger("decision_agent_v2")


class DecisionAgentV2(BaseAgent):
    """Enhanced Decision Agent with quality scores and red flags."""
    
    def __init__(self):
        """Initialize enhanced decision agent."""
        super().__init__(
            agent_name="Decision Agent V2 (Investment Memo)",
            temperature=0.2
        )
        self.provenance = ProvenanceTracker()
    
    def analyze(
        self,
        doc_id: str,
        risk_tolerance: str = "moderate",
        investment_horizon: str = "medium_term"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive investment memo.
        
        Args:
            doc_id: Document identifier
            risk_tolerance: Investor risk tolerance
            investment_horizon: Investment timeframe
        
        Returns:
            Investment memo with recommendation
        """
        logger.info(f"Generating investment memo for {doc_id}")
        
        # Load previous analyses
        summary = self._load_analysis(doc_id, "summary_analysis_v2")
        swot = self._load_analysis(doc_id, "swot_analysis_v2")
        metrics = self._load_analysis(doc_id, "metrics_analysis_v2")
        
        # Calculate quality scores
        quality_score = self._calculate_quality_score(metrics)
        balance_sheet_risk = self._assess_balance_sheet_risk(metrics)
        earnings_quality = self._check_earnings_quality(metrics)
        
        # Assess narrative consistency
        narrative_check = self._check_narrative_consistency(doc_id, summary, metrics)
        
        # Generate thesis
        bull_thesis = self._generate_bull_thesis(summary, swot, metrics)
        bear_thesis = self._generate_bear_thesis(summary, swot, metrics)
        
        # Identify red flags
        red_flags = self._identify_red_flags(earnings_quality, balance_sheet_risk, swot)
        
        # Generate final recommendation
        recommendation = self._generate_recommendation(
            quality_score=quality_score,
            balance_sheet_risk=balance_sheet_risk,
            earnings_quality=earnings_quality,
            swot=swot,
            risk_tolerance=risk_tolerance,
            investment_horizon=investment_horizon
        )
        
        # What to monitor
        monitoring_plan = self._create_monitoring_plan(swot, metrics, red_flags)
        
        # Compile investment memo
        result = {
            "doc_id": doc_id,
            "agent": self.agent_name,
            "investment_memo": {
                "recommendation": recommendation,
                "quality_score": quality_score,
                "balance_sheet_risk": balance_sheet_risk,
                "earnings_quality": earnings_quality,
                "bull_thesis": bull_thesis,
                "bear_thesis": bear_thesis,
                "red_flags": red_flags,
                "narrative_consistency": narrative_check,
                "monitoring_plan": monitoring_plan
            },
            "parameters": {
                "risk_tolerance": risk_tolerance,
                "investment_horizon": investment_horizon
            },
            "provenance": self.provenance.to_dict(),
            "metadata": {
                "analyses_available": {
                    "summary": summary is not None,
                    "swot": swot is not None,
                    "metrics": metrics is not None
                }
            }
        }
        
        # Save analysis
        self.save_analysis(doc_id, result, "decision_analysis_v2")
        
        logger.info(f"✅ Investment memo generated: {recommendation['rating']}")
        return result
    
    def _load_analysis(self, doc_id: str, filename: str) -> Optional[Dict]:
        """Load previous analysis."""
        analysis_path = config.paths.final_dir / doc_id / f"{filename}.json"
        
        if not analysis_path.exists():
            logger.warning(f"Analysis not found: {filename}")
            return None
        
        with open(analysis_path, 'r') as f:
            return json.load(f)
    
    def _calculate_quality_score(self, metrics: Optional[Dict]) -> Dict[str, Any]:
        """
        Calculate quality score (0-100).
        
        Components:
        - Profitability (margins)
        - FCF conversion
        - Margin stability
        """
        if not metrics:
            return {"score": None, "components": {}, "rating": "Unknown"}
        
        scores = {}
        m = metrics.get('metrics', {})
        
        # Profitability score (0-40 points)
        prof = m.get('profitability', {})
        gross_margin = prof.get('gross_margin', {}).get('value')
        operating_margin = prof.get('operating_margin', {}).get('value')
        net_margin = prof.get('net_margin', {}).get('value')
        
        prof_score = 0
        if gross_margin:
            if gross_margin >= 60: prof_score += 15
            elif gross_margin >= 40: prof_score += 10
            elif gross_margin >= 20: prof_score += 5
        
        if operating_margin:
            if operating_margin >= 30: prof_score += 15
            elif operating_margin >= 15: prof_score += 10
            elif operating_margin >= 5: prof_score += 5
        
        if net_margin:
            if net_margin >= 20: prof_score += 10
            elif net_margin >= 10: prof_score += 7
            elif net_margin >= 5: prof_score += 3
        
        scores['profitability'] = prof_score
        
        # FCF conversion score (0-30 points)
        cf = m.get('cash_flow_analysis', {})
        fcf_conversion = cf.get('fcf_conversion', {}).get('value')
        ocf_margin = cf.get('ocf_margin', {}).get('value')
        
        cf_score = 0
        if fcf_conversion:
            if fcf_conversion >= 100: cf_score += 20
            elif fcf_conversion >= 80: cf_score += 15
            elif fcf_conversion >= 60: cf_score += 10
        
        if ocf_margin:
            if ocf_margin >= 25: cf_score += 10
            elif ocf_margin >= 15: cf_score += 7
            elif ocf_margin >= 10: cf_score += 3
        
        scores['cash_flow'] = min(cf_score, 30)
        
        # Margin stability score (0-30 points)
        # Simplified - would need historical data for true stability
        stability_score = 20  # Base assumption
        scores['margin_stability'] = stability_score
        
        total_score = sum(scores.values())
        
        # Rating
        if total_score >= 80:
            rating = "Excellent"
        elif total_score >= 60:
            rating = "Good"
        elif total_score >= 40:
            rating = "Fair"
        else:
            rating = "Poor"
        
        return {
            "score": total_score,
            "components": scores,
            "rating": rating
        }
    
    def _assess_balance_sheet_risk(self, metrics: Optional[Dict]) -> Dict[str, Any]:
        """
        Assess balance sheet risk.
        
        Components:
        - Leverage
        - Coverage
        - Liquidity
        """
        if not metrics:
            return {"risk_level": "Unknown", "factors": []}
        
        m = metrics.get('metrics', {})
        liq = m.get('liquidity', {})
        
        risk_factors = []
        risk_score = 0  # 0 = low risk, 100 = high risk
        
        # Leverage check
        debt_to_equity = liq.get('debt_to_equity', {}).get('value')
        if debt_to_equity:
            if debt_to_equity > 2.0:
                risk_factors.append("High leverage (D/E > 2.0)")
                risk_score += 30
            elif debt_to_equity > 1.0:
                risk_factors.append("Moderate leverage (D/E > 1.0)")
                risk_score += 15
        
        # Liquidity check
        current_ratio = liq.get('current_ratio', {}).get('value')
        if current_ratio:
            if current_ratio < 1.0:
                risk_factors.append("Low liquidity (Current Ratio < 1.0)")
                risk_score += 25
            elif current_ratio < 1.5:
                risk_factors.append("Tight liquidity (Current Ratio < 1.5)")
                risk_score += 10
        
        # Quick ratio check
        quick_ratio = liq.get('quick_ratio', {}).get('value')
        if quick_ratio and quick_ratio < 0.8:
            risk_factors.append("Weak quick ratio (< 0.8)")
            risk_score += 15
        
        # Risk level
        if risk_score >= 50:
            risk_level = "High"
        elif risk_score >= 25:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "factors": risk_factors
        }
    
    def _check_earnings_quality(self, metrics: Optional[Dict]) -> Dict[str, Any]:
        """
        Check for earnings quality red flags.
        
        Checks:
        - DSO spike
        - Receivables growth > revenue growth
        - Margin compression
        - FCF < Net Income
        """
        if not metrics:
            return {"quality": "Unknown", "red_flags": []}
        
        m = metrics.get('metrics', {})
        red_flags = []
        
        # DSO check (would need historical)
        eff = m.get('efficiency', {})
        dso = eff.get('days_sales_outstanding', {}).get('value')
        if dso and dso > 90:
            red_flags.append("High DSO (> 90 days) - potential collection issues")
        
        # FCF vs Net Income
        cf = m.get('cash_flow_analysis', {})
        fcf_conv = cf.get('fcf_conversion', {}).get('value')
        if fcf_conv and fcf_conv < 70:
            red_flags.append("Low FCF conversion (< 70%) - earnings quality concern")
        
        # Margin check (would need historical for compression)
        prof = m.get('profitability', {})
        net_margin = prof.get('net_margin', {}).get('value')
        if net_margin and net_margin < 5:
            red_flags.append("Low net margin (< 5%) - profitability concern")
        
        # Quality rating
        if len(red_flags) == 0:
            quality = "High"
        elif len(red_flags) <= 1:
            quality = "Medium"
        else:
            quality = "Low"
        
        return {
            "quality": quality,
            "red_flags": red_flags,
            "num_flags": len(red_flags)
        }
    
    def _check_narrative_consistency(
        self,
        doc_id: str,
        summary: Optional[Dict],
        metrics: Optional[Dict]
    ) -> Dict[str, Any]:
        """Check if MD&A narrative matches the numbers."""
        
        if not summary or not metrics:
            return {"consistent": None, "notes": []}
        
        # This would analyze management's claims vs actual metrics
        # Simplified version
        
        notes = []
        
        # Check if positive narrative is backed by metrics
        m = metrics.get('metrics', {})
        prof = m.get('profitability', {})
        
        net_margin = prof.get('net_margin', {}).get('value')
        if net_margin and net_margin < 5:
            notes.append("Management may be overly optimistic given low margins")
        
        return {
            "consistent": len(notes) == 0,
            "notes": notes
        }
    
    def _generate_bull_thesis(
        self,
        summary: Optional[Dict],
        swot: Optional[Dict],
        metrics: Optional[Dict]
    ) -> List[str]:
        """Generate bull thesis points."""
        
        thesis_points = []
        
        if swot:
            strengths = swot.get('swot_analysis', {}).get('strengths', {}).get('items', [])
            for s in strengths[:3]:
                thesis_points.append(f"Strength: {s.get('strength', 'N/A')}")
        
        if swot:
            opps = swot.get('swot_analysis', {}).get('opportunities', {}).get('items', [])
            for o in opps[:2]:
                thesis_points.append(f"Opportunity: {o.get('opportunity', 'N/A')}")
        
        return thesis_points[:5]
    
    def _generate_bear_thesis(
        self,
        summary: Optional[Dict],
        swot: Optional[Dict],
        metrics: Optional[Dict]
    ) -> List[str]:
        """Generate bear thesis points."""
        
        thesis_points = []
        
        if swot:
            weaknesses = swot.get('swot_analysis', {}).get('weaknesses', {}).get('items', [])
            for w in weaknesses[:3]:
                thesis_points.append(f"Weakness: {w.get('weakness', 'N/A')}")
        
        if swot:
            threats = swot.get('swot_analysis', {}).get('threats', {}).get('items', [])
            for t in threats[:2]:
                thesis_points.append(f"Threat: {t.get('threat', 'N/A')}")
        
        return thesis_points[:5]
    
    def _identify_red_flags(
        self,
        earnings_quality: Dict,
        balance_sheet_risk: Dict,
        swot: Optional[Dict]
    ) -> List[str]:
        """Compile all red flags."""
        
        flags = []
        
        # Earnings quality flags
        flags.extend(earnings_quality.get('red_flags', []))
        
        # Balance sheet flags
        flags.extend(balance_sheet_risk.get('factors', []))
        
        # SWOT-based flags
        if swot:
            threats = swot.get('swot_analysis', {}).get('threats', {}).get('items', [])
            high_sev_threats = [t for t in threats if t.get('severity') == 'High']
            for t in high_sev_threats[:2]:
                flags.append(f"High-severity threat: {t.get('threat', 'N/A')}")
        
        return flags[:10]  # Top 10 flags
    
    def _generate_recommendation(
        self,
        quality_score: Dict,
        balance_sheet_risk: Dict,
        earnings_quality: Dict,
        swot: Optional[Dict],
        risk_tolerance: str,
        investment_horizon: str
    ) -> Dict[str, Any]:
        """Generate final investment recommendation."""
        
        # Score-based logic
        q_score = quality_score.get('score', 50)
        bs_risk = balance_sheet_risk.get('risk_score', 50)
        eq_flags = earnings_quality.get('num_flags', 0)
        
        # Composite score
        composite = q_score - (bs_risk * 0.5) - (eq_flags * 10)
        
        # Rating decision
        if composite >= 70:
            rating = "Strong Buy"
            confidence = "High"
        elif composite >= 55:
            rating = "Buy"
            confidence = "Medium-High"
        elif composite >= 40:
            rating = "Hold"
            confidence = "Medium"
        elif composite >= 25:
            rating = "Sell"
            confidence = "Medium"
        else:
            rating = "Strong Sell"
            confidence = "High"
        
        # Adjust for risk tolerance
        if risk_tolerance == "conservative" and bs_risk.get('risk_level') == "High":
            if rating in ["Strong Buy", "Buy"]:
                rating = "Hold"
                confidence = "Low-Medium"
        
        # Rationale
        rationale = [
            f"Quality Score: {q_score}/100 ({quality_score.get('rating')})",
            f"Balance Sheet Risk: {balance_sheet_risk.get('risk_level')}",
            f"Earnings Quality: {earnings_quality.get('quality')}",
        ]
        
        return {
            "rating": rating,
            "confidence": confidence,
            "composite_score": round(composite, 1),
            "rationale": rationale,
            "target_investor": self._describe_target_investor(
                rating, risk_tolerance, investment_horizon
            )
        }
    
    def _describe_target_investor(
        self,
        rating: str,
        risk_tolerance: str,
        horizon: str
    ) -> str:
        """Describe ideal investor profile for this investment."""
        
        if rating in ["Strong Buy", "Buy"]:
            return f"{risk_tolerance.title()} investors with {horizon.replace('_', ' ')} horizon seeking quality opportunities"
        elif rating == "Hold":
            return "Current shareholders; new investors should wait for better entry"
        else:
            return "Not recommended for most investor profiles"
    
    def _create_monitoring_plan(
        self,
        swot: Optional[Dict],
        metrics: Optional[Dict],
        red_flags: List[str]
    ) -> List[str]:
        """Create monitoring plan for next quarter/year."""
        
        monitors = []
        
        # Monitor red flags
        if red_flags:
            monitors.append(f"Track resolution of {len(red_flags)} identified red flags")
        
        # Monitor key metrics
        if metrics:
            monitors.extend([
                "Monitor quarterly margin trends",
                "Watch FCF conversion sustainability",
                "Track DSO and working capital metrics"
            ])
        
        # Monitor risks
        if swot:
            threats = swot.get('swot_analysis', {}).get('threats', {}).get('items', [])
            if threats:
                monitors.append(f"Monitor {len(threats)} identified threat factors")
        
        return monitors[:7]


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate investment memo")
    parser.add_argument("--doc-id", required=True, help="Document ID")
    parser.add_argument("--risk-tolerance", default="moderate",
                       choices=["conservative", "moderate", "aggressive"])
    parser.add_argument("--horizon", default="medium_term",
                       choices=["short_term", "medium_term", "long_term"])
    
    args = parser.parse_args()
    
    agent = DecisionAgentV2()
    result = agent.analyze(
        args.doc_id,
        risk_tolerance=args.risk_tolerance,
        investment_horizon=args.horizon
    )
    
    memo = result['investment_memo']
    
    print("\n" + "="*80)
    print("INVESTMENT MEMO")
    print("="*80)
    
    rec = memo['recommendation']
    print(f"\n🎯 RECOMMENDATION: {rec['rating']}")
    print(f"📊 Confidence: {rec['confidence']}")
    print(f"💯 Composite Score: {rec['composite_score']}/100")
    
    print("\n📈 BULL THESIS:")
    for point in memo['bull_thesis']:
        print(f"  • {point}")
    
    print("\n📉 BEAR THESIS:")
    for point in memo['bear_thesis']:
        print(f"  • {point}")
    
    print("\n🚩 RED FLAGS:")
    for flag in memo['red_flags']:
        print(f"  • {flag}")
    
    print("\n👀 MONITORING PLAN:")
    for item in memo['monitoring_plan']:
        print(f"  • {item}")


if __name__ == "__main__":
    main()
