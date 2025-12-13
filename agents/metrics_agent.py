"""
Metrics Agent (Primary)

Extracts and validates key financial metrics from SEC filings:
- Revenue, net income, EPS
- Balance sheet metrics (assets, liabilities, equity)
- Cash flow metrics
- Key ratios (margins, ROE, liquidity, leverage)
- Trend analysis (YoY, QoQ growth)
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from .base_agent import BaseAgent
from utils import config, get_logger

logger = get_logger("metrics_agent")


class MetricsAgent(BaseAgent):
    """Agent for financial metrics extraction and analysis."""
    
    def __init__(self):
        """Initialize metrics agent."""
        super().__init__(
            agent_name="Metrics Agent",
            temperature=0.1  # Very low temperature for precise numbers
        )
    
    def analyze(self, doc_id: str) -> Dict[str, Any]:
        """
        Extract and analyze financial metrics.
        
        Args:
            doc_id: Document identifier
        
        Returns:
            Dictionary containing financial metrics and analysis
        """
        logger.info(f"Extracting financial metrics for {doc_id}")
        
        # Load XBRL facts (primary source of truth)
        xbrl_metrics = self._extract_xbrl_metrics(doc_id)
        
        # Extract table-based metrics as backup/validation
        table_metrics = self._extract_table_metrics(doc_id)
        
        # Calculate derived metrics and ratios
        ratios = self._calculate_ratios(xbrl_metrics)
        
        # Perform trend analysis
        trends = self._analyze_trends(doc_id)
        
        # Generate metrics summary
        summary = self._generate_metrics_summary(xbrl_metrics, ratios, trends)
        
        # Compile results
        result = {
            "doc_id": doc_id,
            "agent": self.agent_name,
            "metrics": {
                "xbrl_metrics": xbrl_metrics,
                "table_metrics": table_metrics,
                "calculated_ratios": ratios,
                "trends": trends
            },
            "summary": summary,
            "metadata": {
                "xbrl_facts_count": len(self._load_xbrl_facts(doc_id)),
                "tables_analyzed": len(self._load_tables(doc_id))
            }
        }
        
        # Save analysis
        self.save_analysis(doc_id, result, "metrics_analysis")
        
        logger.info(f"✅ Metrics analysis complete for {doc_id}")
        return result
    
    def _load_xbrl_facts(self, doc_id: str) -> List[Dict]:
        """Load XBRL facts."""
        xbrl_path = config.paths.processed_dir / doc_id / "xbrl_facts.jsonl"
        
        if not xbrl_path.exists():
            return []
        
        facts = []
        with open(xbrl_path, 'r') as f:
            for line in f:
                facts.append(json.loads(line.strip()))
        
        return facts
    
    def _load_tables(self, doc_id: str) -> List[Dict]:
        """Load table metadata."""
        tables_path = config.paths.processed_dir / doc_id / "tables_index.jsonl"
        
        if not tables_path.exists():
            return []
        
        tables = []
        with open(tables_path, 'r') as f:
            for line in f:
                tables.append(json.loads(line.strip()))
        
        return tables
    
    def _extract_xbrl_metrics(self, doc_id: str) -> Dict[str, Any]:
        """Extract key metrics from XBRL facts."""
        facts = self._load_xbrl_facts(doc_id)
        
        # Define key metric concepts
        metric_concepts = {
            # Income Statement
            "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
            "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold"],
            "gross_profit": ["GrossProfit"],
            "operating_income": ["OperatingIncomeLoss"],
            "net_income": ["NetIncomeLoss", "ProfitLoss"],
            "eps_basic": ["EarningsPerShareBasic"],
            "eps_diluted": ["EarningsPerShareDiluted"],
            
            # Balance Sheet
            "assets": ["Assets"],
            "current_assets": ["AssetsCurrent"],
            "liabilities": ["Liabilities"],
            "current_liabilities": ["LiabilitiesCurrent"],
            "equity": ["StockholdersEquity", "ShareholdersEquity"],
            "cash": ["Cash", "CashAndCashEquivalentsAtCarryingValue"],
            "debt": ["LongTermDebt", "DebtCurrent"],
            
            # Cash Flow
            "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
            "investing_cash_flow": ["NetCashProvidedByUsedInInvestingActivities"],
            "financing_cash_flow": ["NetCashProvidedByUsedInFinancingActivities"],
            "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
        }
        
        metrics = {}
        
        for metric_name, concepts in metric_concepts.items():
            for concept in concepts:
                # Find matching facts
                matching_facts = [f for f in facts if concept.lower() in f['concept'].lower()]
                
                if matching_facts:
                    # Take the first matching fact (most recent period)
                    fact = matching_facts[0]
                    
                    try:
                        value = float(fact['value']) if fact['value'] else None
                    except (ValueError, TypeError):
                        value = fact['value']
                    
                    metrics[metric_name] = {
                        "value": value,
                        "units": fact.get('units'),
                        "period": fact.get('period'),
                        "concept": fact['concept']
                    }
                    break
        
        return metrics
    
    def _extract_table_metrics(self, doc_id: str) -> Dict[str, Any]:
        """Extract metrics from tables (backup/validation)."""
        tables = self._load_tables(doc_id)
        
        # For now, just return table references
        # Full implementation would parse CSV files
        
        financial_tables = []
        
        for table in tables:
            # Identify financial statement tables by keywords
            table_id = table.get('table_id', '').lower()
            caption = table.get('caption', '').lower()
            
            if any(keyword in table_id + caption for keyword in [
                'statement', 'balance', 'income', 'cash', 'operation', 'financial'
            ]):
                financial_tables.append({
                    "table_id": table['table_id'],
                    "page": table['page'],
                    "csv_path": table['csv_path'],
                    "quality": table.get('quality_score')
                })
        
        return {"financial_tables": financial_tables}
    
    def _calculate_ratios(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate financial ratios from extracted metrics."""
        ratios = {}
        
        try:
            # Get values (safely)
            revenue = self._get_metric_value(metrics, 'revenue')
            cost_of_revenue = self._get_metric_value(metrics, 'cost_of_revenue')
            gross_profit = self._get_metric_value(metrics, 'gross_profit')
            operating_income = self._get_metric_value(metrics, 'operating_income')
            net_income = self._get_metric_value(metrics, 'net_income')
            assets = self._get_metric_value(metrics, 'assets')
            current_assets = self._get_metric_value(metrics, 'current_assets')
            liabilities = self._get_metric_value(metrics, 'liabilities')
            current_liabilities = self._get_metric_value(metrics, 'current_liabilities')
            equity = self._get_metric_value(metrics, 'equity')
            debt = self._get_metric_value(metrics, 'debt')
            
            # Profitability Ratios
            if revenue and revenue != 0:
                if gross_profit is not None:
                    ratios['gross_margin'] = (gross_profit / revenue) * 100
                elif cost_of_revenue is not None:
                    ratios['gross_margin'] = ((revenue - cost_of_revenue) / revenue) * 100
                
                if operating_income is not None:
                    ratios['operating_margin'] = (operating_income / revenue) * 100
                
                if net_income is not None:
                    ratios['net_margin'] = (net_income / revenue) * 100
            
            # Return Ratios
            if equity and equity != 0 and net_income is not None:
                ratios['roe'] = (net_income / equity) * 100
            
            if assets and assets != 0 and net_income is not None:
                ratios['roa'] = (net_income / assets) * 100
            
            # Liquidity Ratios
            if current_liabilities and current_liabilities != 0:
                if current_assets is not None:
                    ratios['current_ratio'] = current_assets / current_liabilities
            
            # Leverage Ratios
            if equity and equity != 0 and debt is not None:
                ratios['debt_to_equity'] = debt / equity
            
            if assets and assets != 0 and debt is not None:
                ratios['debt_to_assets'] = debt / assets
        
        except Exception as e:
            logger.warning(f"Error calculating ratios: {e}")
        
        return ratios
    
    def _get_metric_value(self, metrics: Dict, key: str) -> Optional[float]:
        """Safely get metric value."""
        if key in metrics and metrics[key]:
            value = metrics[key].get('value')
            if isinstance(value, (int, float)):
                return float(value)
        return None
    
    def _analyze_trends(self, doc_id: str) -> Dict[str, Any]:
        """Analyze trends (requires historical data)."""
        # For now, return placeholder
        # Full implementation would compare with previous filings
        
        return {
            "note": "Trend analysis requires multiple filings from the same company",
            "available": False,
            "growth_metrics": {}
        }
    
    def _generate_metrics_summary(
        self,
        metrics: Dict,
        ratios: Dict,
        trends: Dict
    ) -> str:
        """Generate summary of financial metrics."""
        
        # Format metrics for prompt
        metrics_text = []
        
        for key, value in metrics.items():
            if value and isinstance(value, dict):
                val = value.get('value')
                units = value.get('units', '')
                if val is not None:
                    metrics_text.append(f"- {key.replace('_', ' ').title()}: {val:,} {units}")
        
        ratios_text = []
        for key, value in ratios.items():
            if value is not None:
                ratios_text.append(f"- {key.replace('_', ' ').title()}: {value:.2f}%")
        
        prompt = f"""
Provide a comprehensive financial analysis summary based on the following metrics:

KEY METRICS:
{chr(10).join(metrics_text) if metrics_text else 'Not available'}

FINANCIAL RATIOS:
{chr(10).join(ratios_text) if ratios_text else 'Not available'}

Summary should include:
1. Overall financial health assessment
2. Key strengths in the financial position
3. Areas of concern or weakness
4. Comparison to industry benchmarks (if applicable)
5. Investment implications

Financial Analysis Summary:
"""
        
        return self.generate_response(prompt)


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract financial metrics")
    parser.add_argument("--doc-id", required=True, help="Document ID")
    
    args = parser.parse_args()
    
    agent = MetricsAgent()
    result = agent.analyze(args.doc_id)
    
    print("\n" + "="*80)
    print("FINANCIAL METRICS ANALYSIS")
    print("="*80)
    
    # Display key metrics
    metrics = result['metrics']['xbrl_metrics']
    print("\n📊 KEY METRICS:")
    for key, value in metrics.items():
        if value:
            val = value.get('value')
            units = value.get('units', '')
            print(f"  {key.replace('_', ' ').title()}: {val} {units}")
    
    # Display ratios
    ratios = result['metrics']['calculated_ratios']
    if ratios:
        print("\n📈 FINANCIAL RATIOS:")
        for key, value in ratios.items():
            print(f"  {key.replace('_', ' ').title()}: {value:.2f}%")
    
    # Display summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(result['summary'])


if __name__ == "__main__":
    main()
