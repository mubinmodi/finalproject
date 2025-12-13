"""
Enhanced Metrics Agent (V2) - Main Quant Engine

Comprehensive KPI extraction with strict provenance:
- Standard financial ratios
- Industry-specific metrics from MD&A
- Formula documentation
- Source validation (XBRL > Tables > Notes)
"""

import json
import csv
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from pydantic import BaseModel
from .base_agent import BaseAgent
from utils import get_logger, ProvenanceTracker, config

logger = get_logger("metrics_agent_v2")


class KPI(BaseModel):
    """Structured KPI with full provenance."""
    
    name: str
    value: float
    formula: str
    numerator: Dict[str, Any]  # {value, source, line_item}
    denominator: Optional[Dict[str, Any]] = None
    fiscal_period: str
    units: Optional[str] = None
    source: str  # "XBRL", "Table", "Calculated"
    provenance: Dict[str, Any]  # {table_id, page, line_num}


class MetricsAgentV2(BaseAgent):
    """Enhanced Metrics Agent with comprehensive KPI extraction."""
    
    def __init__(self):
        """Initialize enhanced metrics agent."""
        super().__init__(
            agent_name="Metrics Agent V2 (Quant Engine)",
            temperature=0.0  # Zero temperature for precise numbers
        )
        self.provenance = ProvenanceTracker()
    
    def analyze(self, doc_id: str) -> Dict[str, Any]:
        """
        Extract comprehensive financial metrics with provenance.
        
        Args:
            doc_id: Document identifier
        
        Returns:
            Dictionary containing all KPIs with full provenance
        """
        logger.info(f"Extracting comprehensive metrics for {doc_id}")
        
        # Load data sources (priority order)
        xbrl_facts = self._load_xbrl_facts(doc_id)
        tables = self._load_tables(doc_id)
        
        # Extract core financial statement values
        income_statement = self._extract_income_statement(xbrl_facts, tables)
        balance_sheet = self._extract_balance_sheet(xbrl_facts, tables)
        cash_flow = self._extract_cash_flow(xbrl_facts, tables)
        
        # Calculate standard KPIs
        profitability = self._calculate_profitability(income_statement)
        liquidity = self._calculate_liquidity(balance_sheet)
        efficiency = self._calculate_efficiency(income_statement, balance_sheet)
        cash_flow_metrics = self._calculate_cash_flow_metrics(cash_flow, income_statement)
        
        # Extract industry-specific metrics from MD&A
        industry_metrics = self._extract_industry_metrics(doc_id)
        
        # Compile results
        result = {
            "doc_id": doc_id,
            "agent": self.agent_name,
            "metrics": {
                "income_statement": income_statement,
                "balance_sheet": balance_sheet,
                "cash_flow": cash_flow,
                "profitability": profitability,
                "liquidity": liquidity,
                "efficiency": efficiency,
                "cash_flow_analysis": cash_flow_metrics,
                "industry_specific": industry_metrics
            },
            "provenance": self.provenance.to_dict(),
            "metadata": {
                "xbrl_facts_count": len(xbrl_facts),
                "tables_analyzed": len(tables),
                "total_kpis": self._count_kpis(profitability, liquidity, efficiency, cash_flow_metrics)
            }
        }
        
        # Save analysis
        self.save_analysis(doc_id, result, "metrics_analysis_v2")
        
        logger.info(f"✅ Comprehensive metrics extracted")
        return result
    
    def _load_xbrl_facts(self, doc_id: str) -> List[Dict]:
        """Load XBRL facts."""
        try:
            xbrl_path = config.paths.processed_dir / doc_id / "xbrl_facts.jsonl"
            facts = []
            with open(xbrl_path, 'r') as f:
                for line in f:
                    facts.append(json.loads(line.strip()))
            return facts
        except:
            logger.warning("Could not load XBRL facts")
            return []
    
    def _load_tables(self, doc_id: str) -> List[Dict]:
        """Load table metadata and data."""
        try:
            tables_path = config.paths.processed_dir / doc_id / "tables_index.jsonl"
            tables = []
            with open(tables_path, 'r') as f:
                for line in f:
                    table_meta = json.loads(line.strip())
                    
                    # Load CSV data
                    csv_path = Path(table_meta['csv_path'])
                    if csv_path.exists():
                        with open(csv_path, 'r') as csvf:
                            reader = csv.reader(csvf)
                            table_meta['data'] = list(reader)
                    
                    tables.append(table_meta)
            return tables
        except:
            logger.warning("Could not load tables")
            return []
    
    def _extract_income_statement(
        self,
        xbrl_facts: List[Dict],
        tables: List[Dict]
    ) -> Dict[str, Any]:
        """Extract income statement line items."""
        
        # Map of concepts to search for (XBRL priority)
        concepts = {
            "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
            "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold"],
            "gross_profit": ["GrossProfit"],
            "research_development": ["ResearchAndDevelopmentExpense"],
            "selling_general_admin": ["SellingGeneralAndAdministrativeExpense"],
            "operating_expenses": ["OperatingExpenses"],
            "operating_income": ["OperatingIncomeLoss"],
            "interest_expense": ["InterestExpense"],
            "income_before_tax": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"],
            "tax_expense": ["IncomeTaxExpenseBenefit"],
            "net_income": ["NetIncomeLoss", "ProfitLoss"],
            "eps_basic": ["EarningsPerShareBasic"],
            "eps_diluted": ["EarningsPerShareDiluted"],
        }
        
        line_items = {}
        
        for item_name, concept_list in concepts.items():
            value, source = self._find_value_in_sources(concept_list, xbrl_facts, tables)
            if value is not None:
                line_items[item_name] = {
                    "value": value,
                    "source": source,
                    "concept": concept_list[0] if source == "XBRL" else None
                }
        
        return line_items
    
    def _extract_balance_sheet(
        self,
        xbrl_facts: List[Dict],
        tables: List[Dict]
    ) -> Dict[str, Any]:
        """Extract balance sheet line items."""
        
        concepts = {
            # Assets
            "cash": ["Cash", "CashAndCashEquivalentsAtCarryingValue"],
            "short_term_investments": ["ShortTermInvestments"],
            "accounts_receivable": ["AccountsReceivableNetCurrent"],
            "inventory": ["InventoryNet"],
            "current_assets": ["AssetsCurrent"],
            "ppe_net": ["PropertyPlantAndEquipmentNet"],
            "goodwill": ["Goodwill"],
            "intangible_assets": ["IntangibleAssetsNetExcludingGoodwill"],
            "total_assets": ["Assets"],
            
            # Liabilities
            "accounts_payable": ["AccountsPayableCurrent"],
            "short_term_debt": ["DebtCurrent"],
            "current_liabilities": ["LiabilitiesCurrent"],
            "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
            "total_liabilities": ["Liabilities"],
            
            # Equity
            "stockholders_equity": ["StockholdersEquity"],
            "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
        }
        
        line_items = {}
        
        for item_name, concept_list in concepts.items():
            value, source = self._find_value_in_sources(concept_list, xbrl_facts, tables)
            if value is not None:
                line_items[item_name] = {
                    "value": value,
                    "source": source,
                    "concept": concept_list[0] if source == "XBRL" else None
                }
        
        return line_items
    
    def _extract_cash_flow(
        self,
        xbrl_facts: List[Dict],
        tables: List[Dict]
    ) -> Dict[str, Any]:
        """Extract cash flow statement line items."""
        
        concepts = {
            "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
            "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
            "investing_cash_flow": ["NetCashProvidedByUsedInInvestingActivities"],
            "financing_cash_flow": ["NetCashProvidedByUsedInFinancingActivities"],
            "dividends_paid": ["PaymentsOfDividends"],
            "stock_repurchases": ["PaymentsForRepurchaseOfCommonStock"],
            "free_cash_flow": ["FreeCashFlow"],  # May not exist, will calculate
        }
        
        line_items = {}
        
        for item_name, concept_list in concepts.items():
            value, source = self._find_value_in_sources(concept_list, xbrl_facts, tables)
            if value is not None:
                line_items[item_name] = {
                    "value": value,
                    "source": source,
                    "concept": concept_list[0] if source == "XBRL" else None
                }
        
        # Calculate FCF if not provided
        if "free_cash_flow" not in line_items:
            if "operating_cash_flow" in line_items and "capex" in line_items:
                ocf = line_items["operating_cash_flow"]["value"]
                capex = abs(line_items["capex"]["value"])  # CapEx usually negative
                fcf = ocf - capex
                
                line_items["free_cash_flow"] = {
                    "value": fcf,
                    "source": "Calculated",
                    "formula": "Operating Cash Flow - CapEx"
                }
        
        return line_items
    
    def _find_value_in_sources(
        self,
        concepts: List[str],
        xbrl_facts: List[Dict],
        tables: List[Dict]
    ) -> Tuple[Optional[float], str]:
        """
        Find value with source priority: XBRL > Tables.
        
        Returns:
            (value, source) tuple
        """
        # Try XBRL first
        for concept in concepts:
            for fact in xbrl_facts:
                if concept.lower() in fact['concept'].lower():
                    try:
                        value = float(fact['value'])
                        return (value, "XBRL")
                    except:
                        continue
        
        # Try tables (basic keyword matching)
        # In production, would use more sophisticated table parsing
        for concept in concepts:
            for table in tables:
                if 'data' in table:
                    # Search for concept in table
                    # This is simplified - real implementation would be more robust
                    pass
        
        return (None, "Not Found")
    
    def _calculate_profitability(self, income_statement: Dict) -> Dict[str, Any]:
        """Calculate profitability ratios."""
        
        ratios = {}
        
        revenue = self._get_value(income_statement, "revenue")
        gross_profit = self._get_value(income_statement, "gross_profit")
        cost_of_revenue = self._get_value(income_statement, "cost_of_revenue")
        operating_income = self._get_value(income_statement, "operating_income")
        net_income = self._get_value(income_statement, "net_income")
        
        # Gross Margin
        if gross_profit and revenue and revenue != 0:
            ratios["gross_margin"] = {
                "value": (gross_profit / revenue) * 100,
                "formula": "(Gross Profit / Revenue) * 100",
                "numerator": {"name": "Gross Profit", "value": gross_profit},
                "denominator": {"name": "Revenue", "value": revenue},
                "units": "%"
            }
        elif cost_of_revenue and revenue and revenue != 0:
            gp = revenue - cost_of_revenue
            ratios["gross_margin"] = {
                "value": (gp / revenue) * 100,
                "formula": "((Revenue - Cost of Revenue) / Revenue) * 100",
                "numerator": {"name": "Revenue - COGS", "value": gp},
                "denominator": {"name": "Revenue", "value": revenue},
                "units": "%"
            }
        
        # Operating Margin
        if operating_income and revenue and revenue != 0:
            ratios["operating_margin"] = {
                "value": (operating_income / revenue) * 100,
                "formula": "(Operating Income / Revenue) * 100",
                "numerator": {"name": "Operating Income", "value": operating_income},
                "denominator": {"name": "Revenue", "value": revenue},
                "units": "%"
            }
        
        # Net Margin
        if net_income and revenue and revenue != 0:
            ratios["net_margin"] = {
                "value": (net_income / revenue) * 100,
                "formula": "(Net Income / Revenue) * 100",
                "numerator": {"name": "Net Income", "value": net_income},
                "denominator": {"name": "Revenue", "value": revenue},
                "units": "%"
            }
        
        # EBITDA Margin (simplified - would need D&A from cash flow)
        # Not calculated here without depreciation data
        
        return ratios
    
    def _calculate_liquidity(self, balance_sheet: Dict) -> Dict[str, Any]:
        """Calculate liquidity and solvency ratios."""
        
        ratios = {}
        
        current_assets = self._get_value(balance_sheet, "current_assets")
        current_liabilities = self._get_value(balance_sheet, "current_liabilities")
        cash = self._get_value(balance_sheet, "cash")
        accounts_receivable = self._get_value(balance_sheet, "accounts_receivable")
        inventory = self._get_value(balance_sheet, "inventory")
        total_assets = self._get_value(balance_sheet, "total_assets")
        total_liabilities = self._get_value(balance_sheet, "total_liabilities")
        long_term_debt = self._get_value(balance_sheet, "long_term_debt")
        equity = self._get_value(balance_sheet, "stockholders_equity")
        
        # Current Ratio
        if current_assets and current_liabilities and current_liabilities != 0:
            ratios["current_ratio"] = {
                "value": current_assets / current_liabilities,
                "formula": "Current Assets / Current Liabilities",
                "numerator": {"name": "Current Assets", "value": current_assets},
                "denominator": {"name": "Current Liabilities", "value": current_liabilities},
                "units": "x"
            }
        
        # Quick Ratio
        if current_assets and inventory and current_liabilities and current_liabilities != 0:
            quick_assets = current_assets - inventory
            ratios["quick_ratio"] = {
                "value": quick_assets / current_liabilities,
                "formula": "(Current Assets - Inventory) / Current Liabilities",
                "numerator": {"name": "Quick Assets", "value": quick_assets},
                "denominator": {"name": "Current Liabilities", "value": current_liabilities},
                "units": "x"
            }
        
        # Debt to Equity
        if long_term_debt and equity and equity != 0:
            ratios["debt_to_equity"] = {
                "value": long_term_debt / equity,
                "formula": "Long-term Debt / Stockholders' Equity",
                "numerator": {"name": "Long-term Debt", "value": long_term_debt},
                "denominator": {"name": "Equity", "value": equity},
                "units": "x"
            }
        
        # Debt to Assets
        if total_liabilities and total_assets and total_assets != 0:
            ratios["debt_to_assets"] = {
                "value": total_liabilities / total_assets,
                "formula": "Total Liabilities / Total Assets",
                "numerator": {"name": "Total Liabilities", "value": total_liabilities},
                "denominator": {"name": "Total Assets", "value": total_assets},
                "units": "x"
            }
        
        return ratios
    
    def _calculate_efficiency(
        self,
        income_statement: Dict,
        balance_sheet: Dict
    ) -> Dict[str, Any]:
        """Calculate efficiency ratios."""
        
        ratios = {}
        
        revenue = self._get_value(income_statement, "revenue")
        cogs = self._get_value(income_statement, "cost_of_revenue")
        inventory = self._get_value(balance_sheet, "inventory")
        accounts_receivable = self._get_value(balance_sheet, "accounts_receivable")
        total_assets = self._get_value(balance_sheet, "total_assets")
        
        # Inventory Turnover
        if cogs and inventory and inventory != 0:
            ratios["inventory_turnover"] = {
                "value": cogs / inventory,
                "formula": "Cost of Revenue / Inventory",
                "numerator": {"name": "Cost of Revenue", "value": cogs},
                "denominator": {"name": "Inventory", "value": inventory},
                "units": "x"
            }
        
        # Days Sales Outstanding (DSO)
        if revenue and accounts_receivable:
            days = 365
            dso = (accounts_receivable / revenue) * days
            ratios["days_sales_outstanding"] = {
                "value": dso,
                "formula": "(Accounts Receivable / Revenue) * 365",
                "numerator": {"name": "Accounts Receivable", "value": accounts_receivable},
                "denominator": {"name": "Daily Revenue", "value": revenue / days},
                "units": "days"
            }
        
        # Asset Turnover
        if revenue and total_assets and total_assets != 0:
            ratios["asset_turnover"] = {
                "value": revenue / total_assets,
                "formula": "Revenue / Total Assets",
                "numerator": {"name": "Revenue", "value": revenue},
                "denominator": {"name": "Total Assets", "value": total_assets},
                "units": "x"
            }
        
        return ratios
    
    def _calculate_cash_flow_metrics(
        self,
        cash_flow: Dict,
        income_statement: Dict
    ) -> Dict[str, Any]:
        """Calculate cash flow metrics."""
        
        metrics = {}
        
        fcf = self._get_value(cash_flow, "free_cash_flow")
        operating_cf = self._get_value(cash_flow, "operating_cash_flow")
        capex = self._get_value(cash_flow, "capex")
        net_income = self._get_value(income_statement, "net_income")
        revenue = self._get_value(income_statement, "revenue")
        
        # FCF Conversion (FCF / Net Income)
        if fcf and net_income and net_income != 0:
            metrics["fcf_conversion"] = {
                "value": (fcf / net_income) * 100,
                "formula": "(Free Cash Flow / Net Income) * 100",
                "numerator": {"name": "FCF", "value": fcf},
                "denominator": {"name": "Net Income", "value": net_income},
                "units": "%"
            }
        
        # CapEx Intensity
        if capex and revenue and revenue != 0:
            metrics["capex_intensity"] = {
                "value": (abs(capex) / revenue) * 100,
                "formula": "(CapEx / Revenue) * 100",
                "numerator": {"name": "CapEx", "value": abs(capex)},
                "denominator": {"name": "Revenue", "value": revenue},
                "units": "%"
            }
        
        # Operating Cash Flow Margin
        if operating_cf and revenue and revenue != 0:
            metrics["ocf_margin"] = {
                "value": (operating_cf / revenue) * 100,
                "formula": "(Operating Cash Flow / Revenue) * 100",
                "numerator": {"name": "Operating CF", "value": operating_cf},
                "denominator": {"name": "Revenue", "value": revenue},
                "units": "%"
            }
        
        return metrics
    
    def _extract_industry_metrics(self, doc_id: str) -> Dict[str, Any]:
        """Extract industry-specific metrics from MD&A using RAG."""
        
        vector_store = self.create_vector_store(doc_id)
        
        # Common industry metrics to search for
        metric_queries = [
            "ARR (Annual Recurring Revenue)",
            "NRR (Net Revenue Retention)",
            "CAC (Customer Acquisition Cost)",
            "LTV (Lifetime Value)",
            "Churn rate",
            "ARPU (Average Revenue Per User)",
            "NIM (Net Interest Margin)",
            "Same-store sales",
            "Occupancy rate",
            "Yield metrics",
        ]
        
        found_metrics = []
        
        for metric_name in metric_queries:
            chunks = self.retrieve_relevant_chunks(
                doc_id=doc_id,
                query=f"What is the {metric_name} metric disclosed?",
                k=2,
                vector_store=vector_store
            )
            
            if chunks:
                # Check if metric is actually mentioned
                text = chunks[0]['text']
                if any(term.lower() in text.lower() for term in metric_name.split()):
                    found_metrics.append({
                        "metric": metric_name,
                        "context": text[:300],
                        "page": chunks[0].get('page', 0)
                    })
        
        return {
            "found_metrics": found_metrics,
            "count": len(found_metrics)
        }
    
    def _get_value(self, data: Dict, key: str) -> Optional[float]:
        """Safely extract value from data dict."""
        if key in data and data[key]:
            return data[key].get('value')
        return None
    
    def _count_kpis(self, *metric_dicts) -> int:
        """Count total KPIs calculated."""
        return sum(len(d) for d in metric_dicts if isinstance(d, dict))


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract comprehensive metrics")
    parser.add_argument("--doc-id", required=True, help="Document ID")
    
    args = parser.parse_args()
    
    agent = MetricsAgentV2()
    result = agent.analyze(args.doc_id)
    
    metrics = result['metrics']
    
    print("\n" + "="*80)
    print("COMPREHENSIVE FINANCIAL METRICS")
    print("="*80)
    
    print("\n💰 PROFITABILITY RATIOS:")
    for name, data in metrics['profitability'].items():
        print(f"  {name.replace('_', ' ').title()}: {data['value']:.2f}{data['units']}")
        print(f"    Formula: {data['formula']}")
    
    print("\n🏦 LIQUIDITY & SOLVENCY:")
    for name, data in metrics['liquidity'].items():
        print(f"  {name.replace('_', ' ').title()}: {data['value']:.2f}{data['units']}")
    
    print("\n⚡ EFFICIENCY RATIOS:")
    for name, data in metrics['efficiency'].items():
        print(f"  {name.replace('_', ' ').title()}: {data['value']:.2f}{data['units']}")
    
    print("\n💵 CASH FLOW ANALYSIS:")
    for name, data in metrics['cash_flow_analysis'].items():
        print(f"  {name.replace('_', ' ').title()}: {data['value']:.2f}{data['units']}")
    
    if metrics['industry_specific']['found_metrics']:
        print("\n🎯 INDUSTRY-SPECIFIC METRICS:")
        for metric in metrics['industry_specific']['found_metrics']:
            print(f"  • {metric['metric']} (Page {metric['page'] + 1})")


if __name__ == "__main__":
    main()
