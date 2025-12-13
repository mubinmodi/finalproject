# Project Green Lattern V2 - Enhanced Agent Workflow

## 🎉 What's New in V2

### Enhanced Agents with Provenance

All V2 agents now include:
- **Full citation tracking** with page numbers, sections, and extraction methods
- **Evidence-based analysis** - no generic claims without concrete backing
- **Year-over-year delta analysis** - compare filings across years
- **Quality scoring** - quantitative assessment of investment quality
- **Red flags detection** - earnings quality and accounting smoke signals

### Agent Enhancements

#### Summary Agent V2 (`summary_agent_v2.py`)
- **Key Findings** (5-10 bullets) from priority sections
- **Company Direction** (1 paragraph strategic overview)
- **Delta Analysis** - what changed vs last year
- **Financial Highlights** - headline metrics from XBRL
- **New Risk Factors** - highlighted from Item 1A

**Example Output:**
```json
{
  "key_findings": [
    {
      "finding": "Revenue grew 15% YoY driven by new product launches",
      "section": "Item 7",
      "page": 45,
      "citation_ids": [12, 13]
    }
  ],
  "company_direction": {
    "text": "The company is pivoting to AI-driven solutions...",
    "citation_ids": [5, 6, 7]
  }
}
```

#### SWOT Agent V2 (`swot_agent_v2.py`) - **Hostile Witness Mode**
- **Evidence-driven** - "No generic 'strong brand' unless backed by data"
- **Specific data targets**:
  - Strengths: pricing power (Item 7), margin stability (Item 8), IP (Item 1)
  - Weaknesses: liquidity constraints, dependencies (Item 1A, 7)
  - Opportunities: CapEx plans, TAM expansion (Item 7, 1)
  - Threats: regulatory risks, competitive pressure (Item 1A, 3, 7)
- **Risk Factor Delta** - YoY comparison of Item 1A:
  - Added risks → new threats
  - Removed risks → threat reduction
  - Language shifts → sentiment changes

**Example Output:**
```json
{
  "swot_analysis": {
    "strengths": {
      "items": [
        {
          "strength": "Demonstrated pricing power with 8% price increases absorbed by customers",
          "evidence": "Management disclosed successful 8% price increase in Q2 with no volume impact",
          "significance": "High",
          "category": "Pricing Power",
          "citation_ids": [23],
          "page": 47
        }
      ]
    }
  },
  "risk_factor_delta": {
    "added_risks": ["New cybersecurity threat", "Supply chain dependency"],
    "heightened_risks": ["Regulatory compliance costs increased"],
    "sentiment_shifts": ["Language changed from 'managing' to 'challenged by'"]
  }
}
```

#### Metrics Agent V2 (`metrics_agent_v2.py`) - **Quant Engine**
- **Comprehensive KPI extraction** with strict provenance
- **Priority of sources**: XBRL → Tables → Notes → MD&A
- **For each KPI**:
  - Value
  - Formula
  - Numerator + denominator values + sources
  - Fiscal period alignment
  - Exact provenance (table/line/page)
- **Standard ratios**:
  - Profitability: Gross/Operating/Net margins, ROE, ROA
  - Liquidity: Current, Quick, D/E, Interest Coverage
  - Efficiency: Inventory Turnover, DSO, Asset Turnover
  - Cash Flow: FCF, FCF Conversion, CapEx Intensity
- **Industry-specific metrics** (RAG extraction from Item 7):
  - ARR, NRR, CAC, LTV (SaaS)
  - NIM (Banking)
  - Same-store sales (Retail)
  - Occupancy rate (Real Estate)

**Example Output:**
```json
{
  "metrics": {
    "profitability": {
      "gross_margin": {
        "value": 42.5,
        "formula": "(Gross Profit / Revenue) * 100",
        "numerator": {
          "name": "Gross Profit",
          "value": 85000000,
          "source": "XBRL",
          "concept": "GrossProfit"
        },
        "denominator": {
          "name": "Revenue",
          "value": 200000000,
          "source": "XBRL",
          "concept": "Revenues"
        },
        "units": "%"
      }
    }
  }
}
```

#### Decision Agent V2 (`decision_agent_v2.py`) - **Investment Memo**
- **Quality Score** (0-100):
  - Profitability component (margins)
  - FCF conversion component
  - Margin stability component
- **Balance Sheet Risk Assessment**:
  - Leverage ratios
  - Coverage ratios
  - Liquidity metrics
- **Earnings Quality Checks**:
  - DSO spike detection
  - Receivables growth > revenue growth
  - Margin compression
  - FCF < Net Income
- **Narrative Consistency**:
  - MD&A claims vs actual numbers
  - Management optimism vs reality
- **Clear Recommendation**:
  - Buy / Sell / Hold (with confidence level)
  - Bull thesis
  - Bear thesis
  - Red flags
  - What to monitor next

**Example Output:**
```json
{
  "investment_memo": {
    "recommendation": {
      "rating": "Buy",
      "confidence": "Medium-High",
      "composite_score": 67.5,
      "rationale": [
        "Quality Score: 72/100 (Good)",
        "Balance Sheet Risk: Low",
        "Earnings Quality: High"
      ]
    },
    "quality_score": {
      "score": 72,
      "components": {
        "profitability": 32,
        "cash_flow": 25,
        "margin_stability": 15
      },
      "rating": "Good"
    },
    "red_flags": [
      "High DSO (> 90 days) - potential collection issues",
      "Moderate leverage (D/E > 1.0)"
    ],
    "monitoring_plan": [
      "Monitor quarterly margin trends",
      "Watch FCF conversion sustainability",
      "Track DSO and working capital metrics"
    ]
  }
}
```

## 🖥️ Streamlit UI

Beautiful, interactive web interface for exploring analyses:

### Features
- **Document Selection** - Browse all processed filings
- **Analysis Dashboard** - Overview of all analyses
- **Interactive Tabs**:
  - Overview (quick stats and metrics)
  - Executive Summary (key findings + delta)
  - SWOT Analysis (hostile witness mode)
  - Financial Metrics (with charts)
  - Investment Decision (recommendation + red flags)
  - Citations (full provenance tracking)
- **Clickable Citations** - See exact source of each claim
- **Visualizations** - Charts and graphs for financial metrics
- **Export Functionality** - Download reports

### Launch UI

```bash
streamlit run streamlit_app.py
```

Then navigate to `http://localhost:8501`

## 🚀 Quick Start with V2

### 1. Process a Filing

```bash
# Download and process
python run_pipeline.py --ticker AAPL --form-type 10-K --limit 1
```

### 2. Run V2 Agents

```bash
# Run all V2 agents with default settings
python run_agents_v2.py --doc-id AAPL_10-K_0001628280-23-026301

# With prior year comparison (delta analysis)
python run_agents_v2.py \
  --doc-id AAPL_10-K_2023 \
  --prior-doc-id AAPL_10-K_2022

# Custom investor profile
python run_agents_v2.py \
  --doc-id AAPL_10-K_0001628280-23-026301 \
  --risk-tolerance aggressive \
  --horizon long_term
```

### 3. View in UI

```bash
streamlit run streamlit_app.py
```

## 📊 Output Structure

Each V2 analysis includes:

```
data/final/{doc_id}/
├── summary_analysis_v2.json      # Executive brief with citations
├── swot_analysis_v2.json         # Evidence-based SWOT + delta
├── metrics_analysis_v2.json      # Comprehensive KPIs with formulas
└── decision_analysis_v2.json     # Investment memo with red flags
```

## 🎯 Key Improvements

### 1. Provenance Tracking
Every piece of information includes:
- Page number
- Section/Item label
- Bounding box (optional)
- Extraction method
- Confidence score

### 2. Evidence-Based Analysis
- No generic claims
- Concrete data backing
- Quantitative validation
- Source citations

### 3. Year-over-Year Comparison
- Delta analysis across filings
- Risk factor changes
- Sentiment shifts
- Metric trends

### 4. Quality Assessment
- Investment quality scores
- Earnings quality checks
- Red flags detection
- Accounting smoke signals

### 5. Interactive UI
- Beautiful visualizations
- Clickable citations
- Export capabilities
- Multi-document comparison

## 📚 Documentation

- **Full Agent Specs**: See agent files for detailed docstrings
- **Provenance System**: `utils/provenance.py`
- **UI Guide**: Interactive tooltips in Streamlit app
- **API Reference**: Each agent has a `main()` function for CLI usage

## 🔧 Advanced Usage

### Compare Multiple Companies

```bash
# Process multiple filings
for ticker in AAPL MSFT GOOGL; do
  python run_pipeline.py --ticker $ticker --form-type 10-K --limit 1
  python run_agents_v2.py --doc-id ${ticker}_10-K_*
done

# View in UI for side-by-side comparison
streamlit run streamlit_app.py
```

### Historical Trend Analysis

```bash
# Download last 3 years
python run_pipeline.py --ticker AAPL --form-type 10-K --limit 3

# Analyze with delta
python run_agents_v2.py --doc-id AAPL_10-K_2023 --prior-doc-id AAPL_10-K_2022
python run_agents_v2.py --doc-id AAPL_10-K_2022 --prior-doc-id AAPL_10-K_2021
```

### Custom Analysis Workflow

```python
from agents import SummaryAgentV2, SWOTAgentV2, MetricsAgentV2, DecisionAgentV2

# Initialize agents
summary_agent = SummaryAgentV2()
swot_agent = SWOTAgentV2()
metrics_agent = MetricsAgentV2()
decision_agent = DecisionAgentV2()

# Run analyses
summary = summary_agent.analyze(doc_id, prior_doc_id)
swot = swot_agent.analyze(doc_id, prior_doc_id)
metrics = metrics_agent.analyze(doc_id)
decision = decision_agent.analyze(doc_id, risk_tolerance="aggressive")

# Access results
print(summary['executive_brief']['key_findings'])
print(decision['investment_memo']['recommendation']['rating'])
```

## 🐛 Troubleshooting

### UI Not Loading
```bash
# Check if streamlit is installed
pip install streamlit plotly

# Run with verbose logging
streamlit run streamlit_app.py --logger.level=debug
```

### Missing Citations
- Ensure V2 agents were run (not V1)
- Check that provenance tracking is enabled in config
- Verify XBRL and table data is available

### Performance Issues
- V2 agents are more comprehensive and may take longer
- Consider running stages in parallel
- Use `--no-validate` flag for faster processing

## 📈 Roadmap

- [ ] Multi-document comparison view in UI
- [ ] Historical trend charts
- [ ] PDF export of investment memos
- [ ] Real-time filing monitoring
- [ ] API endpoints for programmatic access

---

**V2 Status: Production Ready** ✅

Enhanced agents with full provenance tracking, evidence-based analysis, and beautiful UI!
