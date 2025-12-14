# 📊 Multi-Year Comparison Feature Guide

## Overview

This feature allows you to download, process, and compare multiple years of SEC 10-K filings side-by-side. Perfect for analyzing trends, tracking strategic changes, and identifying YoY patterns.

---

## 🚀 Quick Start (3 Steps)

### Step 1: Download Multiple Years

Download the last 3 years of 10-K filings for Apple:

```bash
python download_multi_year.py --ticker AAPL --years 3
```

**What this does:**
- Downloads 3 most recent 10-K filings
- Processes each through the full 6-stage pipeline
- Runs all 4 agents on each filing
- Takes ~10-15 minutes per filing

**Output:**
```
data/final/
├── AAPL_10-K_2023_xxx/  # FY2023 analysis
├── AAPL_10-K_2024_xxx/  # FY2024 analysis
└── AAPL_10-K_2025_xxx/  # FY2025 analysis
```

---

### Step 2: Run Comparison Analysis

Compare all years:

```bash
python run_comparison.py --ticker AAPL
```

**What this does:**
- Finds all processed AAPL 10-K filings
- Compares financial metrics YoY
- Analyzes SWOT evolution
- Tracks risk factor changes
- Generates insights

**Output:**
```
data/final/comparison_AAPL.json
```

---

### Step 3: View in Streamlit

```bash
streamlit run streamlit_app.py
```

Then:
1. Select any AAPL filing in the sidebar
2. Go to **"Comparison"** tab
3. See interactive charts and trends!

---

## 📋 What You'll See in the Comparison Tab

### 1. **Key Insights** 🔍
```
- Revenue grew 5.2% YoY to $416,161M
- Threat profile increased by 2 items
- Gross margins contracted by 1.2pp
```

### 2. **Financial Trends** 💰

**Revenue Chart:**
- Line graph showing revenue over 3 years
- Clear visualization of growth trajectory

**Profitability Margins:**
- Gross, Operating, and Net margins plotted together
- Spot margin expansion/contraction trends

**YoY Changes:**
- Expandable cards for each year transition
- Shows growth rates and margin changes
- Delta indicators (🔼 green, 🔽 red)

### 3. **SWOT Evolution** 🎯

**Grouped Bar Chart:**
- Tracks number of Strengths, Weaknesses, Opportunities, Threats
- Visual comparison across years
- Identify if company is improving or facing new challenges

### 4. **Risk Factor Evolution** 🚨

**Risk Timeline:**
- New risks added each year
- Existing risks that intensified
- Summary trend (expanding/stable/contracting)

---

## 🛠️ Advanced Usage

### Download Specific Years

```bash
# Download last 5 years
python download_multi_year.py --ticker AAPL --years 5

# Download 10-Q quarterly filings
python download_multi_year.py --ticker AAPL --years 4 --filing-type 10-Q
```

### Process Existing Files

If you already downloaded filings but want to reprocess:

```bash
# Skip download, just run pipeline + agents
python download_multi_year.py --ticker AAPL --skip-download --years 3

# Skip pipeline, just run agents
python download_multi_year.py --ticker AAPL --skip-download --skip-pipeline --years 3
```

### Compare Specific Documents

```bash
python run_comparison.py --doc-ids \
    AAPL_10-K_2023_0000320193-23-000106 \
    AAPL_10-K_2024_0000320193-24-000123 \
    AAPL_10-K_2025_0000320193-25-000079
```

---

## 📊 Example: Analyzing Apple's 3-Year Trend

### Full Workflow

```bash
# 1. Download 3 years of Apple 10-Ks
python download_multi_year.py --ticker AAPL --years 3

# Expected output:
# ✅ Downloaded 3 years of 10-K filings
# 🔄 Processing AAPL_10-K_2023...
#   ✅ Stage 1: Layout complete
#   ✅ Stage 2: Text complete
#   ...
# 🤖 Running agents...
# ✅ All agents complete for AAPL_10-K_2023
# [Repeats for 2024, 2025]

# 2. Run comparison
python run_comparison.py --ticker AAPL

# Expected output:
# 💰 Financial Trends:
#   2023 → 2024:
#     Revenue Growth: +8.2%
#     Net Income Growth: +12.4%
#     Gross Margin Change: +0.5pp
#   2024 → 2025:
#     Revenue Growth: +5.2%
#     Net Income Growth: +7.8%
#     Gross Margin Change: -0.2pp
# 
# 🔍 Key Insights:
#   • Revenue grew 5.2% YoY to $416,161M
#   • Investment recommendation changed from Hold to Buy

# 3. View in Streamlit
streamlit run streamlit_app.py
# Open http://localhost:8501
# Go to "Comparison" tab
```

---

## 🔍 Comparison Metrics Tracked

### Financial Metrics
- **Revenue** (growth %)
- **Net Income** (growth %)
- **Gross Margin** (pp change)
- **Operating Margin** (pp change)
- **Net Margin** (pp change)

### SWOT Metrics
- Number of Strengths/Weaknesses/Opportunities/Threats
- Added/removed items per year
- Category shifts

### Risk Metrics
- New risks (Item 1A additions)
- Heightened risks (existing risks with stronger language)
- Removed/diminished risks

### Strategic Insights
- Investment recommendation changes
- Quality score trends
- Red flag evolution

---

## 📁 File Structure

After running multi-year analysis:

```
Project_Green_Lattern/
├── data/
│   ├── raw/
│   │   └── sec-edgar-filings/
│   │       └── AAPL/
│   │           └── 10-K/
│   │               ├── 2023_0000320193-23-000106/
│   │               ├── 2024_0000320193-24-000123/
│   │               └── 2025_0000320193-25-000079/
│   │
│   ├── processed/
│   │   ├── AAPL_10-K_2023_0000320193-23-000106/
│   │   ├── AAPL_10-K_2024_0000320193-24-000123/
│   │   └── AAPL_10-K_2025_0000320193-25-000079/
│   │
│   └── final/
│       ├── AAPL_10-K_2023_0000320193-23-000106/
│       │   ├── summary_analysis_v2.json
│       │   ├── swot_analysis_v2.json
│       │   ├── metrics_analysis_v2.json
│       │   └── decision_analysis_v2.json
│       ├── AAPL_10-K_2024_0000320193-24-000123/
│       ├── AAPL_10-K_2025_0000320193-25-000079/
│       └── comparison_AAPL.json  # ← Comparison results
│
├── download_multi_year.py       # Download + process script
├── run_comparison.py            # Comparison analysis script
└── agents/
    └── comparison_agent.py      # Comparison agent logic
```

---

## 🎯 Use Cases

### 1. **Investment Research**
- Track revenue/profit growth trends
- Identify margin expansion/compression
- Spot strategic shifts (Buy → Sell rating changes)

### 2. **Competitor Analysis**
- Download 3 years for multiple companies
- Compare growth rates
- Benchmark profitability margins

### 3. **Risk Assessment**
- Monitor risk profile evolution
- Identify emerging threats early
- Track management's risk disclosure patterns

### 4. **Due Diligence**
- Comprehensive 3-year financial history
- SWOT trend analysis
- Citation-backed evidence for all claims

---

## ⚠️ Important Notes

### Processing Time
- **Per filing:** ~10-15 minutes
- **3 filings:** ~30-45 minutes total
- Includes: Download → Pipeline (6 stages) → Agents (4 agents)

### Storage Requirements
- **Per filing:** ~50-100 MB
- **3 years:** ~150-300 MB
- Includes: Raw PDFs, processed data, vector stores, analysis

### API Costs (OpenAI)
- **Per filing:** ~$0.50-1.00 (depending on filing size)
- **3 years:** ~$1.50-3.00 total
- Uses `gpt-4o-mini` for cost efficiency

### Rate Limits
- SEC Edgar: 10 requests/second (handled automatically)
- OpenAI: Tier-based (script includes retry logic)

---

## 🐛 Troubleshooting

### "No filings found"

**Problem:** `download_multi_year.py` couldn't find downloaded files

**Solution:**
```bash
# Check if files were downloaded
ls data/raw/sec-edgar-filings/AAPL/10-K/

# If empty, try downloading again
python download_multi_year.py --ticker AAPL --years 3
```

### "No comparison analysis found"

**Problem:** Comparison tab shows warning in Streamlit

**Solution:**
```bash
# Run comparison agent
python run_comparison.py --ticker AAPL

# Then refresh Streamlit
```

### Pipeline fails for one filing

**Problem:** One year fails but others succeed

**Solution:**
- Script continues processing other years
- Check logs for specific error
- Common issues: Missing data, HTML-only filing, XBRL not available

**Workaround:**
```bash
# Process individual filing
python run_pipeline.py --doc-id AAPL_10-K_2024_xxx

# Then re-run comparison
python run_comparison.py --ticker AAPL
```

### Slow download speed

**Problem:** Downloads taking too long

**Solution:**
```bash
# Download only (skip processing)
python download_multi_year.py --ticker AAPL --years 3 --skip-pipeline --skip-agents

# Then process in batches later
python download_multi_year.py --ticker AAPL --skip-download --years 3
```

---

## 🔮 Coming Soon

- [ ] **Quarterly comparison** (10-Q filings)
- [ ] **Multi-company comparison** (benchmark against competitors)
- [ ] **Custom metrics** (add your own comparison metrics)
- [ ] **Export to Excel** (comparison tables)
- [ ] **PDF report generation** (automated investment memo)

---

## 💡 Pro Tips

1. **Start with 2 years** to test the workflow before downloading 3-5 years
2. **Use `--skip-download`** if you need to reprocess with updated agents
3. **Check Streamlit first** - you might already have filings downloaded
4. **Use comparison for board presentations** - the charts are presentation-ready
5. **Combine with decision agent** - see how investment rating evolved

---

**Last Updated:** December 13, 2025  
**Status:** ✅ Fully Functional  
**Next Steps:** Download your first multi-year comparison!

```bash
python download_multi_year.py --ticker AAPL --years 3
python run_comparison.py --ticker AAPL
streamlit run streamlit_app.py
```
