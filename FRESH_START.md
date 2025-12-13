# 🚀 Fresh Start Guide - SEC Filings Analysis System

## ✅ **System Ready - Clean Setup Complete**

All data has been cleared. Your system is configured with:
- ✅ **Gemini API** (`gemini-2.5-flash`)
- ✅ **FAISS Vector Store** (sentence-transformers)
- ✅ **All Dependencies Installed**

---

## 📋 **Quick Start (3 Simple Steps)**

### Step 1: Download & Process a Filing (2-3 minutes)

```bash
# Download Apple's latest 10-K filing and process it
python run_pipeline.py --ticker AAPL --form-type 10-K --limit 1

# This will:
# - Download the filing from SEC Edgar
# - Extract text (Stage 2)
# - Create chunks (Stage 5)
# - Build FAISS vector index
```

### Step 2: Run AI Analysis (1-2 minutes)

```bash
# Run all V2 agents with Gemini AI
python run_agents_v2.py --doc-id <DOC_ID_FROM_STEP1>

# This will generate:
# - Executive Summary with key findings
# - SWOT Analysis (hostile witness mode)
# - Financial Metrics & KPIs
# - Investment Decision recommendation
```

### Step 3: View Results in UI

```bash
# Launch interactive dashboard
streamlit run streamlit_app.py

# Then open: http://localhost:8501
```

---

## 🔑 **Configuration**

Your `.env` file is already configured:
```
GEMINI_API_KEY=AIzaSyAKq-iNYVVmNymbamLVh1ngDaSVZAebkhU
GEMINI_MODEL=gemini-2.5-flash
SEC_USER_AGENT=MubinModi mubinmodi@gmail.com
```

---

## 📊 **Pipeline Stages**

| Stage | Name | Status | Description |
|-------|------|--------|-------------|
| 0 | Download | ✅ Working | Downloads SEC filings |
| 1 | Layout | ⚠️ Optional | Detectron2 layout detection (skip if issues) |
| 2 | Text | ✅ Working | Extracts text from HTML/PDF |
| 3 | Tables | ⚠️ Optional | Camelot table extraction (requires PDF) |
| 4 | XBRL | ⚠️ Optional | XBRL financial data (not always available) |
| 5 | Chunks | ✅ Working | Creates RAG-ready chunks |

**Recommended:** Run stages `0,2,5` for reliable operation.

---

## 🎯 **Example: Full Analysis**

```bash
# 1. Process Apple 10-K (most recent)
python run_pipeline.py --ticker AAPL --form-type 10-K --limit 1 --stages 0,2,5

# 2. Note the doc_id from output (e.g., AAPL_10-K_0000320193-25-000079)

# 3. Run analysis
python run_agents_v2.py --doc-id AAPL_10-K_0000320193-25-000079

# 4. View in UI
streamlit run streamlit_app.py
```

---

## 📁 **Data Structure**

After running, your data will be organized as:

```
data/
├── raw/                    # Downloaded filings
│   └── {doc_id}/
│       ├── filing.html     # Original HTML filing
│       └── manifest.json   # Metadata
│
├── processed/              # Extracted data
│   └── {doc_id}/
│       ├── tokens.jsonl    # Extracted tokens
│       └── text_blocks.jsonl
│
├── final/                  # Analysis results
│   └── {doc_id}/
│       ├── chunks.jsonl    # RAG chunks
│       ├── filing.md       # Markdown version
│       ├── summary_analysis_v2.json
│       ├── swot_analysis_v2.json
│       ├── metrics_analysis_v2.json
│       └── decision_analysis_v2.json
│
└── vector_stores/          # FAISS indices
    └── {doc_id}/
        ├── vector_index.faiss
        └── chunk_mapping.json
```

---

## 🔧 **Advanced Options**

### Download Multiple Filings
```bash
python run_pipeline.py --ticker MSFT --form-type 10-K --limit 3
```

### Compare Year-over-Year
```bash
python run_agents_v2.py --doc-id AAPL_10-K_2024 --prior-doc-id AAPL_10-K_2023
```

### Specific Agents Only
```bash
python run_agents_v2.py --doc-id AAPL_10-K_xxx --agents summary,swot
```

### Custom Risk Tolerance
```bash
python run_agents_v2.py --doc-id AAPL_10-K_xxx --risk-tolerance aggressive
```

---

## 🐛 **Troubleshooting**

### If Stage 1 (Layout) fails:
```bash
# Skip it - not critical
python run_pipeline.py --ticker AAPL --form-type 10-K --limit 1 --stages 0,2,5
```

### If Gemini rate limit hit:
```bash
# Wait 60 seconds and retry
sleep 60 && python run_agents_v2.py --doc-id <DOC_ID>
```

### If PDF not found:
System will automatically process HTML files instead.

---

## 📚 **Learn More**

- `README.md` - Full project documentation
- `README_V2.md` - V2 agent enhancements
- `RUN_ME_FIRST.md` - Detailed setup guide
- `GEMINI_INTEGRATION_STATUS.md` - Integration details

---

## ✨ **You're All Set!**

Start with Step 1 above to process your first filing! 🎯
