# 🚀 Run Me First - Quick Setup Guide

Welcome to Project Green Lattern! Follow these steps to get started.

## ⚡ Quick Start (5 Minutes)

### Step 1: Install Dependencies

```bash
# Navigate to project directory
cd /Users/mubinmodi/Project_Green_Lattern

# Install all required packages
pip install -r requirements.txt
```

**Note:** This will install:
- SEC filing downloaders
- PDF processing tools (pdfplumber, Camelot)
- LLM libraries (OpenAI, LangChain)
- UI framework (Streamlit, Plotly)

### Step 2: Configure Environment

```bash
# Copy example config
cp env.example .env

# Edit .env file with your settings
nano .env  # or use any text editor
```

**Required settings in `.env`:**
```bash
# OpenAI API Key (required for agents)
OPENAI_API_KEY=sk-your-key-here

# SEC requires your email
SEC_USER_AGENT=YourName your.email@example.com
```

### Step 3: Install System Dependencies

**macOS:**
```bash
# Install Tesseract OCR
brew install tesseract

# Install Ghostscript (for Camelot tables)
brew install ghostscript
```

**Ubuntu/Linux:**
```bash
# Install Tesseract OCR
sudo apt-get update
sudo apt-get install tesseract-ocr

# Install Ghostscript
sudo apt-get install ghostscript python3-tk
```

**Windows:**
- Download Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
- Install and add to PATH

### Step 4: Test with a Simple Example

```bash
# Download and analyze Apple's latest 10-K
python run_pipeline.py --ticker AAPL --form-type 10-K --limit 1

# This will take 5-10 minutes and download/process the filing
```

**What this does:**
1. Downloads AAPL 10-K from SEC EDGAR
2. Extracts layout, text, tables, XBRL
3. Creates RAG-ready chunks
4. Saves to `data/` directory

### Step 5: Run Analysis Agents

```bash
# Get the doc_id from previous output (looks like: AAPL_10-K_0001628280-23-026301)
python run_agents_v2.py --doc-id AAPL_10-K_0001628280-23-026301

# Or use a wildcard
python run_agents_v2.py --doc-id AAPL_10-K_*
```

**What this does:**
1. Runs Summary Agent (key findings + delta)
2. Runs SWOT Agent (hostile witness mode)
3. Runs Metrics Agent (comprehensive KPIs)
4. Runs Decision Agent (investment memo)

### Step 6: Launch Interactive UI

```bash
# Start Streamlit app
streamlit run streamlit_app.py

# Open browser to: http://localhost:8501
```

**What you'll see:**
- Interactive dashboard
- Analysis results with visualizations
- Clickable citations
- Financial metric charts
- Investment recommendation

---

## 🎯 Troubleshooting

### Issue: "No module named 'X'"

**Solution:**
```bash
pip install -r requirements.txt --upgrade
```

### Issue: "OpenAI API error"

**Solution:**
```bash
# Check your .env file
cat .env

# Make sure OPENAI_API_KEY is set correctly
# Get a key from: https://platform.openai.com/api-keys
```

### Issue: "Tesseract not found"

**Solution:**
```bash
# macOS
brew install tesseract

# Find tesseract path
which tesseract

# Add to .env
echo "TESSERACT_CMD=/usr/local/bin/tesseract" >> .env
```

### Issue: "Camelot table extraction failed"

**Solution:**
```bash
# Install ghostscript
brew install ghostscript  # macOS
sudo apt-get install ghostscript  # Linux
```

### Issue: "LayoutParser/Detectron2 errors"

**Solution:**
```bash
# Skip layout detection (optional stage)
python run_pipeline.py --ticker AAPL --stages 0,2,3,4,5

# This skips stage 1 (layout) but runs everything else
```

### Issue: "Pipeline takes too long"

**Solution:**
```bash
# Skip validation for faster processing
python run_pipeline.py --ticker AAPL --no-validate
```

---

## 📝 Common Workflows

### 1. Quick Analysis (10 minutes)

```bash
# Download, process, and analyze
python run_pipeline.py --ticker MSFT --form-type 10-K --limit 1
python run_agents_v2.py --doc-id MSFT_10-K_*
streamlit run streamlit_app.py
```

### 2. With Year-over-Year Comparison

```bash
# Download last 2 years
python run_pipeline.py --ticker AAPL --form-type 10-K --limit 2

# Analyze with delta
python run_agents_v2.py \
  --doc-id AAPL_10-K_2023 \
  --prior-doc-id AAPL_10-K_2022

# View in UI
streamlit run streamlit_app.py
```

### 3. Compare Multiple Companies

```bash
# Download filings for multiple companies
for ticker in AAPL MSFT GOOGL; do
  python run_pipeline.py --ticker $ticker --form-type 10-K --limit 1
  python run_agents_v2.py --doc-id ${ticker}_10-K_*
done

# View all in UI
streamlit run streamlit_app.py
```

### 4. Quarterly Analysis (10-Q)

```bash
# Download last 4 quarters
python run_pipeline.py --ticker AAPL --form-type 10-Q --limit 4

# Analyze each
for doc in data/raw/AAPL_10-Q_*/; do
  doc_id=$(basename $doc)
  python run_agents_v2.py --doc-id $doc_id
done
```

---

## 🔍 Understanding the Output

After running the pipeline and agents, you'll have:

```
data/
├── raw/                          # Downloaded filings
│   └── AAPL_10-K_*/
│       ├── filing.pdf           # Original filing
│       ├── xbrl/                # XBRL data
│       └── manifest.json        # Metadata
│
├── processed/                    # Intermediate outputs
│   └── AAPL_10-K_*/
│       ├── blocks.jsonl         # Layout blocks
│       ├── tokens.jsonl         # Text with bounding boxes
│       ├── tables/              # Extracted tables (CSV)
│       └── xbrl_facts.jsonl     # Financial facts
│
└── final/                        # Analysis results
    └── AAPL_10-K_*/
        ├── filing.md            # Human-readable summary
        ├── chunks.jsonl         # RAG-ready chunks
        ├── summary_analysis_v2.json    # Executive brief
        ├── swot_analysis_v2.json       # SWOT with evidence
        ├── metrics_analysis_v2.json    # Financial KPIs
        └── decision_analysis_v2.json   # Investment memo
```

---

## 💡 Pro Tips

### 1. Use wildcards for doc IDs
```bash
# Instead of typing full doc_id
python run_agents_v2.py --doc-id AAPL_10-K_*
```

### 2. Run specific pipeline stages
```bash
# Only extract tables (stage 3)
python run_pipeline.py --doc-id AAPL_10-K_* --stages 3
```

### 3. Run specific agents
```bash
# Only run metrics and decision agents
python run_agents_v2.py --doc-id AAPL_10-K_* --agents metrics,decision
```

### 4. Custom investor profile
```bash
# For aggressive investors
python run_agents_v2.py \
  --doc-id AAPL_10-K_* \
  --risk-tolerance aggressive \
  --horizon long_term
```

### 5. View results without UI
```bash
# View markdown summary
cat data/final/AAPL_10-K_*/filing.md

# View JSON analysis
cat data/final/AAPL_10-K_*/decision_analysis_v2.json | jq
```

---

## 🎓 Learning Resources

- **Full Documentation:** `README.md`
- **V2 Features:** `README_V2.md`
- **Technical Details:** `PROJECT_SUMMARY.md`
- **Quick Reference:** `QUICKSTART.md`

---

## ⚙️ System Requirements

**Minimum:**
- Python 3.8+
- 8GB RAM
- 5GB disk space

**Recommended:**
- Python 3.10+
- 16GB RAM
- 20GB disk space (for multiple filings)
- GPU (optional, for faster layout detection)

---

## 🆘 Getting Help

**Common Issues:**
1. Check `.env` file is configured
2. Verify Tesseract is installed: `tesseract --version`
3. Check OpenAI API key is valid
4. Ensure sufficient disk space

**Still stuck?**
- Check error logs in `logs/` directory
- Review inline documentation in Python files
- Read troubleshooting section above

---

## ✅ Success Checklist

- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `.env` file configured with API key
- [ ] Tesseract OCR installed
- [ ] Pipeline runs successfully
- [ ] Agents complete analysis
- [ ] UI launches in browser

**Once all checked, you're ready to analyze SEC filings!** 🚀

---

## 🎉 Quick Win

Want to see results immediately? Try this:

```bash
# One command to test everything (macOS/Linux)
pip install -r requirements.txt && \
cp env.example .env && \
echo "Replace OPENAI_API_KEY in .env first!" && \
python run_pipeline.py --ticker AAPL --form-type 10-K --limit 1 && \
python run_agents_v2.py --doc-id AAPL_10-K_* && \
streamlit run streamlit_app.py
```

**Windows PowerShell:**
```powershell
pip install -r requirements.txt
Copy-Item env.example .env
# Edit .env to add your OPENAI_API_KEY
python run_pipeline.py --ticker AAPL --form-type 10-K --limit 1
python run_agents_v2.py --doc-id (Get-ChildItem data/raw/AAPL_10-K_* | Select-Object -First 1).Name
streamlit run streamlit_app.py
```

---

**Happy Analyzing! 📊🚀**
