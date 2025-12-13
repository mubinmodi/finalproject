# Quick Start Guide 🚀

Get up and running with Project Green Lattern in minutes!

## Prerequisites

- Python 3.8+
- Tesseract OCR (for text extraction)
- OpenAI API key (for agents)

## Installation

### 1. Clone and Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Tesseract OCR
# macOS:
brew install tesseract

# Ubuntu/Debian:
sudo apt-get install tesseract-ocr

# Windows:
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

### 2. Configure

Create a `.env` file:

```bash
cp env.example .env
```

Edit `.env` and add:
- Your OpenAI API key
- Your email for SEC EDGAR (required)
- Tesseract path (if needed)

```env
OPENAI_API_KEY=sk-...
SEC_USER_AGENT=YourName your.email@example.com
```

## Quick Example

### Step 1: Download and Process a Filing

```bash
python run_pipeline.py --ticker AAPL --form-type 10-K --limit 1
```

This will:
- Download Apple's latest 10-K filing
- Extract layout, text, tables, and XBRL data
- Create RAG-ready chunks
- Save everything to `data/` directory

**Expected time:** 5-10 minutes (depending on document size)

### Step 2: Run Analysis Agents

```bash
# Get the doc_id from previous output, then:
python run_agents.py --doc-id AAPL_10-K_0001628280-23-026301
```

This will:
- Generate executive summary
- Perform SWOT analysis
- Extract financial metrics
- Provide investment recommendation

**Expected time:** 2-5 minutes (with API calls)

### Step 3: View Results

```bash
# View markdown summary
cat data/final/AAPL_10-K_*/filing.md

# View JSON analysis
cat data/final/AAPL_10-K_*/summary_analysis.json
cat data/final/AAPL_10-K_*/metrics_analysis.json
cat data/final/AAPL_10-K_*/decision_analysis.json
```

## Common Use Cases

### 1. Compare Multiple Companies

```bash
# Download filings for multiple companies
python run_pipeline.py --ticker AAPL --form-type 10-K --limit 1
python run_pipeline.py --ticker MSFT --form-type 10-K --limit 1
python run_pipeline.py --ticker GOOGL --form-type 10-K --limit 1

# Analyze each
python run_agents.py --doc-id AAPL_10-K_*
python run_agents.py --doc-id MSFT_10-K_*
python run_agents.py --doc-id GOOGL_10-K_*
```

### 2. Historical Analysis (Quarterly Filings)

```bash
# Download last 4 quarters
python run_pipeline.py --ticker AAPL --form-type 10-Q --limit 4

# Analyze each quarter
python run_agents.py --doc-id AAPL_10-Q_*
```

### 3. Custom Investor Profile

```bash
# Conservative investor, long-term horizon
python run_agents.py \
  --doc-id AAPL_10-K_* \
  --risk-tolerance conservative \
  --horizon long_term

# Aggressive investor, short-term
python run_agents.py \
  --doc-id AAPL_10-K_* \
  --risk-tolerance aggressive \
  --horizon short_term
```

### 4. Run Specific Pipeline Stages

```bash
# Only extract tables (useful for debugging)
python run_pipeline.py --doc-id AAPL_10-K_* --stages 3

# Re-run text extraction and chunking
python run_pipeline.py --doc-id AAPL_10-K_* --stages 2,5
```

### 5. Run Specific Agents

```bash
# Only financial metrics
python run_agents.py --doc-id AAPL_10-K_* --agents metrics

# Summary and SWOT only
python run_agents.py --doc-id AAPL_10-K_* --agents summary,swot
```

## Output Structure

```
data/
├── raw/                      # Downloaded filings
│   └── AAPL_10-K_*/
│       ├── filing.pdf
│       ├── filing.html
│       ├── xbrl/
│       └── manifest.json
├── processed/                # Intermediate outputs
│   └── AAPL_10-K_*/
│       ├── blocks.jsonl      # Layout blocks
│       ├── tokens.jsonl      # Text tokens
│       ├── text_blocks.jsonl # Text blocks
│       ├── tables/           # Extracted tables (CSV)
│       ├── tables_index.jsonl
│       └── xbrl_facts.jsonl  # XBRL data
└── final/                    # Final outputs
    └── AAPL_10-K_*/
        ├── filing.md         # Markdown summary
        ├── chunks.jsonl      # RAG-ready chunks
        ├── summary_analysis.json
        ├── swot_analysis.json
        ├── metrics_analysis.json
        └── decision_analysis.json
```

## Troubleshooting

### Issue: "Tesseract not found"

**Solution:**
```bash
# Install Tesseract
brew install tesseract  # macOS

# Or set path in .env
TESSERACT_CMD=/usr/local/bin/tesseract
```

### Issue: "OpenAI API error"

**Solution:**
- Check your API key in `.env`
- Verify you have API credits
- Check rate limits

### Issue: "Layout detection fails"

**Solution:**
```bash
# Install detectron2 dependencies
pip install torch torchvision
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### Issue: "Camelot table extraction fails"

**Solution:**
```bash
# Install system dependencies
sudo apt-get install ghostscript python3-tk  # Linux
brew install ghostscript  # macOS
```

### Issue: "Document not found"

**Solution:**
- Make sure to run pipeline first
- Use correct doc_id (check `data/raw/` directory)
- Doc IDs are in format: `{TICKER}_{FORM}_{ACCESSION}`

## Performance Tips

1. **Skip validation for faster processing:**
   ```bash
   python run_pipeline.py --ticker AAPL --no-validate
   ```

2. **Process multiple documents in parallel:**
   ```bash
   # Use multiple terminals or background jobs
   python run_pipeline.py --ticker AAPL &
   python run_pipeline.py --ticker MSFT &
   ```

3. **Limit layout detection (if having issues):**
   ```bash
   # Skip layout detection, run other stages
   python run_pipeline.py --doc-id AAPL_10-K_* --stages 0,2,3,4,5
   ```

4. **Use local models (if no OpenAI key):**
   - The pipeline stages work without LLM
   - Agents require LLM but show fallback messages

## Next Steps

1. **Explore the data:**
   - Open `filing.md` in a markdown viewer
   - Load `chunks.jsonl` into a vector database
   - Analyze `metrics_analysis.json` in Excel/Python

2. **Customize agents:**
   - Modify agent prompts in `agents/`
   - Add new analysis agents
   - Integrate with your own models

3. **Build dashboards:**
   - Use Streamlit/Gradio for web UI
   - Visualize metrics over time
   - Compare multiple companies

4. **Automate:**
   - Set up cron jobs for daily downloads
   - Monitor new filings automatically
   - Send alerts on significant changes

## Getting Help

- 📖 Read the full [README.md](README.md)
- 🐛 Report issues on GitHub
- 💬 Join discussions
- 📧 Contact maintainers

---

**Happy analyzing! 📊🚀**
