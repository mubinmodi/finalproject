# Project Green Lattern 🏢📊

**Open-Source SEC Filings Pipeline + Multi-Agent Investment Analysis**

A fully reproducible pipeline that ingests SEC filings (10-K/10-Q), extracts text, tables, layout, and XBRL data, and powers intelligent investment analysis agents.

## 🎯 Objective

Build a complete pipeline that:
- Downloads SEC filings with XBRL attachments
- Extracts text with word-level provenance
- Detects and extracts tables (financial statements, footnotes)
- Parses XBRL for numeric ground truth
- Creates RAG-ready document chunks
- Provides 4 specialized analysis agents

## 🏗️ Architecture

### Pipeline Stages

1. **Stage 0: Download + Organize** - Fetch SEC filings + XBRL
2. **Stage 1: Layout Detection** - Identify text/table/figure blocks
3. **Stage 2: Text Extraction** - PDF text + OCR fallback with bounding boxes
4. **Stage 3: Table Extraction** - Financial tables as CSV with quality metrics
5. **Stage 4: XBRL Extraction** - Canonical numeric facts
6. **Stage 5: Document Store** - RAG-ready chunks with full provenance

### Analysis Agents

- **Summary Agent** - Executive summary generation
- **SWOT Agent** - Strengths, Weaknesses, Opportunities, Threats
- **Metrics Agent** - Key financial metrics and trend analysis
- **Decision Agent** - Investment suggestions and rationale

## 🛠️ Technology Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| Filing Download | sec-edgar-downloader | SEC EDGAR API integration |
| Text Extraction | pdfplumber | Primary text + word boxes |
| OCR Fallback | Tesseract (pytesseract) | Handle scanned PDFs |
| Table Extraction | Camelot + pdfplumber | Financial statement tables |
| Layout Detection | LayoutParser | Block-level document structure |
| XBRL Parsing | Arelle | Canonical financial data |
| Pipeline Orchestration | DVC | Reproducible ML pipeline |
| Agents | LangChain + OpenAI | Multi-agent analysis |
| Vector Store | ChromaDB | RAG document retrieval |

## 📁 Project Structure

```
Project_Green_Lattern/
├── data/
│   ├── raw/              # Downloaded filings
│   │   └── {doc_id}/
│   │       ├── filing.pdf
│   │       ├── xbrl/
│   │       └── manifest.json
│   ├── processed/        # Intermediate outputs
│   │   └── {doc_id}/
│   │       ├── blocks.jsonl
│   │       ├── tokens.jsonl
│   │       ├── text_blocks.jsonl
│   │       ├── tables/
│   │       └── xbrl_facts.jsonl
│   └── final/            # RAG-ready outputs
│       └── {doc_id}/
│           ├── filing.md
│           └── chunks.jsonl
├── pipeline/
│   ├── stage0_download.py
│   ├── stage1_layout.py
│   ├── stage2_text.py
│   ├── stage3_tables.py
│   ├── stage4_xbrl.py
│   └── stage5_chunks.py
├── agents/
│   ├── base_agent.py
│   ├── summary_agent.py
│   ├── swot_agent.py
│   ├── metrics_agent.py
│   └── decision_agent.py
├── utils/
│   ├── config.py
│   ├── logging_utils.py
│   └── validation.py
├── run_pipeline.py
├── run_agents.py
├── requirements.txt
├── dvc.yaml
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <your-repo>
cd Project_Green_Lattern

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Tesseract (for OCR)
# macOS: brew install tesseract
# Ubuntu: sudo apt-get install tesseract-ocr
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
```

### 2. Configuration

Create a `.env` file:

```bash
# Optional: OpenAI API key for agents
OPENAI_API_KEY=your_key_here

# Tesseract path (if not in PATH)
TESSERACT_CMD=/usr/local/bin/tesseract
```

### 3. Run Pipeline

```bash
# Download and process a single filing
python run_pipeline.py --ticker AAPL --form-type 10-K --limit 1

# Process multiple filings
python run_pipeline.py --ticker MSFT --form-type 10-Q --limit 4

# Run specific stages
python run_pipeline.py --ticker AAPL --stages 0,1,2
```

### 4. Run Agents

```bash
# Analyze a processed filing
python run_agents.py --doc-id AAPL_10-K_2023

# Run specific agents
python run_agents.py --doc-id AAPL_10-K_2023 --agents summary,metrics
```

## 📊 Pipeline Details

### Stage 0: Download + Organize

- Uses `sec-edgar-downloader` to fetch filings from SEC EDGAR
- Downloads both HTML/PDF filings and XBRL attachments
- Creates consistent directory structure with manifest

### Stage 1: Layout Detection

- Runs LayoutParser with pre-trained Detectron2 models
- Detects blocks: text, title, table, figure, list
- Outputs bounding boxes for downstream routing

### Stage 2: Text Extraction

- Primary: `pdfplumber` for native PDF text + word-level boxes
- Fallback: Tesseract OCR when text quality is poor
- Preserves provenance (page, bbox, method used)

### Stage 3: Table Extraction

- Tries multiple methods per table:
  - Camelot lattice (for ruled tables)
  - Camelot stream (for borderless tables)
  - pdfplumber table finder
- Saves CSV + metadata (method, quality score)

### Stage 4: XBRL Extraction

- Parses XBRL using Arelle
- Extracts: concept, context, period, value, units
- Provides numeric ground truth for validation

### Stage 5: Canonical Document Store

- Creates sectioned Markdown documents
- Generates JSONL chunks with full provenance
- Ready for vector embedding and RAG retrieval

## 🤖 Agent System

### Summary Agent
- Generates executive summaries
- Highlights key business developments
- Extracts management discussion insights

### SWOT Agent
- Identifies strengths, weaknesses, opportunities, threats
- Structured analysis with evidence citations
- Risk factor analysis

### Metrics Agent (Primary)
- Extracts and validates financial KPIs
- Trend analysis (YoY, QoQ growth)
- Ratio calculations (margins, liquidity, leverage)

### Decision/Suggestion Agent
- Investment recommendations
- Evidence-based rationale
- Risk assessment

## 🔄 Reproducibility with DVC

The entire pipeline is orchestrated with DVC:

```bash
# Initialize DVC
dvc init

# Run full pipeline
dvc repro

# View pipeline DAG
dvc dag
```

## 📈 Output Examples

### JSONL Chunk Format

```json
{
  "doc_id": "AAPL_10-K_2023",
  "chunk_id": "chunk_001",
  "item": "Item 1",
  "section": "Business Overview",
  "page": 5,
  "bbox": [72, 100, 540, 300],
  "text": "Apple Inc. designs, manufactures...",
  "extractor": "pdfplumber",
  "source_path": "data/raw/AAPL_10-K_2023/filing.pdf"
}
```

### Table CSV + Metadata

```json
{
  "table_id": "table_003",
  "page": 45,
  "bbox": [50, 150, 560, 400],
  "caption": "Consolidated Statement of Operations",
  "method": "camelot_lattice",
  "quality_score": 0.95,
  "csv_path": "data/processed/AAPL_10-K_2023/tables/table_003.csv"
}
```

## 🧪 Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=pipeline --cov=agents
```

## 📝 License

MIT License - Feel free to use and modify

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 🙏 Acknowledgments

- SEC EDGAR for public financial data
- Open-source tools: LayoutParser, Camelot, Arelle, pdfplumber
- Detectron2 and PyTorch teams

---

**Built with ❤️ for transparent financial analysis**
