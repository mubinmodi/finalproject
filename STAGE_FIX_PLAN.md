# 🔧 Pipeline Stage Fix Plan

## Current Issues & Solutions:

### Stage 0 (Download) ✅ Working
**Issue:** Only downloads HTML, no PDF or XBRL
**Solution:** 
- SEC Edgar doesn't always provide PDF versions
- HTML is sufficient for text extraction
- XBRL must be requested separately (10-K/A filings)

### Stage 1 (Layout Detection) ❌ Needs Fix
**Issue:** Detectron2 model caching to temp directory
**Solution Options:**
1. **Fix cache directory** - Set MODEL_CACHE env variable
2. **Skip if not critical** - Text extraction works without it
3. **Use simpler method** - Block detection from HTML structure

**Recommended:** Fix cache directory

### Stage 2 (Text Extraction) ✅ Working
**Status:** Successfully extracts from HTML
**Output:** 32,213 tokens extracted

### Stage 3 (Table Extraction) ❌ Needs Fix
**Issue:** Camelot requires PDF, we only have HTML
**Solution Options:**
1. **Convert HTML to PDF** - Use weasyprint (already attempted)
2. **Extract tables from HTML** - Use pandas read_html()
3. **Skip if no PDF** - Graceful degradation

**Recommended:** Extract tables directly from HTML

### Stage 4 (XBRL Extraction) ❌ Needs Fix
**Issue:** No XBRL files downloaded
**Solution:**
- Download XBRL separately (different filing type)
- Or skip for now - text has most info

**Recommended:** Make XBRL optional, add separate download option

### Stage 5 (Chunking) ✅ Working
**Status:** Successfully created 280 chunks
**Output:** RAG-ready chunks with FAISS index

## Implementation Priority:

1. **HIGH:** Fix Stage 3 (Tables from HTML)
2. **MEDIUM:** Fix Stage 1 (Layout cache)
3. **LOW:** Add XBRL download option

## Quick Wins:

### 1. HTML Table Extraction (Stage 3)
Replace Camelot with pandas for HTML:
```python
import pandas as pd
from bs4 import BeautifulSoup

def extract_tables_from_html(html_path):
    # Read HTML tables directly
    tables = pd.read_html(html_path)
    return tables
```

### 2. Fix Layout Model Cache (Stage 1)
Set environment variable:
```python
import os
os.environ['TORCH_HOME'] = '/path/to/writable/cache'
os.environ['DETECTRON2_DATASETS'] = '/path/to/writable/cache'
```

### 3. Make XBRL Optional
Add flag to skip XBRL if not available:
```python
if not xbrl_path.exists():
    logger.warning("XBRL not found, skipping")
    return empty_result
```
