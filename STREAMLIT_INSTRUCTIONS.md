# 🚀 Streamlit UI - Clean Startup Instructions

## ✅ All Issues Are Fixed!

All agents are now working with **real Apple 10-K data**:
- ✅ Summary Agent: 4 key findings
- ✅ SWOT Agent: 16 items (4+4+4+4)
- ✅ Metrics Agent: Real financials (Revenue: $416B, Gross Margin: 46.91%)
- ✅ Decision Agent: Buy recommendation with 55/100 score

---

## 🔧 How to Restart Streamlit Cleanly

### Option 1: Use the Clean Startup Script (Recommended)

```bash
./run_streamlit_clean.sh
```

This script:
- Clears Streamlit's cache
- Suppresses harmless warnings
- Starts Streamlit with fresh data

### Option 2: Manual Restart

1. **Stop the current Streamlit** (press Ctrl+C twice in the terminal)

2. **Clear cache and restart:**
   ```bash
   export TOKENIZERS_PARALLELISM=false
   streamlit run streamlit_app.py --server.headless=true
   ```

3. **Hard refresh your browser:**
   - **Mac:** ⌘ + Shift + R
   - **Windows:** Ctrl + Shift + R

---

## ⚠️  Understanding the Warnings (All Harmless)

### 1. Pydantic Warning
```
UserWarning: Field name "validate" in "XBRLConfig" shadows an attribute in parent "BaseModel"
```
**What it is:** Pydantic field name conflict  
**Impact:** None - the code works correctly  
**Can ignore:** Yes

### 2. Torch Warning
```
RuntimeError: Tried to instantiate class '__path__._path'
```
**What it is:** Torch internal path inspection issue in Streamlit's file watcher  
**Impact:** None - only happens during Streamlit startup, doesn't affect functionality  
**Can ignore:** Yes

### 3. Tokenizers Warning
```
huggingface/tokenizers: The current process just got forked, after parallelism has already been used
```
**What it is:** HuggingFace tokenizers fork warning  
**Impact:** None - suppressed by `TOKENIZERS_PARALLELISM=false`  
**Can ignore:** Yes

---

## 📊 What You Should See in the UI

### Overview Tab
- 4 agent cards with status indicators
- Quick metrics summary

### Summary Tab
- **4 Key Findings** from Apple's 10-K
- Citations with page numbers
- Financial highlights section

### SWOT Tab
- **4 Strengths** (e.g., demand stimulation, IP portfolio)
- **4 Weaknesses** (e.g., asset decrease, IT dependencies)
- **4 Opportunities** (e.g., tax savings, market expansion)
- **4 Threats** (e.g., privacy compliance, competition)

### Metrics Tab
- **Income Statement:** Revenue $416,161M, Net Income $112,010M
- **Profitability Ratios:** 
  - Gross Margin: 46.91%
  - Operating Margin: 31.97%
  - Net Margin: 26.92%
- **Liquidity:** Debt-to-Assets: 0.79x

### Decision Tab
- **Recommendation:** Buy
- **Confidence:** Medium-High
- **Composite Score:** 55/100
- **Quality Score:** 55/100 (Fair)
- **2 Red Flags** identified
- Bull thesis (5 points) and Bear thesis (5 points)

---

## 🐛 If Data Still Appears Empty

1. **Check terminal for agent execution logs** - should show:
   ```
   ✅ Summary Agent complete
   ✅ SWOT Agent complete
   ✅ Metrics Agent complete
   ✅ Decision Agent complete
   ```

2. **Verify analysis files exist:**
   ```bash
   ls -lh data/final/AAPL_10-K_0000320193-25-000079/*_v2.json
   ```
   Should show 4 files, all from Dec 13 17:36-17:37

3. **Force regenerate analysis:**
   ```bash
   rm -f data/final/AAPL_10-K_0000320193-25-000079/*_v2.json
   python run_agents_v2.py --doc-id AAPL_10-K_0000320193-25-000079
   ```

4. **Then restart Streamlit** using Option 1 above

---

## 🎯 Next Steps

Once Streamlit is displaying data correctly:
1. ✅ Explore all 4 tabs
2. ✅ Test citations and provenance
3. ✅ Try different companies (download + run pipeline + run agents)
4. ✅ Customize agents for your specific needs

---

## 📚 Files Generated

All analysis files are in:
```
data/final/AAPL_10-K_0000320193-25-000079/
  ├── summary_analysis_v2.json   (7.5 KB)
  ├── swot_analysis_v2.json      (15 KB)
  ├── metrics_analysis_v2.json   (5.1 KB)
  └── decision_analysis_v2.json  (3.0 KB)
```

All contain real Apple financial data extracted from the 10-K filing!

---

## 💡 Pro Tips

1. **Browser devtools:** Open developer console (F12) to see any JS errors
2. **Streamlit logs:** Watch the terminal for real-time agent execution
3. **Force refresh:** Always hard refresh (Ctrl+Shift+R) after regenerating analysis
4. **Cache issues:** Use `run_streamlit_clean.sh` to avoid stale cache

---

**Last Updated:** Dec 13, 2025  
**Status:** ✅ All agents working with real data
