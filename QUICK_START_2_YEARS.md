# 🚀 Quick Start: 2-Year Comparison (Faster!)

## ⚡ Why 2 Years?

- **Faster:** ~20-30 minutes instead of 30-45 minutes
- **Still useful:** Shows YoY trends clearly
- **Lower cost:** ~$1-2 instead of $1.50-3
- **Good for testing:** Verify everything works before scaling up

---

## 🎯 One-Command Workflow

```bash
./quick_start_comparison.sh AAPL 2
```

**Or just:**

```bash
./quick_start_comparison.sh AAPL
```

(Defaults to 2 years now)

---

## 📋 What You'll Get (2 Years)

### Downloads:
- FY2024 (most recent)
- FY2025 (current)

### Comparison Shows:
- Revenue growth % (2024 → 2025)
- Margin changes (Gross/Operating/Net)
- SWOT evolution
- New/heightened risks

### Charts in Streamlit:
- ✅ Revenue trend line (2 points)
- ✅ Margin trends (2 points)
- ✅ SWOT bars (2 years)
- ✅ Risk timeline (2 years)

---

## ⏱️ Time Breakdown (2 Years)

| Step | Time per Filing | Total |
|------|----------------|-------|
| Download | 1-2 min | ~2-4 min |
| Pipeline (6 stages) | 3-5 min | ~6-10 min |
| Agents (4 agents) | 5-8 min | ~10-16 min |
| **Total** | **10-15 min** | **~20-30 min** |

---

## 💰 Cost (2 Years)

- **OpenAI API:** ~$1.00-2.00 total
- **Storage:** ~100-200 MB
- **Network:** Minimal (SEC filings ~50 MB each)

---

## 🧪 Quick Test First

Before running the full 2-year workflow, test download works:

```bash
# Test download only (1 minute)
python download_multi_year.py --ticker AAPL --years 1 --skip-pipeline --skip-agents
```

If you see:
```
✅ Downloaded 1 10-K filings for AAPL
```

Then proceed with 2 years! ✅

---

## 🚀 Run 2-Year Comparison

### Method 1: Super Easy
```bash
./quick_start_comparison.sh AAPL
```

### Method 2: Step-by-Step
```bash
# 1. Download and process 2 years (~20-30 min)
python download_multi_year.py --ticker AAPL --years 2

# 2. Run comparison
python run_comparison.py --ticker AAPL

# 3. View results
streamlit run streamlit_app.py
```

---

## 📊 Expected Output

### Console:
```bash
📥 Downloading last 2 years of 10-K filings for AAPL
✅ Downloaded 2 10-K filings for AAPL

📁 Found 2 filings to process:
  - AAPL_10-K_2024_0000320193-24-000123
  - AAPL_10-K_2025_0000320193-25-000079

🔄 Processing AAPL_10-K_2024...
  ✅ Stage 1-5 complete
🤖 Running agents...
  ✅ All agents complete

[Repeats for 2025]

💰 Financial Trends:
  2024 → 2025:
    Revenue Growth: +5.2%
    Net Income Growth: +7.8%

✅ Comparison complete
```

### Streamlit (Comparison Tab):
- Revenue trend line showing 2024 → 2025
- Margin evolution charts
- SWOT bars comparing 2024 vs 2025
- Risk factor changes

---

## 🎯 Want More Years Later?

Once 2-year comparison works, you can add more:

```bash
# Add one more year (FY2023)
python download_multi_year.py --ticker AAPL --years 3

# Re-run comparison with all 3 years
python run_comparison.py --ticker AAPL
```

The system will find all available years and compare them all!

---

## 💡 Pro Tips

### 1. Test with 1 Year First
```bash
python download_multi_year.py --ticker AAPL --years 1 --skip-pipeline --skip-agents
```
Confirms download works (~1 minute)

### 2. Process in Stages
```bash
# Download only
python download_multi_year.py --ticker AAPL --years 2 --skip-pipeline --skip-agents

# Then process later
python download_multi_year.py --ticker AAPL --skip-download --years 2
```

### 3. Parallel Later (Future)
When we add parallel processing, 2 years will take ~12-15 minutes total!

---

## ✅ Ready to Try?

**Super quick test (1 min):**
```bash
python download_multi_year.py --ticker AAPL --years 1 --skip-pipeline --skip-agents
```

**Full 2-year comparison (~20-30 min):**
```bash
./quick_start_comparison.sh AAPL
```

---

**Default:** ✅ **Now 2 Years**  
**Time:** ⚡ **20-30 minutes**  
**Cost:** 💰 **~$1-2**  
**Result:** 📊 **Complete YoY comparison**
