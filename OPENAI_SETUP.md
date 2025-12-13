# 🔄 Switched to OpenAI ChatGPT

## ✅ **What Changed:**

Your system has been migrated from **Gemini** to **OpenAI ChatGPT**.

### **Why?**
- **Gemini Free Tier:** 20 requests/day (hit quota immediately)
- **OpenAI Free Tier:** $5 credit = ~10,000 requests (way more generous!)

---

## 🔑 **Step 1: Get Your OpenAI API Key**

### **Option A: Use Existing Key (if you have one)**
If you already have an OpenAI API key, skip to Step 2.

### **Option B: Create New Key (Free $5 Credit)**

1. Go to: https://platform.openai.com/signup
2. Sign up with email/Google/Microsoft
3. Verify your account
4. Go to: https://platform.openai.com/api-keys
5. Click **"Create new secret key"**
6. Copy the key (starts with `sk-...`)

**Note:** New accounts get $5 free credit (expires after 3 months)

---

## 🔧 **Step 2: Update Your .env File**

Open your `.env` file and update it:

```bash
# Open the .env file
nano .env

# Or use any text editor:
open -a TextEdit .env
```

**Replace the Gemini section with OpenAI:**

```bash
# OpenAI API Configuration
OPENAI_API_KEY=sk-your-actual-api-key-here
OPENAI_MODEL=gpt-4o-mini

# SEC Edgar User Agent (Required)
SEC_USER_AGENT=MubinModi mubinmodi@gmail.com

# Optional: Tesseract Path (for OCR)
TESSERACT_CMD=/opt/homebrew/bin/tesseract
```

**Important:** Replace `sk-your-actual-api-key-here` with your real API key!

---

## 💰 **Cost Estimates:**

### **Model: gpt-4o-mini (Recommended)**
- **Input:** $0.15 per 1M tokens
- **Output:** $0.60 per 1M tokens
- **Per Analysis:** ~$0.01-0.02 (1-2 cents)
- **With $5 credit:** ~250-500 full analyses

### **Alternative: gpt-3.5-turbo (Cheaper)**
- **Input:** $0.50 per 1M tokens  
- **Output:** $1.50 per 1M tokens
- **Per Analysis:** ~$0.005 (half a cent)
- **With $5 credit:** ~1,000 full analyses

To use gpt-3.5-turbo, change in .env:
```bash
OPENAI_MODEL=gpt-3.5-turbo
```

---

## ✅ **Step 3: Test the Setup**

```bash
# Test with a simple query
python -c "
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[{'role': 'user', 'content': 'Say hello!'}],
    max_tokens=10
)
print('✅ OpenAI API working!')
print(f'Response: {response.choices[0].message.content}')
"
```

If you see "✅ OpenAI API working!" - you're ready to go!

---

## 🚀 **Step 4: Run Your Analysis**

Now run the agents with OpenAI:

```bash
# Your FAISS index is already built, so this will be fast!
python run_agents_v2.py --doc-id AAPL_10-K_0000320193-25-000079
```

---

## 📊 **What Changed Technically:**

### **Before (Gemini):**
```python
import google.generativeai as genai
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")
response = model.generate_content(prompt)
```

### **After (OpenAI):**
```python
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)
```

---

## 🔍 **Monitor Your Usage:**

Track your API usage at: https://platform.openai.com/usage

---

## ❓ **Troubleshooting:**

### **Error: "Incorrect API key provided"**
- Double-check your API key in `.env`
- Make sure it starts with `sk-`
- No extra spaces or quotes

### **Error: "You exceeded your current quota"**
- Check your usage at: https://platform.openai.com/usage
- You may need to add payment method (still uses free $5 credit first)

### **Error: "Rate limit reached"**
- OpenAI limits: 3 requests/min (free tier)
- The code has automatic retry with exponential backoff
- Just wait a few seconds and it will retry

---

## 🎯 **Benefits of OpenAI:**

1. **Higher Free Tier:** $5 credit vs 20 requests/day
2. **Better Models:** GPT-4o-mini is excellent quality
3. **Proven Reliability:** More stable API
4. **Better Documentation:** Extensive examples
5. **Cost-Effective:** Very cheap per request

---

## 🔄 **Next Steps:**

1. **Get API key** from OpenAI
2. **Update `.env`** with `OPENAI_API_KEY`
3. **Run test** to verify it works
4. **Run agents** to complete your analysis

---

**Ready to proceed? Once you've updated your `.env` file, just run:**

```bash
python run_agents_v2.py --doc-id AAPL_10-K_0000320193-25-000079
```

🎉 **Your FAISS vector index is already built, so the analysis will start immediately!**
