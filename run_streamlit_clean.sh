#!/bin/bash

# Suppress harmless warnings
export TOKENIZERS_PARALLELISM=false

# Clear Streamlit cache
echo "🧹 Clearing Streamlit cache..."
rm -rf ~/.streamlit/cache 2>/dev/null

# Start Streamlit with fresh cache
echo "🚀 Starting Streamlit with clean cache..."
streamlit run streamlit_app.py
