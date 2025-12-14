#!/bin/bash

# Quick Start: Multi-Year Comparison
# Downloads 3 years of filings, processes them, and runs comparison

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  📊 Multi-Year SEC Filing Comparison - Quick Start${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

# Get ticker from user
if [ -z "$1" ]; then
    read -p "Enter stock ticker (e.g., AAPL): " TICKER
    TICKER=$(echo "$TICKER" | tr '[:lower:]' '[:upper:]')
else
    TICKER=$(echo "$1" | tr '[:lower:]' '[:upper:]')
fi

# Get number of years
if [ -z "$2" ]; then
    read -p "How many years to compare? (default: 2): " YEARS
    YEARS=${YEARS:-2}
else
    YEARS=$2
fi

echo
echo -e "${GREEN}✓${NC} Ticker: ${TICKER}"
echo -e "${GREEN}✓${NC} Years: ${YEARS}"
echo

# Confirm
read -p "Proceed with download and analysis? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

echo
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}  Step 1/3: Downloading & Processing ${YEARS} Years of Filings${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo
echo "This will take approximately $(($YEARS * 10))-$(($YEARS * 15)) minutes..."
echo

python download_multi_year.py --ticker "$TICKER" --years "$YEARS"

if [ $? -ne 0 ]; then
    echo
    echo -e "${YELLOW}⚠️  Download/processing encountered some errors but may have partial results${NC}"
    echo "Continuing to comparison step..."
fi

echo
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}  Step 2/3: Running Comparison Analysis${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

python run_comparison.py --ticker "$TICKER"

if [ $? -ne 0 ]; then
    echo
    echo -e "${YELLOW}⚠️  Comparison analysis encountered errors${NC}"
    echo "Check if all filings were processed successfully."
    exit 1
fi

echo
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  ✅ Multi-Year Comparison Complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo
echo "📊 Results saved to: data/final/comparison_${TICKER}.json"
echo
echo "🚀 Next step: View in Streamlit"
echo "   Run: streamlit run streamlit_app.py"
echo "   Then go to the 'Comparison' tab"
echo
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

# Ask if user wants to start Streamlit
read -p "Start Streamlit now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo
    echo "Starting Streamlit..."
    echo "Open http://localhost:8501 in your browser"
    echo
    streamlit run streamlit_app.py
fi
