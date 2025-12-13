#!/bin/bash
# Complete Installation Script for Project Green Lattern
# This installs ALL dependencies for ALL pipeline stages

set -e  # Exit on error

echo "========================================"
echo "Project Green Lattern - Complete Setup"
echo "========================================"

# Check if in conda environment
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠️  Warning: Not in a conda environment"
    echo "Recommended: conda activate your_env"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Step 1: Installing system dependencies (wkhtmltopdf)"
echo "========================================"
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "Installing wkhtmltopdf via Homebrew..."
    if ! command -v wkhtmltopdf &> /dev/null; then
        brew install wkhtmltopdf
    else
        echo "✓ wkhtmltopdf already installed"
    fi
    
    # Install Tesseract if not present
    if ! command -v tesseract &> /dev/null; then
        echo "Installing Tesseract OCR..."
        brew install tesseract
    else
        echo "✓ Tesseract already installed"
    fi
    
    # Install Ghostscript if not present
    if ! command -v gs &> /dev/null; then
        echo "Installing Ghostscript..."
        brew install ghostscript
    else
        echo "✓ Ghostscript already installed"
    fi
    
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    echo "Installing system dependencies..."
    sudo apt-get update
    sudo apt-get install -y wkhtmltopdf tesseract-ocr ghostscript python3-tk
else
    echo "⚠️  Unknown OS. Please install wkhtmltopdf manually:"
    echo "   https://wkhtmltopdf.org/downloads.html"
fi

echo ""
echo "Step 2: Installing Python packages"
echo "========================================"

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install base requirements
echo "Installing base requirements..."
python -m pip install -r requirements-minimal.txt

echo ""
echo "Step 3: Installing HTML to PDF converters"
echo "========================================"
python -m pip install pdfkit weasyprint

echo ""
echo "Step 4: Installing PyTorch (for layout detection)"
echo "========================================"
# Install PyTorch (CPU version for compatibility)
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "Step 5: Installing Detectron2 (for layout detection)"
echo "========================================"
# Install detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

echo ""
echo "Step 6: Installing LayoutParser"
echo "========================================"
python -m pip install 'layoutparser[ocr]'

echo ""
echo "✅ Installation Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Configure your .env file:"
echo "   cp env.example .env"
echo "   nano .env  # Add your OPENAI_API_KEY"
echo ""
echo "2. Test the installation:"
echo "   python run_pipeline.py --ticker AAPL --form-type 10-K --limit 1"
echo ""
echo "3. Run agents:"
echo "   python run_agents_v2.py --doc-id AAPL_10-K_*"
echo ""
echo "4. Launch UI:"
echo "   streamlit run streamlit_app.py"
echo ""
