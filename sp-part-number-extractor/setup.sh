#!/bin/bash
# Setup script for Part Number Extractor
# Run this script to set up the project for the first time

set -e  # Exit on error

echo "=========================================="
echo "Part Number Extractor - Setup Script"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python --version

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Linux/Mac
    source venv/bin/activate
fi
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
echo ""
echo "Installing required packages..."
pip install -r requirements.txt

# Test installation
echo ""
echo "Testing installation..."
python test_installation.py

echo ""
echo "=========================================="
echo "Setup completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate  (Linux/Mac)"
echo "   venv\\Scripts\\activate     (Windows)"
echo ""
echo "2. Create sample data:"
echo "   python examples/create_sample_data.py"
echo ""
echo "3. Start labeling:"
echo "   python scripts/interactive_label.py --input data/raw/your_bom.csv"
echo ""
echo "4. Train model:"
echo "   python scripts/train_interactive.py"
echo ""
