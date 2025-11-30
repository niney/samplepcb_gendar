@echo off
REM Setup script for Part Number Extractor (Windows)
REM Run this script to set up the project for the first time

echo ==========================================
echo Part Number Extractor - Setup Script
echo ==========================================
echo.

REM Check Python version
echo Checking Python version...
python --version

REM Create virtual environment
echo.
echo Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo Virtual environment created
) else (
    echo Virtual environment already exists
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Create setup.py if not exists
echo.
echo Checking setup.py...
if not exist "setup.py" (
    echo Creating setup.py...
    python -c "content = '''from setuptools import setup, find_packages\n\nsetup(\n    name=\"sp-part-number-extractor\",\n    version=\"1.0.0\",\n    description=\"BOM Part Number Extractor using NER\",\n    packages=find_packages(),\n    python_requires=\">=3.8\",\n    install_requires=[\n        \"torch>=1.9.0\",\n        \"transformers>=4.20.0\",\n        \"pandas>=1.3.0\",\n        \"numpy>=1.20.0\",\n        \"scikit-learn>=0.24.0\",\n        \"pyyaml>=5.4.0\",\n        \"tqdm>=4.60.0\",\n        \"safetensors>=0.3.0\",\n    ],\n)\n'''; open('setup.py', 'w').write(content)"
    echo setup.py created
) else (
    echo setup.py already exists
)

REM Install requirements
echo.
echo Installing required packages...
pip install -r requirements.txt

REM Install package in editable mode
echo.
echo Installing package in editable mode...
pip install -e .

REM Test installation
echo.
echo Testing installation...
python test_installation.py

echo.
echo ==========================================
echo Setup completed!
echo ==========================================
echo.
echo Next steps:
echo 1. Activate virtual environment:
echo    venv\Scripts\activate
echo.
echo 2. Create sample data:
echo    python examples\create_sample_data.py
echo.
echo 3. Start labeling:
echo    python scripts\interactive_label.py --input data\raw\your_bom.csv
echo.
echo 4. Train model:
echo    python scripts\train_interactive.py
echo.

pause
