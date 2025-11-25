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

REM Install requirements
echo.
echo Installing required packages...
pip install -r requirements.txt

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
