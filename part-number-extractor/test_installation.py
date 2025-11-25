"""
Simple test to verify installation
"""

import sys

def test_imports():
    """Test if required packages are installed"""
    print("Testing package imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("✗ PyTorch not installed")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError:
        print("✗ Transformers not installed")
        return False
    
    try:
        import pandas
        print(f"✓ Pandas {pandas.__version__}")
    except ImportError:
        print("✗ Pandas not installed")
        return False
    
    try:
        import numpy
        print(f"✓ NumPy {numpy.__version__}")
    except ImportError:
        print("✗ NumPy not installed")
        return False
    
    try:
        from seqeval.metrics import f1_score
        print(f"✓ seqeval installed")
    except ImportError:
        print("✗ seqeval not installed")
        return False
    
    print("\n✓ All required packages are installed!")
    return True

def test_project_structure():
    """Test if project structure is correct"""
    print("\nTesting project structure...")
    from pathlib import Path
    
    required_dirs = [
        'src/data_preparation',
        'src/model',
        'src/training',
        'src/evaluation',
        'src/inference',
        'src/utils',
        'scripts',
        'configs',
        'data',
        'models',
        'logs',
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} not found")
            all_exist = False
    
    if all_exist:
        print("\n✓ Project structure is correct!")
    else:
        print("\n✗ Some directories are missing")
    
    return all_exist

def test_module_imports():
    """Test if custom modules can be imported"""
    print("\nTesting custom module imports...")
    
    try:
        from src.model.ner_model import create_model
        print("✓ src.model.ner_model")
    except ImportError as e:
        print(f"✗ src.model.ner_model: {e}")
        return False
    
    try:
        from src.data_preparation.preprocessor import BOMDataPreprocessor
        print("✓ src.data_preparation.preprocessor")
    except ImportError as e:
        print(f"✗ src.data_preparation.preprocessor: {e}")
        return False
    
    try:
        from src.training.trainer import train_model
        print("✓ src.training.trainer")
    except ImportError as e:
        print(f"✗ src.training.trainer: {e}")
        return False
    
    try:
        from src.inference.predictor import PartNumberPredictor
        print("✓ src.inference.predictor")
    except ImportError as e:
        print(f"✗ src.inference.predictor: {e}")
        return False
    
    print("\n✓ All custom modules can be imported!")
    return True

def main():
    """Run all tests"""
    print("="*80)
    print("Part Number Extractor - Installation Test")
    print("="*80)
    print()
    
    results = []
    
    # Test package imports
    results.append(test_imports())
    
    # Test project structure
    results.append(test_project_structure())
    
    # Test custom module imports
    results.append(test_module_imports())
    
    # Summary
    print("\n" + "="*80)
    if all(results):
        print("✓ ALL TESTS PASSED!")
        print("Your installation is ready to use.")
        print("\nNext steps:")
        print("1. Prepare your BOM data in data/raw/")
        print("2. Run: python scripts/interactive_label.py --input data/raw/your_bom.csv")
        print("3. Run: python scripts/train_interactive.py")
    else:
        print("✗ SOME TESTS FAILED")
        print("Please check the error messages above and:")
        print("1. Make sure virtual environment is activated: source venv/bin/activate")
        print("2. Install packages: pip install -r requirements.txt")
        print("3. Check project structure")
    print("="*80)

if __name__ == '__main__':
    main()
