from setuptools import setup, find_packages

setup(
    name="sp-part-number-extractor",
    version="1.0.0",
    description="BOM Part Number Extractor using NER",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=0.24.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.60.0",
        "safetensors>=0.3.0",
    ],
)
