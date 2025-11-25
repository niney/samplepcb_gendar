"""
Data loader and dataset classes for BOM NER training
"""

import json
import pandas as pd
from torch.utils.data import Dataset
from typing import List, Dict, Optional
from pathlib import Path


class BOMDataset(Dataset):
    """PyTorch Dataset for BOM NER data"""

    def __init__(self, data: List[Dict], preprocessor):
        """
        Args:
            data: List of dictionaries with 'cells' and 'labels' keys
            preprocessor: BOMDataPreprocessor instance
        """
        self.data = data
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        processed = self.preprocessor.prepare_row_for_ner(
            item['cells'],
            item.get('labels')
        )
        return processed


def load_bom_data(file_path: str) -> List[Dict]:
    """
    Load BOM data from JSON file
    
    Expected format:
    [
        {
            "row_id": "001",
            "cells": ["C29 C33", "CC0402KRX7R9BB102", "CAP CER 1000PF", ...],
            "labels": ["REFERENCE", "PART_NUMBER", "DESCRIPTION", ...]
        },
        ...
    ]
    
    Args:
        file_path: Path to JSON data file
    
    Returns:
        List of data dictionaries
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_bom_data(data: List[Dict], file_path: str):
    """
    Save BOM data to JSON file
    
    Args:
        data: List of data dictionaries
        file_path: Output file path
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_unlabeled_bom(file_path: str, has_header: bool = False) -> List[List[str]]:
    """
    Load unlabeled BOM file (CSV or Excel) for prediction
    
    Args:
        file_path: Path to BOM file
        has_header: Whether file has header row
    
    Returns:
        List of rows, where each row is a list of cell values
    """
    file_path = str(file_path)
    
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, header=0 if has_header else None)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path, header=0 if has_header else None)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # Convert to list of lists, handling NaN values
    rows = df.fillna('').astype(str).values.tolist()
    return rows


def split_data(
    data: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> tuple:
    """
    Split data into train, validation, and test sets
    
    Args:
        data: Full dataset
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    import random
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Shuffle data
    random.seed(random_seed)
    data_copy = data.copy()
    random.shuffle(data_copy)
    
    # Calculate split indices
    total = len(data_copy)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # Split
    train_data = data_copy[:train_end]
    val_data = data_copy[train_end:val_end]
    test_data = data_copy[val_end:]
    
    return train_data, val_data, test_data
