#!/usr/bin/env python
"""
Data Splitting Utility
Split labeled data into train, validation, and test sets
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_preparation.data_loader import load_bom_data, save_bom_data, split_data


def load_data_from_path(input_path: str) -> list:
    """
    Load data from file or folder

    Args:
        input_path: Path to a JSON file or folder containing JSON files

    Returns:
        List of all loaded data
    """
    path = Path(input_path)

    if path.is_file():
        return load_bom_data(str(path))
    elif path.is_dir():
        all_data = []
        json_files = sorted(path.glob("*.json"))

        if not json_files:
            raise ValueError(f"No JSON files found in folder: {input_path}")

        for json_file in json_files:
            print(f"  Loading: {json_file.name}")
            data = load_bom_data(str(json_file))
            all_data.extend(data)

        print(f"  Loaded {len(json_files)} files")
        return all_data
    else:
        raise ValueError(f"Path does not exist: {input_path}")


def main():
    parser = argparse.ArgumentParser(description='Split data into train/val/test sets')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input labeled data file (JSON) or folder containing JSON files'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.7,
        help='Training set ratio (default: 0.7)'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.15,
        help='Validation set ratio (default: 0.15)'
    )
    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.15,
        help='Test set ratio (default: 0.15)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data',
        help='Output directory (default: data)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    data = load_data_from_path(args.input)
    print(f"Loaded {len(data)} samples total")

    print(f"\nSplitting data...")
    print(f"  Train: {args.train_ratio:.1%}")
    print(f"  Val:   {args.val_ratio:.1%}")
    print(f"  Test:  {args.test_ratio:.1%}")

    train_data, val_data, test_data = split_data(
        data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed
    )

    # Save splits
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = output_dir / 'train.json'
    val_file = output_dir / 'val.json'
    test_file = output_dir / 'test.json'

    print(f"\nSaving splits...")
    save_bom_data(train_data, str(train_file))
    print(f"  Train: {len(train_data)} samples -> {train_file}")

    save_bom_data(val_data, str(val_file))
    print(f"  Val:   {len(val_data)} samples -> {val_file}")

    save_bom_data(test_data, str(test_file))
    print(f"  Test:  {len(test_data)} samples -> {test_file}")

    print(f"\n✓ Data splitting completed!")


if __name__ == '__main__':
    main()
