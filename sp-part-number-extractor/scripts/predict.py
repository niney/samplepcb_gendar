#!/usr/bin/env python
"""
Prediction script for BOM Part Number extraction
"""

import argparse
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.inference.predictor import SpPartNumberPredictor
from src.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description='Predict Part Numbers from BOM file')
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='Input BOM file (CSV or Excel)'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='output_predictions.csv',
        help='Output file path'
    )
    parser.add_argument(
        '--confidence_threshold',
        type=float,
        default=0.7,
        help='Minimum confidence threshold'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--has_header',
        action='store_true',
        help='Input file has header row'
    )
    args = parser.parse_args()

    # Setup logger
    logger = setup_logger('prediction')

    # Load model
    logger.info(f"Loading model from {args.model_path}...")
    predictor = SpPartNumberPredictor(args.model_path)

    # Read input file
    logger.info(f"Reading input file: {args.input_file}")
    if args.input_file.endswith('.csv'):
        df = pd.read_csv(args.input_file, header=0 if args.has_header else None)
    elif args.input_file.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(args.input_file, header=0 if args.has_header else None)
    else:
        logger.error(f"Unsupported file format: {args.input_file}")
        return

    logger.info(f"Loaded {len(df)} rows")

    # Convert to list of rows
    rows = df.fillna('').astype(str).values.tolist()

    # Run predictions
    logger.info(f"Running predictions on {len(rows)} rows...")
    results = predictor.batch_predict(rows, batch_size=args.batch_size)

    # Add predictions to dataframe
    df['predicted_part_number'] = [r['part_number'] for r in results]
    df['confidence'] = [r['confidence'] for r in results]
    df['cell_index'] = [r['cell_index'] for r in results]
    df['needs_review'] = df['confidence'] < args.confidence_threshold

    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == '.csv':
        df.to_csv(output_path, index=False)
    else:
        df.to_excel(output_path, index=False)

    # Print statistics
    total = len(results)
    found = sum(1 for r in results if r['part_number'] is not None)
    high_conf = sum(1 for r in results if r['confidence'] >= args.confidence_threshold)
    needs_review = total - high_conf

    logger.info("\n" + "="*60)
    logger.info("Prediction Summary")
    logger.info("="*60)
    logger.info(f"Total rows processed:        {total}")
    logger.info(f"Part numbers found:          {found} ({found/total*100:.1f}%)")
    logger.info(f"High confidence (>={args.confidence_threshold}): {high_conf} ({high_conf/total*100:.1f}%)")
    logger.info(f"Needs review:                {needs_review} ({needs_review/total*100:.1f}%)")
    logger.info(f"Results saved to:            {output_path}")
    logger.info("="*60)


if __name__ == '__main__':
    main()
