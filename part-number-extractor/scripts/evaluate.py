#!/usr/bin/env python
"""
Evaluation script for BOM Part Number NER model
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoTokenizer
from src.model.ner_model import BOMPartNumberNER
from src.data_preparation.data_loader import BOMDataset, load_bom_data
from src.data_preparation.preprocessor import BOMDataPreprocessor
from src.evaluation.metrics import compute_metrics, detailed_classification_report
from src.utils.logger import setup_logger
from transformers import Trainer, TrainingArguments


def main():
    parser = argparse.ArgumentParser(description='Evaluate BOM Part Number NER Model')
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model'
    )
    parser.add_argument(
        '--test_data',
        type=str,
        default='data/test.json',
        help='Test data path'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='evaluation_results',
        help='Output directory for evaluation results'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )
    args = parser.parse_args()

    # Setup logger
    logger = setup_logger('evaluation')
    logger.info("Starting evaluation script")

    # Load test data
    logger.info(f"Loading test data from {args.test_data}")
    test_data = load_bom_data(args.test_data)
    logger.info(f"Loaded {len(test_data)} test samples")

    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = BOMPartNumberNER.from_pretrained(args.model_path)
    
    # Get tokenizer (assume bert-base-uncased, adjust if needed)
    tokenizer_name = 'bert-base-uncased'  # Could read from config
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    preprocessor = BOMDataPreprocessor(tokenizer)

    # Create test dataset
    test_dataset = BOMDataset(test_data, preprocessor)

    # Create trainer for evaluation
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=args.batch_size,
        do_train=False,
        do_eval=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # Evaluate
    logger.info("Running evaluation...")
    results = trainer.evaluate()

    # Print results
    logger.info("\n" + "="*60)
    logger.info("Evaluation Results")
    logger.info("="*60)
    for metric, value in results.items():
        logger.info(f"{metric:30s}: {value:.4f}")
    logger.info("="*60)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()
