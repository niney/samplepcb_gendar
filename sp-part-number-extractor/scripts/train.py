#!/usr/bin/env python
"""
Training script for BOM Part Number NER model
"""

import argparse
import yaml
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoTokenizer
from src.model.ner_model import create_model
from src.training.trainer import train_model, create_training_args
from src.data_preparation.data_loader import BOMDataset, load_bom_data
from src.data_preparation.preprocessor import BOMDataPreprocessor
from src.evaluation.metrics import compute_metrics
from src.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description='Train Part Number NER Model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/bert_base.yaml',
        help='Config file path'
    )
    parser.add_argument(
        '--train_data',
        type=str,
        default='data/train.json',
        help='Training data path'
    )
    parser.add_argument(
        '--val_data',
        type=str,
        default='data/val.json',
        help='Validation data path'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='models/checkpoint',
        help='Output directory for model checkpoints'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default=None,
        help='Pretrained model name (overrides config)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (overrides config)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    args = parser.parse_args()

    # Setup logger
    logger = setup_logger('training')
    logger.info("Starting training script")

    # Load configuration
    config = {}
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {args.config}")
    else:
        logger.warning(f"Config file not found: {args.config}. Using defaults.")
        config = {
            'model': {'name': 'bert-base-uncased', 'num_labels': 3},
            'training': {
                'num_epochs': 10,
                'batch_size': 16,
                'learning_rate': 2e-5,
                'warmup_ratio': 0.1,
                'weight_decay': 0.01,
                'fp16': True
            }
        }

    # Override config with command line arguments
    if args.model_name:
        config['model']['name'] = args.model_name
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size

    # Load data
    logger.info(f"Loading training data from {args.train_data}")
    train_data = load_bom_data(args.train_data)
    logger.info(f"Loaded {len(train_data)} training samples")

    logger.info(f"Loading validation data from {args.val_data}")
    val_data = load_bom_data(args.val_data)
    logger.info(f"Loaded {len(val_data)} validation samples")

    # Initialize tokenizer and preprocessor
    model_name = config['model']['name']
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    preprocessor = BOMDataPreprocessor(tokenizer)

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = BOMDataset(train_data, preprocessor)
    val_dataset = BOMDataset(val_data, preprocessor)

    # Create model
    logger.info(f"Creating model: {model_name}")
    model = create_model(
        model_name=model_name,
        num_labels=config['model']['num_labels']
    )

    # Create training arguments
    training_config = config.get('training', {})
    learning_rate = training_config.get('learning_rate', 2e-5)
    # Ensure learning_rate is float (YAML may parse scientific notation as string)
    if isinstance(learning_rate, str):
        learning_rate = float(learning_rate)
    
    training_args = create_training_args(
        output_dir=args.output_dir,
        num_train_epochs=training_config.get('num_epochs', 10),
        per_device_train_batch_size=training_config.get('batch_size', 16),
        per_device_eval_batch_size=training_config.get('batch_size', 16) * 2,
        learning_rate=learning_rate,
        warmup_ratio=training_config.get('warmup_ratio', 0.1),
        weight_decay=training_config.get('weight_decay', 0.01),
        fp16=training_config.get('fp16', True),
    )

    # Train model
    logger.info("Starting model training...")
    trainer = train_model(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        output_dir=args.output_dir,
        training_args=training_args,
    )

    # Evaluate on validation set
    logger.info("Evaluating model...")
    eval_results = trainer.evaluate()
    
    logger.info("Evaluation Results:")
    for metric, value in eval_results.items():
        logger.info(f"  {metric}: {value:.4f}")

    # Save final model
    final_model_path = Path(args.output_dir) / 'final_model'
    logger.info(f"Saving final model to {final_model_path}")
    trainer.save_model(str(final_model_path))

    # Save evaluation results
    results_file = Path(args.output_dir) / 'final_results.json'
    with open(results_file, 'w') as f:
        json.dump(eval_results, f, indent=2)

    logger.info(f"Training completed! Model saved to {args.output_dir}")
    logger.info(f"Evaluation results saved to {results_file}")


if __name__ == '__main__':
    main()
