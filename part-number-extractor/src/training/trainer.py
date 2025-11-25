"""
Training module for BOM NER model
"""

import torch
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from pathlib import Path
from typing import Optional, Dict
import json


def create_training_args(
    output_dir: str = './models/checkpoint',
    num_train_epochs: int = 10,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 32,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    logging_steps: int = 100,
    save_strategy: str = 'epoch',
    evaluation_strategy: str = 'epoch',
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = 'f1',
    fp16: bool = True,
    **kwargs
) -> TrainingArguments:
    """
    Create training arguments for Hugging Face Trainer
    
    Args:
        output_dir: Directory to save model checkpoints
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Training batch size per device
        per_device_eval_batch_size: Evaluation batch size per device
        learning_rate: Learning rate
        warmup_ratio: Warmup ratio for learning rate scheduler
        weight_decay: Weight decay for regularization
        logging_steps: Log every N steps
        save_strategy: When to save checkpoints
        evaluation_strategy: When to evaluate
        load_best_model_at_end: Load best model at end
        metric_for_best_model: Metric to use for best model selection
        fp16: Use mixed precision training
        **kwargs: Additional arguments
    
    Returns:
        TrainingArguments instance
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        logging_dir=f'{output_dir}/logs',
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        eval_strategy=evaluation_strategy,  # Updated: evaluation_strategy -> eval_strategy
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=True,
        fp16=fp16 and torch.cuda.is_available(),
        save_total_limit=3,  # Keep only 3 best checkpoints
        report_to=[],  # Disable reporting to avoid tensorboard dependency
        **kwargs
    )

    return training_args


def train_model(
    model,
    train_dataset,
    eval_dataset,
    compute_metrics,
    output_dir: str = './models/checkpoint',
    training_args: Optional[TrainingArguments] = None,
    early_stopping_patience: int = 5,
) -> Trainer:
    """
    Train BOM NER model
    
    Args:
        model: BOMPartNumberNER model
        train_dataset: Training dataset
        eval_dataset: Validation dataset
        compute_metrics: Function to compute evaluation metrics
        output_dir: Directory to save model
        training_args: Training arguments (optional, will create default if None)
        early_stopping_patience: Patience for early stopping
    
    Returns:
        Trained Trainer instance
    """
    # Create training arguments if not provided
    if training_args is None:
        training_args = create_training_args(output_dir=output_dir)

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    final_model_path = Path(output_dir) / 'final_model'
    trainer.save_model(str(final_model_path))
    print(f"Model saved to {final_model_path}")

    # Save training results
    results_path = Path(output_dir) / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(trainer.state.log_history, f, indent=2)

    return trainer


class BOMTrainingConfig:
    """Configuration class for training parameters"""

    def __init__(self, config_dict: Optional[Dict] = None):
        """
        Args:
            config_dict: Dictionary with configuration parameters
        """
        # Default configuration
        self.model_name = 'bert-base-uncased'
        self.num_labels = 3
        self.max_length = 512
        
        self.num_train_epochs = 10
        self.batch_size = 16
        self.learning_rate = 2e-5
        self.warmup_ratio = 0.1
        self.weight_decay = 0.01
        
        self.output_dir = './models/checkpoint'
        self.fp16 = True
        
        # Update with provided config
        if config_dict:
            self.__dict__.update(config_dict)

    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return self.__dict__.copy()

    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load configuration from YAML file"""
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)

    def save_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        import yaml
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
