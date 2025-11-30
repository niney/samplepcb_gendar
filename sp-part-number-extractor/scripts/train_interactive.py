#!/usr/bin/env python
"""
Interactive Training Wizard
Guides users through model training setup
"""

import sys
from pathlib import Path
import yaml
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import inquirer
    from colorama import init, Fore, Style
    init(autoreset=True)
    HAS_INTERACTIVE = True
except ImportError:
    HAS_INTERACTIVE = False
    print("Warning: inquirer or colorama not installed. Install with: pip install inquirer colorama")


class TrainingWizard:
    """대화형 학습 설정 마법사"""

    def __init__(self):
        self.config = {}

    def run(self):
        """마법사 실행"""
        if not HAS_INTERACTIVE:
            print("Error: Required packages not installed")
            print("Install with: pip install inquirer colorama")
            return

        print(f"{Fore.CYAN}{'='*80}")
        print(f"{Fore.GREEN}Model Training Wizard")
        print(f"{Fore.CYAN}{'='*80}\n")

        # 1. Model selection
        print(f"{Fore.YELLOW}Step 1: Select Base Model\n")
        questions = [
            inquirer.List(
                'model',
                message="Select base model",
                choices=[
                    ('BERT-base (Fast, 90-93% accuracy)', 'bert-base-uncased'),
                    ('RoBERTa-base (Balanced, 93-95% accuracy)', 'roberta-base'),
                    ('DeBERTa-v3-base (Best, 95-97% accuracy)', 'microsoft/deberta-v3-base'),
                ],
            ),
        ]
        model_answer = inquirer.prompt(questions)
        if not model_answer:
            return
        self.config['model_name'] = model_answer['model']

        # 2. Training parameters
        print(f"\n{Fore.YELLOW}Step 2: Training Parameters\n")
        questions = [
            inquirer.Text(
                'epochs',
                message="Number of training epochs",
                default="10",
                validate=lambda _, x: x.isdigit() and int(x) > 0,
            ),
            inquirer.Text(
                'batch_size',
                message="Batch size",
                default="16",
                validate=lambda _, x: x.isdigit() and int(x) > 0,
            ),
            inquirer.Text(
                'learning_rate',
                message="Learning rate",
                default="2e-5",
            ),
        ]
        train_answers = inquirer.prompt(questions)
        if not train_answers:
            return
        
        self.config['num_epochs'] = int(train_answers['epochs'])
        self.config['batch_size'] = int(train_answers['batch_size'])
        self.config['learning_rate'] = float(train_answers['learning_rate'])

        # 3. Data paths
        print(f"\n{Fore.YELLOW}Step 3: Data Paths\n")
        questions = [
            inquirer.Text(
                'train_data',
                message="Training data path",
                default="data/train.json",
            ),
            inquirer.Text(
                'val_data',
                message="Validation data path",
                default="data/val.json",
            ),
            inquirer.Text(
                'output_dir',
                message="Model output directory",
                default="models/checkpoint",
            ),
        ]
        data_answers = inquirer.prompt(questions)
        if not data_answers:
            return
        self.config.update(data_answers)

        # 4. Advanced settings
        print(f"\n{Fore.YELLOW}Step 4: Advanced Settings\n")
        questions = [
            inquirer.Confirm(
                'advanced',
                message="Configure advanced settings?",
                default=False,
            ),
        ]
        adv_answer = inquirer.prompt(questions)
        if not adv_answer:
            return

        if adv_answer['advanced']:
            questions = [
                inquirer.Text(
                    'warmup_ratio',
                    message="Warmup ratio",
                    default="0.1",
                ),
                inquirer.Text(
                    'weight_decay',
                    message="Weight decay",
                    default="0.01",
                ),
                inquirer.Confirm(
                    'fp16',
                    message="Use mixed precision (FP16)?",
                    default=True,
                ),
            ]
            adv_answers = inquirer.prompt(questions)
            if adv_answers:
                self.config['warmup_ratio'] = float(adv_answers['warmup_ratio'])
                self.config['weight_decay'] = float(adv_answers['weight_decay'])
                self.config['fp16'] = adv_answers['fp16']

        # 5. Review and confirm
        self.display_config()

        questions = [
            inquirer.Confirm(
                'start',
                message="Start training with these settings?",
                default=True,
            ),
        ]
        confirm = inquirer.prompt(questions)
        
        if not confirm or not confirm['start']:
            print(f"{Fore.YELLOW}Training cancelled.")
            return

        # Save config and provide command
        config_file = self.save_config()
        self.show_command(config_file)

    def display_config(self):
        """설정 표시"""
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.GREEN}Training Configuration")
        print(f"{Fore.CYAN}{'='*80}\n")

        for key, value in self.config.items():
            print(f"{Fore.YELLOW}{key:20s}: {Fore.WHITE}{value}")
        print()

    def save_config(self) -> str:
        """설정 저장"""
        config_dir = Path('configs/generated')
        config_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config_file = config_dir / f'config_{timestamp}.yaml'

        # Format config for YAML
        yaml_config = {
            'model': {
                'name': self.config.get('model_name', 'bert-base-uncased'),
                'num_labels': 3,
            },
            'training': {
                'num_epochs': self.config.get('num_epochs', 10),
                'batch_size': self.config.get('batch_size', 16),
                'learning_rate': self.config.get('learning_rate', 2e-5),
                'warmup_ratio': self.config.get('warmup_ratio', 0.1),
                'weight_decay': self.config.get('weight_decay', 0.01),
                'fp16': self.config.get('fp16', True),
            }
        }

        with open(config_file, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=False)

        print(f"{Fore.GREEN}Configuration saved to {config_file}")
        return str(config_file)

    def show_command(self, config_file: str):
        """학습 명령 표시"""
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.GREEN}Ready to Train!")
        print(f"{Fore.CYAN}{'='*80}\n")
        
        command = f"""python scripts/train.py \\
    --config {config_file} \\
    --train_data {self.config.get('train_data', 'data/train.json')} \\
    --val_data {self.config.get('val_data', 'data/val.json')} \\
    --output_dir {self.config.get('output_dir', 'models/checkpoint')}"""
        
        print(f"{Fore.YELLOW}Run this command to start training:\n")
        print(f"{Fore.WHITE}{command}\n")


def main():
    wizard = TrainingWizard()
    try:
        wizard.run()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Cancelled by user")
    except Exception as e:
        print(f"\n{Fore.RED}Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
