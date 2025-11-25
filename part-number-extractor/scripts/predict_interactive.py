#!/usr/bin/env python
"""
Interactive Prediction Tool
User-friendly CLI for predicting Part Numbers
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False

from src.inference.predictor import PartNumberPredictor


class InteractivePredictor:
    """대화형 Part Number 예측 도구"""

    def __init__(self, model_path: str):
        if HAS_COLORAMA:
            print(f"{Fore.CYAN}Loading model from {model_path}...")
        else:
            print(f"Loading model from {model_path}...")
            
        self.predictor = PartNumberPredictor(model_path)
        
        if HAS_COLORAMA:
            print(f"{Fore.GREEN}Model loaded successfully!\n")
        else:
            print("Model loaded successfully!\n")

    def predict_single_row(self):
        """단일 행 예측 모드"""
        self.print_header("Single Row Prediction Mode")

        while True:
            if HAS_COLORAMA:
                print(f"{Fore.YELLOW}Enter BOM row (comma-separated cells):")
                print(f"{Fore.CYAN}Example: C29 C33,CC0402KRX7R9BB102,CAP CER 1000PF,9,Yageo")
                print(f"{Fore.MAGENTA}(Type 'q' to quit, 'f' for file mode)\n")
                user_input = input(f"{Fore.WHITE}> ").strip()
            else:
                print("Enter BOM row (comma-separated cells):")
                print("Example: C29 C33,CC0402KRX7R9BB102,CAP CER 1000PF,9,Yageo")
                print("(Type 'q' to quit, 'f' for file mode)\n")
                user_input = input("> ").strip()

            if user_input.lower() == 'q':
                break
            elif user_input.lower() == 'f':
                self.predict_file()
                continue

            # Parse cells
            cells = [cell.strip() for cell in user_input.split(',')]

            # Predict
            result = self.predictor.predict(cells)

            # Display result
            self.display_result(cells, result)

    def display_result(self, cells, result):
        """예측 결과 표시"""
        self.print_header("Prediction Result")

        # Show cells
        for idx, cell in enumerate(cells):
            if HAS_COLORAMA:
                print(f"{Fore.WHITE}[{idx}] {cell}")
            else:
                print(f"[{idx}] {cell}")

        # Show prediction
        if HAS_COLORAMA:
            print(f"\n{Fore.YELLOW}Predicted Part Number:")
        else:
            print("\nPredicted Part Number:")
            
        if result['part_number']:
            confidence = result['confidence']
            color = Fore.GREEN if confidence > 0.8 else Fore.YELLOW if HAS_COLORAMA else ""
            
            if HAS_COLORAMA:
                print(f"{Fore.GREEN}  → {result['part_number']}")
                print(f"{color}  Confidence: {confidence:.2%}")
                print(f"{Fore.CYAN}  Cell Index: {result['cell_index']}")
            else:
                print(f"  → {result['part_number']}")
                print(f"  Confidence: {confidence:.2%}")
                print(f"  Cell Index: {result['cell_index']}")
        else:
            if HAS_COLORAMA:
                print(f"{Fore.RED}  → Not found")
            else:
                print("  → Not found")
        print()

    def predict_file(self):
        """파일 예측 모드"""
        if HAS_COLORAMA:
            print(f"\n{Fore.YELLOW}Enter file path:")
            file_path = input(f"{Fore.WHITE}> ").strip()
        else:
            print("\nEnter file path:")
            file_path = input("> ").strip()

        try:
            # Load file
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, header=None)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path, header=None)
            else:
                print("Error: Unsupported file format")
                return

            rows = df.fillna('').astype(str).values.tolist()

            if HAS_COLORAMA:
                print(f"\n{Fore.CYAN}Processing {len(rows)} rows...")
            else:
                print(f"\nProcessing {len(rows)} rows...")

            # Batch predict
            results = self.predictor.batch_predict(rows)

            # Show statistics
            self.display_batch_stats(results)

            # Ask to save
            if HAS_COLORAMA:
                print(f"\n{Fore.YELLOW}Save results? (y/n):")
                save = input(f"{Fore.WHITE}> ").strip().lower()
            else:
                print("\nSave results? (y/n):")
                save = input("> ").strip().lower()

            if save == 'y':
                output_file = file_path.replace('.csv', '_predicted.csv').replace('.xlsx', '_predicted.csv')
                df['predicted_part_number'] = [r['part_number'] for r in results]
                df['confidence'] = [r['confidence'] for r in results]
                df['cell_index'] = [r['cell_index'] for r in results]
                df.to_csv(output_file, index=False)
                
                if HAS_COLORAMA:
                    print(f"{Fore.GREEN}Results saved to {output_file}")
                else:
                    print(f"Results saved to {output_file}")

        except Exception as e:
            if HAS_COLORAMA:
                print(f"{Fore.RED}Error: {e}")
            else:
                print(f"Error: {e}")

    def display_batch_stats(self, results):
        """배치 예측 통계"""
        total = len(results)
        found = sum(1 for r in results if r['part_number'] is not None)
        high_conf = sum(1 for r in results if r['confidence'] > 0.8)
        avg_conf = sum(r['confidence'] for r in results) / total if total > 0 else 0

        self.print_header("Batch Prediction Statistics")
        
        print(f"Total rows:          {total}")
        print(f"Part numbers found:  {found} ({found/total:.1%})")
        print(f"High confidence:     {high_conf} ({high_conf/total:.1%})")
        print(f"Average confidence:  {avg_conf:.2%}")
        print()

    def print_header(self, title):
        """Print section header"""
        if HAS_COLORAMA:
            print(f"\n{Fore.CYAN}{'='*80}")
            print(f"{Fore.GREEN}{title}")
            print(f"{Fore.CYAN}{'='*80}\n")
        else:
            print(f"\n{'='*80}")
            print(title)
            print(f"{'='*80}\n")

    def run(self):
        """메인 실행"""
        self.print_header("Interactive Part Number Predictor")

        while True:
            if HAS_COLORAMA:
                print(f"{Fore.MAGENTA}Select mode:")
                print(f"{Fore.YELLOW}1. Single row prediction")
                print(f"{Fore.YELLOW}2. File prediction")
                print(f"{Fore.YELLOW}3. Quit")
                choice = input(f"\n{Fore.WHITE}> ").strip()
            else:
                print("Select mode:")
                print("1. Single row prediction")
                print("2. File prediction")
                print("3. Quit")
                choice = input("\n> ").strip()

            if choice == '1':
                self.predict_single_row()
            elif choice == '2':
                self.predict_file()
            elif choice == '3':
                if HAS_COLORAMA:
                    print(f"{Fore.GREEN}Goodbye!")
                else:
                    print("Goodbye!")
                break
            else:
                if HAS_COLORAMA:
                    print(f"{Fore.RED}Invalid choice. Try again.")
                else:
                    print("Invalid choice. Try again.")


def main():
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description='Interactive Part Number Predictor')
    parser.add_argument(
        '--model_path',
        type=str,
        required=False,
        default=None,
        help='Path to trained model (optional - will use demo mode if not provided)'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run in demo mode with untrained model (for testing)'
    )
    args = parser.parse_args()

    # Check model path
    if args.model_path is None and not args.demo:
        print("Error: Please provide --model_path or use --demo mode")
        print("\nUsage:")
        print("  python scripts/predict_interactive.py --model_path models/my_model/final_model")
        print("  python scripts/predict_interactive.py --demo  (for testing without trained model)")
        return
    
    if args.demo:
        print("\n" + "="*60)
        print("⚠️  DEMO MODE - Using untrained model")
        print("    Predictions will NOT be accurate!")
        print("    Train a model first for real predictions.")
        print("="*60 + "\n")
        # Use a default path that will trigger demo model creation
        args.model_path = "__demo__"
    
    # Verify model path exists (unless demo mode)
    if args.model_path != "__demo__":
        model_path = Path(args.model_path)
        if not model_path.exists():
            print(f"Error: Model path not found: {args.model_path}")
            print("\nPlease train a model first:")
            print("  python scripts/train_interactive.py")
            print("\nOr use demo mode for testing:")
            print("  python scripts/predict_interactive.py --demo")
            return

    try:
        predictor = InteractivePredictor(args.model_path)
        predictor.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
