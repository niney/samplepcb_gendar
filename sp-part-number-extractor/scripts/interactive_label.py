#!/usr/bin/env python
"""
Interactive BOM Labeling Tool
Helps users label BOM data for training
"""

import sys
from pathlib import Path
import json
import pandas as pd
from colorama import init, Fore, Style

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

init(autoreset=True)


class InteractiveLabelingTool:
    """대화형 BOM 라벨링 도구"""

    def __init__(self, input_file: str, output_file: str = 'data/labeled.json'):
        self.input_file = input_file
        self.output_file = output_file
        self.labels = [
            'REFERENCE',
            'PART_NUMBER',
            'DESCRIPTION',
            'QUANTITY',
            'MANUFACTURER',
            'PACKAGE',
            'OTHER'
        ]
        self.data = []
        self.current_index = 0
        self.rows = []

    def load_data(self):
        """Load input BOM file"""
        print(f"{Fore.CYAN}Loading data from {self.input_file}...")
        
        if self.input_file.endswith('.csv'):
            df = pd.read_csv(self.input_file, header=None)
        elif self.input_file.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(self.input_file, header=None)
        else:
            raise ValueError(f"Unsupported file format: {self.input_file}")
        
        self.rows = df.fillna('').astype(str).values.tolist()
        print(f"{Fore.GREEN}Loaded {len(self.rows)} rows")

    def display_row(self, cells):
        """행 데이터 표시"""
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.YELLOW}Row {self.current_index + 1} of {len(self.rows)}")
        print(f"{Fore.CYAN}{'='*80}\n")

        for idx, cell in enumerate(cells):
            # Truncate long cells
            cell_str = str(cell)
            if len(cell_str) > 60:
                cell_str = cell_str[:57] + '...'
            print(f"{Fore.GREEN}[{idx}] {Fore.WHITE}{cell_str}")

    def display_menu(self):
        """라벨 선택 메뉴"""
        print(f"\n{Fore.MAGENTA}Available Labels:")
        for idx, label in enumerate(self.labels, 1):
            color = Fore.YELLOW if label != 'PART_NUMBER' else Fore.RED
            print(f"{color}{idx}. {label}")
        
        print(f"\n{Fore.CYAN}Controls:")
        print(f"  {Fore.WHITE}1-{len(self.labels)}: {Fore.CYAN}Select label")
        print(f"  {Fore.WHITE}n: {Fore.CYAN}Next row (after labeling)")
        print(f"  {Fore.WHITE}s: {Fore.CYAN}Save progress")
        print(f"  {Fore.WHITE}q: {Fore.CYAN}Quit and save")

    def label_row(self, cells):
        """단일 행 라벨링"""
        self.display_row(cells)
        self.display_menu()

        row_labels = []
        
        for cell_idx, cell in enumerate(cells):
            print(f"\n{Fore.GREEN}Label for Cell [{cell_idx}]:")
            print(f"{Fore.WHITE}{cell}")
            print(f"{Fore.YELLOW}Enter label number (1-{len(self.labels)}): ", end='')

            while True:
                try:
                    choice = input().strip()
                    
                    if choice.lower() == 'q':
                        return None
                    
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(self.labels):
                        row_labels.append(self.labels[choice_num - 1])
                        break
                    else:
                        print(f"{Fore.RED}Invalid choice. Try again (1-{len(self.labels)}): ", end='')
                except ValueError:
                    print(f"{Fore.RED}Please enter a number (1-{len(self.labels)}): ", end='')

        return row_labels

    def run(self):
        """라벨링 세션 실행"""
        print(f"{Fore.CYAN}{'='*80}")
        print(f"{Fore.GREEN}BOM Interactive Labeling Tool")
        print(f"{Fore.CYAN}{'='*80}\n")

        self.load_data()

        # Load existing progress if available
        if Path(self.output_file).exists():
            print(f"{Fore.YELLOW}Found existing progress. Loading...")
            with open(self.output_file, 'r') as f:
                self.data = json.load(f)
            self.current_index = len(self.data)
            print(f"{Fore.GREEN}Resuming from row {self.current_index + 1}")

        while self.current_index < len(self.rows):
            cells = [str(cell) for cell in self.rows[self.current_index]]
            
            labels = self.label_row(cells)
            
            if labels is None:
                print(f"{Fore.YELLOW}Quitting...")
                break

            # Save labeled data
            self.data.append({
                'row_id': f"{self.current_index:04d}",
                'cells': cells,
                'labels': labels
            })

            self.current_index += 1

            # Progress
            progress = (self.current_index / len(self.rows)) * 100
            print(f"\n{Fore.GREEN}Progress: {progress:.1f}% ({self.current_index}/{len(self.rows)})")

            # Auto-save every 10 rows
            if self.current_index % 10 == 0:
                self.save_data()
                print(f"{Fore.CYAN}Auto-saved!")

        # Final save
        self.save_data()
        print(f"\n{Fore.GREEN}Labeling session completed!")
        print(f"{Fore.CYAN}Total labeled: {len(self.data)} rows")
        print(f"{Fore.CYAN}Saved to: {self.output_file}")

    def save_data(self):
        """데이터 저장"""
        output_path = Path(self.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Interactive BOM Labeling Tool')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input BOM file (CSV or Excel)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/labeled.json',
        help='Output JSON file'
    )
    args = parser.parse_args()

    try:
        tool = InteractiveLabelingTool(args.input, args.output)
        tool.run()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Interrupted by user. Progress saved.")
    except Exception as e:
        print(f"\n{Fore.RED}Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
