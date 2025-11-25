"""
BOM Data Preprocessor
Converts BOM row data into format suitable for NER model training and inference.
"""

import torch
from transformers import PreTrainedTokenizer
from typing import List, Dict, Optional


class BOMDataPreprocessor:
    """BOM 데이터 전처리 클래스"""

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        """
        Args:
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = {'O': 0, 'B-PART': 1, 'I-PART': 2}
        self.id2label = {v: k for k, v in self.label2id.items()}

    def prepare_row_for_ner(
        self,
        cells: List[str],
        labels: Optional[List[str]] = None
    ) -> Dict:
        """
        행 데이터를 NER 형식으로 변환
        
        Format: [CLS] cell1 [SEP] cell2 [SEP] cell3 [SEP] ...
        
        Args:
            cells: List of cell values in the row
            labels: List of labels corresponding to each cell (optional)
        
        Returns:
            Dictionary with input_ids, attention_mask, and labels (if provided)
        """
        # Build tokens and labels
        tokens = [self.tokenizer.cls_token]
        token_labels = [self.label2id['O']]  # CLS token is always 'O'

        for idx, cell in enumerate(cells):
            # Tokenize cell content
            cell_text = str(cell) if cell is not None else ""
            cell_tokens = self.tokenizer.tokenize(cell_text)
            
            # Add cell tokens
            tokens.extend(cell_tokens)
            
            # Add SEP token after each cell
            tokens.append(self.tokenizer.sep_token)

            # Assign labels if provided
            if labels is not None:
                label = labels[idx]
                if label == 'PART_NUMBER':
                    # B-PART for first token, I-PART for rest
                    cell_labels = [self.label2id['B-PART']]
                    cell_labels.extend([self.label2id['I-PART']] * (len(cell_tokens) - 1))
                else:
                    # All 'O' for non-part-number cells
                    cell_labels = [self.label2id['O']] * len(cell_tokens)
                
                token_labels.extend(cell_labels)
                token_labels.append(self.label2id['O'])  # SEP token is 'O'

        # Convert tokens to IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        # Truncate if too long
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            if labels is not None:
                token_labels = token_labels[:self.max_length]

        # Pad if too short
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids.extend([self.tokenizer.pad_token_id] * padding_length)
            attention_mask.extend([0] * padding_length)
            if labels is not None:
                # Use -100 for padding tokens (ignored in loss calculation)
                token_labels.extend([-100] * padding_length)

        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

        if labels is not None:
            result['labels'] = token_labels

        return result

    def decode_predictions(
        self,
        input_ids: List[int],
        predictions: List[int],
        attention_mask: Optional[List[int]] = None
    ) -> List[str]:
        """
        Convert predicted label IDs back to label strings
        
        Args:
            input_ids: Token IDs
            predictions: Predicted label IDs
            attention_mask: Attention mask to filter padding
        
        Returns:
            List of predicted labels
        """
        if attention_mask is None:
            attention_mask = [1] * len(input_ids)

        labels = []
        for pred_id, mask in zip(predictions, attention_mask):
            if mask == 1:
                labels.append(self.id2label.get(pred_id, 'O'))
            else:
                labels.append('O')

        return labels
