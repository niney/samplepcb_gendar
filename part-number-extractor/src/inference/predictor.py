"""
Part Number Predictor - Inference Engine
"""

import torch
from transformers import AutoTokenizer
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.model.ner_model import BOMPartNumberNER, create_model
from src.data_preparation.preprocessor import BOMDataPreprocessor


class PartNumberPredictor:
    """Part Number 추론 엔진"""
    
    # Model type to pretrained model name mapping
    MODEL_TYPE_MAPPING = {
        'deberta-v2': 'microsoft/deberta-v3-base',
        'deberta': 'microsoft/deberta-base',
        'roberta': 'roberta-base',
        'bert': 'bert-base-uncased',
    }

    def __init__(
        self,
        model_path: str,
        tokenizer_name: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Args:
            model_path: Path to trained model, or "__demo__" for demo mode
            tokenizer_name: Name of tokenizer (auto-detected from model config if not provided)
            device: Device to run inference on (cuda/cpu)
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Loading model on {self.device}...")

        # Demo mode - create untrained model for testing
        if model_path == "__demo__":
            effective_tokenizer = tokenizer_name or 'bert-base-uncased'
            print("Creating demo model (untrained - for testing only)...")
            self.tokenizer = AutoTokenizer.from_pretrained(effective_tokenizer)
            self.model = create_model(model_name=effective_tokenizer)
            self.model.to(self.device)
            self.model.eval()
            self.preprocessor = BOMDataPreprocessor(self.tokenizer)
            print("Demo model loaded successfully!")
            return

        # Check if model_path is a local directory
        model_path_obj = Path(model_path)
        is_local_path = model_path_obj.exists() and model_path_obj.is_dir()

        # Load model
        if is_local_path:
            # Load from local directory
            config_file = model_path_obj / 'config.json'
            model_file = model_path_obj / 'pytorch_model.bin'
            safetensors_file = model_path_obj / 'model.safetensors'
            
            if config_file.exists() and (model_file.exists() or safetensors_file.exists()):
                # Load config to detect model type
                import json
                with open(config_file) as f:
                    config_data = json.load(f)
                
                # Auto-detect model name from config
                model_type = config_data.get('model_type', 'bert')
                detected_model_name = self.MODEL_TYPE_MAPPING.get(model_type, 'bert-base-uncased')
                
                # Use provided tokenizer_name or detected model name
                effective_model_name = tokenizer_name or detected_model_name
                print(f"Detected model type: {model_type}, using: {effective_model_name}")
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(effective_model_name)
                
                # Create model with same architecture
                self.model = create_model(model_name=effective_model_name, num_labels=3)
                
                # Load state dict
                if safetensors_file.exists():
                    from safetensors.torch import load_file
                    state_dict = load_file(safetensors_file)
                    self.model.load_state_dict(state_dict)
                    print(f"Loaded model weights from {safetensors_file}")
                elif model_file.exists():
                    state_dict = torch.load(model_file, map_location=self.device, weights_only=True)
                    self.model.load_state_dict(state_dict)
                    print(f"Loaded model weights from {model_file}")
            else:
                raise FileNotFoundError(
                    f"Model files not found in {model_path}. "
                    f"Expected config.json and pytorch_model.bin or model.safetensors"
                )
        else:
            # Try loading from Hugging Face Hub
            effective_tokenizer = tokenizer_name or 'bert-base-uncased'
            self.tokenizer = AutoTokenizer.from_pretrained(effective_tokenizer)
            try:
                self.model = BOMPartNumberNER.from_pretrained(model_path)
            except Exception as e:
                raise ValueError(
                    f"Could not load model from '{model_path}'. "
                    f"Please provide a valid local path or Hugging Face model ID. Error: {e}"
                )

        self.model.to(self.device)
        self.model.eval()

        # Initialize preprocessor
        self.preprocessor = BOMDataPreprocessor(self.tokenizer)

        print("Model loaded successfully!")

    def predict(self, row_cells: List[str]) -> Dict:
        """
        단일 행에서 Part Number 추출
        
        Args:
            row_cells: List of cell values in the row
        
        Returns:
            Dictionary with:
                - part_number: Extracted Part Number text
                - confidence: Confidence score
                - cell_index: Index of cell containing Part Number
                - cell_labels: Label predictions for each cell
        """
        # Preprocess
        inputs = self.preprocessor.prepare_row_for_ner(row_cells)

        # Convert to tensors
        input_ids = torch.tensor([inputs['input_ids']]).to(self.device)
        attention_mask = torch.tensor([inputs['attention_mask']]).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            predictions = torch.argmax(logits, dim=2)

        # Convert predictions to labels
        pred_labels = [
            self.model.id2label[p.item()]
            for p in predictions[0]
            if inputs['attention_mask'][predictions[0].tolist().index(p.item())] == 1
        ]

        # Calculate confidence scores
        probs = torch.softmax(logits, dim=2)[0]
        confidences = torch.max(probs, dim=1)[0].cpu().numpy()

        # Extract Part Number
        part_number, confidence, cell_index = self.extract_part_number_from_predictions(
            row_cells, pred_labels, confidences, inputs
        )

        return {
            'part_number': part_number,
            'confidence': float(confidence),
            'cell_index': cell_index,
            'cell_labels': pred_labels,
        }

    def extract_part_number_from_predictions(
        self,
        cells: List[str],
        pred_labels: List[str],
        confidences: List[float],
        inputs: Dict
    ) -> Tuple[Optional[str], float, Optional[int]]:
        """
        라벨 예측에서 실제 Part Number 텍스트 추출
        
        Args:
            cells: Original cell values
            pred_labels: Predicted labels for each token
            confidences: Confidence scores for each token
            inputs: Preprocessed inputs
        
        Returns:
            Tuple of (part_number, confidence, cell_index)
        """
        # Find B-PART or I-PART labels
        part_indices = [i for i, label in enumerate(pred_labels) if 'PART' in label]

        if not part_indices:
            return None, 0.0, None

        # Calculate average confidence for Part Number tokens
        avg_confidence = sum(confidences[i] for i in part_indices) / len(part_indices)

        # Map tokens back to cells
        # This is simplified - assumes tokens map sequentially to cells
        # In production, you'd need more sophisticated token-to-cell mapping

        # Find which cell contains the most PART tokens
        cell_part_counts = [0] * len(cells)
        
        # Reconstruct token-to-cell mapping
        token_idx = 1  # Skip [CLS]
        for cell_idx in range(len(cells)):
            cell_tokens = self.tokenizer.tokenize(str(cells[cell_idx]))
            cell_token_count = len(cell_tokens)
            
            # Count PART labels in this cell's tokens
            for _ in range(cell_token_count):
                if token_idx < len(pred_labels) and 'PART' in pred_labels[token_idx]:
                    cell_part_counts[cell_idx] += 1
                token_idx += 1
            
            token_idx += 1  # Skip [SEP]

        # Get cell with most PART tokens
        if max(cell_part_counts) > 0:
            cell_index = cell_part_counts.index(max(cell_part_counts))
            part_number = str(cells[cell_index])
            return part_number, avg_confidence, cell_index

        return None, avg_confidence, None

    def batch_predict(
        self,
        batch_rows: List[List[str]],
        batch_size: int = 32
    ) -> List[Dict]:
        """
        배치 추론
        
        Args:
            batch_rows: List of rows to predict
            batch_size: Batch size for processing
        
        Returns:
            List of prediction results
        """
        results = []
        
        for i in range(0, len(batch_rows), batch_size):
            batch = batch_rows[i:i+batch_size]
            
            # Process batch
            for row in batch:
                result = self.predict(row)
                results.append(result)

        return results

    def predict_with_threshold(
        self,
        row_cells: List[str],
        confidence_threshold: float = 0.7
    ) -> Dict:
        """
        신뢰도 임계값을 적용한 예측
        
        Args:
            row_cells: List of cell values
            confidence_threshold: Minimum confidence threshold
        
        Returns:
            Prediction result with needs_review flag
        """
        result = self.predict(row_cells)
        result['needs_review'] = result['confidence'] < confidence_threshold
        
        return result
