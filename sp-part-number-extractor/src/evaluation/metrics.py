"""
Evaluation metrics for BOM NER model
"""

import numpy as np
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from typing import Dict, List, Tuple


def compute_metrics(pred) -> Dict:
    """
    Compute evaluation metrics for NER model
    
    Args:
        pred: Predictions from model (EvalPrediction object)
    
    Returns:
        Dictionary with metrics
    """
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)

    # Label mappings
    id2label = {0: 'O', 1: 'B-PART', 2: 'I-PART'}

    # Convert IDs to labels, removing special tokens
    true_labels = [
        [id2label[l] for l in label if l != -100]
        for label in labels
    ]
    pred_labels = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Calculate metrics using seqeval
    f1 = f1_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)

    # Calculate Part Number extraction accuracy (business metric)
    part_number_accuracy = compute_part_number_accuracy(true_labels, pred_labels)

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'part_number_accuracy': part_number_accuracy,
    }


def compute_part_number_accuracy(
    true_labels: List[List[str]],
    pred_labels: List[List[str]]
) -> float:
    """
    Part Number 추출 정확도 계산
    실제 비즈니스 메트릭 - Part Number를 정확히 추출했는지 확인
    
    Args:
        true_labels: Ground truth label sequences
        pred_labels: Predicted label sequences
    
    Returns:
        Accuracy score (0-1)
    """
    correct = 0
    total = 0

    for true_seq, pred_seq in zip(true_labels, pred_labels):
        # Extract Part Number positions
        true_part_positions = extract_part_number_positions(true_seq)
        pred_part_positions = extract_part_number_positions(pred_seq)

        total += 1
        if true_part_positions == pred_part_positions:
            correct += 1

    return correct / total if total > 0 else 0.0


def extract_part_number_positions(label_sequence: List[str]) -> List[int]:
    """
    라벨 시퀀스에서 Part Number 위치 추출
    
    Args:
        label_sequence: List of labels
    
    Returns:
        List of token positions that are part of Part Number
    """
    positions = []
    for idx, label in enumerate(label_sequence):
        if label in ['B-PART', 'I-PART']:
            positions.append(idx)
    return positions


def detailed_classification_report(
    true_labels: List[List[str]],
    pred_labels: List[List[str]]
) -> str:
    """
    Generate detailed classification report
    
    Args:
        true_labels: Ground truth label sequences
        pred_labels: Predicted label sequences
    
    Returns:
        Detailed report string
    """
    report = classification_report(true_labels, pred_labels, digits=4)
    return report


class ErrorAnalyzer:
    """오류 분석 도구"""

    def __init__(self, id2label: Dict = None):
        """
        Args:
            id2label: Mapping from label IDs to label strings
        """
        self.id2label = id2label or {0: 'O', 1: 'B-PART', 2: 'I-PART'}

    def analyze_predictions(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        input_texts: List[str] = None
    ) -> Dict:
        """
        예측 결과 분석
        
        Args:
            predictions: Predicted label IDs
            labels: Ground truth label IDs
            input_texts: Original input texts (optional)
        
        Returns:
            Dictionary with error analysis
        """
        predictions = np.argmax(predictions, axis=2)

        # Convert to label strings
        true_labels = [
            [self.id2label[l] for l in label if l != -100]
            for label in labels
        ]
        pred_labels = [
            [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        # Categorize errors
        errors = {
            'false_negative': [],  # Part Number를 못 찾음
            'false_positive': [],  # 다른 것을 Part Number로 오인
            'partial_match': [],   # Part Number를 부분적으로만 찾음
        }

        for idx, (true_seq, pred_seq) in enumerate(zip(true_labels, pred_labels)):
            true_positions = set(extract_part_number_positions(true_seq))
            pred_positions = set(extract_part_number_positions(pred_seq))

            if true_positions and not pred_positions:
                # False Negative: missed Part Number
                error_info = {
                    'index': idx,
                    'true_labels': true_seq,
                    'pred_labels': pred_seq,
                }
                if input_texts and idx < len(input_texts):
                    error_info['text'] = input_texts[idx]
                errors['false_negative'].append(error_info)

            elif pred_positions and not true_positions:
                # False Positive: predicted Part Number where there is none
                error_info = {
                    'index': idx,
                    'true_labels': true_seq,
                    'pred_labels': pred_seq,
                }
                if input_texts and idx < len(input_texts):
                    error_info['text'] = input_texts[idx]
                errors['false_positive'].append(error_info)

            elif true_positions != pred_positions and (true_positions and pred_positions):
                # Partial match: found Part Number but not exactly
                error_info = {
                    'index': idx,
                    'true_labels': true_seq,
                    'pred_labels': pred_seq,
                    'true_positions': sorted(list(true_positions)),
                    'pred_positions': sorted(list(pred_positions)),
                }
                if input_texts and idx < len(input_texts):
                    error_info['text'] = input_texts[idx]
                errors['partial_match'].append(error_info)

        return errors

    def generate_report(self, errors: Dict) -> str:
        """
        오류 리포트 생성
        
        Args:
            errors: Error analysis dictionary
        
        Returns:
            Report string
        """
        report = f"""
Error Analysis Report
=====================

False Negatives: {len(errors['false_negative'])}
- Part Number를 찾지 못한 경우
- 개선 방향: 데이터 증강, 임계값 조정

False Positives: {len(errors['false_positive'])}
- 잘못된 셀을 Part Number로 식별
- 개선 방향: 규칙 기반 필터 추가

Partial Matches: {len(errors['partial_match'])}
- Part Number를 부분적으로만 식별
- 개선 방향: 시퀀스 태깅 정확도 향상

Total Errors: {sum(len(v) for v in errors.values())}
"""
        return report
