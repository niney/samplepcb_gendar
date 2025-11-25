# Transformer 기반 NER 모델을 활용한 Part Number 추출 프로젝트 계획서
## (로컬 개발 버전)

## 1. 프로젝트 개요

### 목표
PCB BOM 데이터에서 헤더 정보 없이 Part Number를 자동으로 추출하는 NER(Named Entity Recognition) 모델을 로컬 환경에서 개발

### 핵심 과제
- Part Number가 랜덤한 열에 위치할 수 있음
- 헤더 없는 데이터에서 각 셀의 역할을 파악해야 함
- 로컬 PC/워크스테이션에서 학습 및 추론 가능해야 함
- 높은 정확도로 실용적 사용 가능해야 함

### 기대 성과
- 정확도: 95% 이상
- 처리 속도: 500-1,000 rows/초 (GPU 기준)
- False Positive Rate: 5% 이하
- 간편한 커맨드라인 실행

---

## 2. 기술 스택

### 핵심 프레임워크
```
- Python 3.9+
- venv (가상환경 - 필수)
- PyTorch 2.0+
- Transformers (Hugging Face) 4.30+
- FastAPI (API 서버)
- pandas, numpy (데이터 처리)
```

**⚠️ 중요: 이 프로젝트는 Python 표준 라이브러리인 venv를 사용하여 가상환경을 구축합니다.**

### 모델 선택지
1. **BERT-base** (추천: 시작용)
   - 빠른 학습, 적은 리소스
   - 정확도: 90-93%

2. **RoBERTa-base** (추천: 균형)
   - BERT 개선 버전
   - 정확도: 93-95%

3. **DeBERTa-v3-base** (추천: 최고 성능)
   - 최신 아키텍처
   - 정확도: 95-97%

### 로컬 개발 환경
- GPU: NVIDIA GTX 1060 이상 (학습용, CUDA 지원)
- CPU: 4 cores 이상
- RAM: 16GB 이상 (최소 8GB)
- Storage: 20GB 이상
- OS: Windows/Linux/MacOS

---

## 3. 데이터 준비 단계

### 3.1 데이터 수집
```
목표: 최소 1,000개의 라벨링된 BOM 행 데이터

데이터 소스:
- 기존 BOM 파일 (500개)
- 공개 데이터셋 (200개)
- 수동 생성 (300개)
```

### 3.2 데이터 포맷 정의
```json
{
  "row_id": "001",
  "cells": [
    "C29 C33 C34",
    "CC0402KRX7R9BB102",
    "CAP CER 1000PF 50V X7R 0402",
    "9",
    "Yageo",
    "1005"
  ],
  "labels": [
    "REFERENCE",
    "PART_NUMBER",
    "DESCRIPTION",
    "QUANTITY",
    "MANUFACTURER",
    "PACKAGE"
  ]
}
```

### 3.3 라벨링 전략

#### Option A: 셀 레벨 분류
각 셀을 독립적으로 분류
```
Input: "CC0402KRX7R9BB102"
Output: PART_NUMBER
```

#### Option B: 시퀀스 태깅 (추천)
전체 행을 하나의 시퀀스로 처리
```
Input: "[CLS] C29 C33 [SEP] CC0402KRX7R9BB102 [SEP] CAP CER [SEP] ..."
Output: [O, O, O, O, B-PART, I-PART, O, O, ...]
```

#### Option C: 행 레벨 멀티 레이블 분류
```
Input: 전체 행
Output: [0, 1, 0, 0, 0, 0]  # 두 번째 셀이 Part Number
```

### 3.4 데이터 증강
```python
증강 기법:
1. 컬럼 순서 셔플 (Part Number 위치 변경)
2. 동의어 치환 (CAP -> Capacitor)
3. 노이즈 추가 (오타, 공백 변경)
4. Back-translation (영어 -> 한국어 -> 영어)
5. Part Number 형식 변형 (하이픈 추가/제거)

목표: 1,000개 -> 5,000개로 증강
```

---

## 4. 모델 개발 단계

### 4.1 아키텍처 설계

#### 방법 1: Token Classification (추천)
```python
"""
[CLS] + Cell1 + [SEP] + Cell2 + [SEP] + ... + [SEP]
                  ↓
         Transformer Encoder
                  ↓
         Token Classification Head
                  ↓
    [O, O, O, B-PART, I-PART, O, ...]
"""

class BOMPartNumberNER(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids, attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)
        return logits
```

#### 방법 2: 셀 임베딩 + Classification
```python
"""
각 셀을 개별 인코딩 -> 분류

Cell1 -> [CLS] text [SEP] -> BERT -> Pool -> FC -> Class
Cell2 -> [CLS] text [SEP] -> BERT -> Pool -> FC -> Class
...
"""

class CellClassifier(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids, attention_mask)
        pooled = outputs.pooler_output
        logits = self.classifier(pooled)
        return logits
```

#### 방법 3: Row-level Multi-label Classification
```python
"""
전체 행 컨텍스트 활용

[CLS] Cell1 [SEP] Cell2 [SEP] ... [SEP]
              ↓
      Transformer Encoder
              ↓
         [SEP] 토큰들의 임베딩
              ↓
        Binary Classifier (각 셀마다)
              ↓
    [0, 1, 0, 0, 0]  # 두 번째 셀이 Part Number
"""
```

### 4.2 구현 코드 구조

```
project/
├── venv/                       # 가상환경 디렉토리 (필수, git에서 제외)
│
├── data/
│   ├── raw/                    # 원본 BOM 파일
│   ├── processed/              # 전처리된 데이터
│   ├── train.json             # 학습 데이터
│   ├── val.json               # 검증 데이터
│   └── test.json              # 테스트 데이터
│
├── src/
│   ├── data_preparation/
│   │   ├── labeling_tool.py   # 라벨링 도구
│   │   ├── data_loader.py     # 데이터 로더
│   │   ├── augmentation.py    # 데이터 증강
│   │   └── preprocessor.py    # 전처리
│   │
│   ├── model/
│   │   ├── ner_model.py       # NER 모델 정의
│   │   ├── cell_classifier.py # 셀 분류 모델
│   │   └── ensemble.py        # 앙상블 모델
│   │
│   ├── training/
│   │   ├── trainer.py         # 학습 로직
│   │   ├── config.py          # 하이퍼파라미터
│   │   └── callbacks.py       # 콜백 (체크포인트 등)
│   │
│   ├── evaluation/
│   │   ├── metrics.py         # 평가 지표
│   │   └── error_analysis.py  # 오류 분석
│   │
│   ├── inference/
│   │   ├── predictor.py       # 추론 엔진
│   │   └── postprocessing.py  # 후처리
│   │
│   └── api/
│       ├── server.py          # FastAPI 서버
│       └── schemas.py         # API 스키마
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
│
├── tests/
│   ├── test_model.py
│   ├── test_api.py
│   └── test_data.py
│
├── configs/
│   ├── bert_base.yaml
│   ├── roberta_base.yaml
│   └── deberta_v3.yaml
│
├── scripts/
│   ├── train.py              # 학습 실행 스크립트
│   ├── predict.py            # 추론 실행 스크립트
│   ├── evaluate.py           # 평가 스크립트
│   ├── interactive_label.py  # 대화형 라벨링 도구
│   ├── train_interactive.py  # 대화형 학습 마법사
│   ├── predict_interactive.py # 대화형 예측 도구
│   ├── evaluate_interactive.py # 대화형 모델 평가기
│   ├── explore_data.py       # 대화형 데이터 탐색기
│   ├── augment_interactive.py # 대화형 데이터 증강 도구
│   ├── config_manager.py     # 대화형 설정 관리자
│   ├── compare_models.py     # 대화형 모델 비교 도구
│   ├── batch_predict.py      # 대화형 배치 처리 모니터
│   ├── debug_console.py      # 대화형 디버깅 콘솔
│   ├── dashboard.py          # 대화형 프로젝트 대시보드
│   └── quality_check.py      # 대화형 품질 검증 도구
│
├── requirements.txt
├── setup.py
├── .gitignore                 # Git 제외 파일 목록
└── README.md
```

**⚠️ .gitignore 파일 예시:**
```
# 가상환경 (중요!)
venv/
env/
.venv/

# Python 캐시
__pycache__/
*.pyc
*.pyo
*.pyd

# 모델 체크포인트 (용량 큰 파일)
models/
*.pt
*.pth
*.bin

# 데이터 파일
data/raw/
data/processed/

# 로그
logs/
*.log

# IDE 설정
.vscode/
.idea/
*.swp

# OS 파일
.DS_Store
Thumbs.db
```

### 4.3 핵심 구현 파일

#### `src/data_preparation/preprocessor.py`
```python
class BOMDataPreprocessor:
    """BOM 데이터 전처리"""

    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def prepare_row_for_ner(self, cells, labels=None):
        """행을 NER 형식으로 변환"""
        # [CLS] cell1 [SEP] cell2 [SEP] ...
        tokens = ['[CLS]']
        token_labels = ['O']

        for idx, cell in enumerate(cells):
            cell_tokens = self.tokenizer.tokenize(str(cell))
            tokens.extend(cell_tokens)
            tokens.append('[SEP]')

            if labels:
                label = labels[idx]
                if label == 'PART_NUMBER':
                    cell_labels = ['B-PART'] + ['I-PART'] * (len(cell_tokens) - 1)
                else:
                    cell_labels = ['O'] * len(cell_tokens)
                token_labels.extend(cell_labels)
                token_labels.append('O')

        # 토큰화 및 인코딩
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        # 패딩
        padding_length = self.max_length - len(input_ids)
        input_ids += [0] * padding_length
        attention_mask += [0] * padding_length
        token_labels += ['O'] * padding_length

        return {
            'input_ids': input_ids[:self.max_length],
            'attention_mask': attention_mask[:self.max_length],
            'labels': token_labels[:self.max_length]
        }
```

#### `src/model/ner_model.py`
```python
from transformers import AutoModel, AutoConfig
import torch.nn as nn

class BOMPartNumberNER(nn.Module):
    """Part Number 추출용 NER 모델"""

    def __init__(self, model_name='bert-base-uncased', num_labels=3):
        super().__init__()
        self.num_labels = num_labels

        # Label: O, B-PART, I-PART
        self.label2id = {'O': 0, 'B-PART': 1, 'I-PART': 2}
        self.id2label = {v: k for k, v in self.label2id.items()}

        # Transformer 백본
        config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name, config=config)

        # Classification head
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        # CRF layer (선택적)
        # self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        # Transformer 인코딩
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)

        # 분류
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.num_labels),
                labels.view(-1)
            )

        return {'loss': loss, 'logits': logits}
```

#### `src/training/trainer.py`
```python
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset

class BOMDataset(Dataset):
    """BOM 데이터셋"""
    def __init__(self, data, preprocessor):
        self.data = data
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return self.preprocessor.prepare_row_for_ner(
            item['cells'],
            item['labels']
        )

def train_model(
    model,
    train_dataset,
    val_dataset,
    output_dir='./models/checkpoint'
):
    """모델 학습"""

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        learning_rate=2e-5,
        fp16=True,  # Mixed precision training
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return trainer
```

#### `src/evaluation/metrics.py`
```python
from seqeval.metrics import classification_report, f1_score
import numpy as np

def compute_metrics(pred):
    """평가 지표 계산"""
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)

    # ID를 라벨로 변환
    true_labels = [[id2label[l] for l in label if l != -100]
                   for label in labels]
    pred_labels = [[id2label[p] for (p, l) in zip(prediction, label) if l != -100]
                   for prediction, label in zip(predictions, labels)]

    # 평가
    f1 = f1_score(true_labels, pred_labels)

    # Part Number 추출 정확도 (실제 비즈니스 메트릭)
    part_number_accuracy = compute_part_number_accuracy(
        true_labels, pred_labels
    )

    return {
        'f1': f1,
        'part_number_accuracy': part_number_accuracy
    }

def compute_part_number_accuracy(true_labels, pred_labels):
    """Part Number 추출 정확도 계산"""
    correct = 0
    total = 0

    for true_seq, pred_seq in zip(true_labels, pred_labels):
        # Part Number 위치 추출
        true_part = extract_part_number_position(true_seq)
        pred_part = extract_part_number_position(pred_seq)

        total += 1
        if true_part == pred_part:
            correct += 1

    return correct / total if total > 0 else 0

def extract_part_number_position(label_sequence):
    """라벨 시퀀스에서 Part Number 위치 추출"""
    positions = []
    for idx, label in enumerate(label_sequence):
        if label.startswith('B-PART') or label.startswith('I-PART'):
            positions.append(idx)
    return positions
```

#### `src/inference/predictor.py`
```python
import torch

class PartNumberPredictor:
    """Part Number 추론 엔진"""

    def __init__(self, model_path, tokenizer_name='bert-base-uncased'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 모델 로드
        self.model = BOMPartNumberNER.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.preprocessor = BOMDataPreprocessor(self.tokenizer)

    def predict(self, row_cells):
        """단일 행에서 Part Number 추출"""
        # 전처리
        inputs = self.preprocessor.prepare_row_for_ner(row_cells)

        # Tensor 변환
        input_ids = torch.tensor([inputs['input_ids']]).to(self.device)
        attention_mask = torch.tensor([inputs['attention_mask']]).to(self.device)

        # 추론
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            predictions = torch.argmax(outputs['logits'], dim=2)

        # 후처리
        pred_labels = [self.model.id2label[p.item()]
                      for p in predictions[0]]

        # Part Number 추출
        part_number, confidence = self.extract_part_number_from_labels(
            row_cells, pred_labels
        )

        return {
            'part_number': part_number,
            'confidence': confidence,
            'cell_labels': pred_labels
        }

    def extract_part_number_from_labels(self, cells, labels):
        """라벨에서 실제 Part Number 텍스트 추출"""
        part_tokens = []
        confidence_scores = []

        current_cell_idx = 0
        for token, label in zip(cells, labels):
            if label.startswith('B-PART'):
                part_tokens = [token]
                current_cell_idx = cells.index(token)
            elif label.startswith('I-PART') and part_tokens:
                # 같은 셀 내에서만
                if cells.index(token) == current_cell_idx:
                    part_tokens.append(token)

        if part_tokens:
            part_number = ' '.join(part_tokens)
            confidence = len([l for l in labels if 'PART' in l]) / len(labels)
            return part_number, confidence

        return None, 0.0

    def batch_predict(self, batch_rows, batch_size=32):
        """배치 추론"""
        results = []
        for i in range(0, len(batch_rows), batch_size):
            batch = batch_rows[i:i+batch_size]
            batch_results = [self.predict(row) for row in batch]
            results.extend(batch_results)
        return results
```

---

## 5. 학습 및 평가 계획

### 5.1 학습 전략

#### 하이퍼파라미터
```yaml
# configs/bert_base.yaml
model:
  name: bert-base-uncased
  num_labels: 3
  dropout: 0.1

training:
  num_epochs: 10
  batch_size: 16
  learning_rate: 2e-5
  warmup_ratio: 0.1
  weight_decay: 0.01

  optimizer: adamw
  scheduler: linear

  gradient_accumulation_steps: 2
  max_grad_norm: 1.0

  fp16: true

data:
  max_length: 512
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
```

#### 학습 스케줄
```
Phase 1: Baseline (3일)
- BERT-base 모델
- 기본 하이퍼파라미터
- 목표: 85% 정확도

Phase 2: 최적화 (5일)
- 하이퍼파라미터 튜닝
- 데이터 증강 적용
- 목표: 90% 정확도

Phase 3: 고도화 (5일)
- RoBERTa/DeBERTa 실험
- 앙상블 기법 적용
- 목표: 95% 정확도
```

### 5.2 평가 지표

#### 모델 성능 지표
```python
평가 지표:
1. Token-level F1 Score
   - B-PART, I-PART에 대한 F1

2. Exact Match Accuracy
   - Part Number를 정확히 추출한 비율

3. Cell-level Accuracy
   - 올바른 셀을 선택한 비율

4. Confidence Score
   - 모델의 확신도

5. False Positive Rate
   - 잘못된 셀을 Part Number로 식별한 비율
```

#### 비즈니스 메트릭
```python
실제 사용 관점 평가:
1. Part Number 추출 성공률 (목표: 95%+)
2. 평균 처리 시간 (목표: <100ms/row)
3. 신뢰도 임계값별 정확도 곡선
4. 오류 타입 분석
   - Type I: Part Number를 못 찾음
   - Type II: 다른 셀을 Part Number로 오인
```

### 5.3 오류 분석 프로세스

```python
# src/evaluation/error_analysis.py

class ErrorAnalyzer:
    """오류 분석 도구"""

    def analyze_predictions(self, predictions, ground_truth):
        """예측 결과 분석"""
        errors = {
            'false_negative': [],  # Part Number를 못 찾음
            'false_positive': [],  # 다른 것을 Part Number로 오인
            'wrong_cell': [],      # Part Number는 찾았지만 다른 셀
        }

        for pred, true in zip(predictions, ground_truth):
            if pred['part_number'] is None and true['part_number'] is not None:
                errors['false_negative'].append((pred, true))
            elif pred['part_number'] != true['part_number']:
                if pred['part_number'] in true['cells']:
                    errors['wrong_cell'].append((pred, true))
                else:
                    errors['false_positive'].append((pred, true))

        return self.generate_report(errors)

    def generate_report(self, errors):
        """오류 리포트 생성"""
        report = f"""
        Error Analysis Report
        =====================

        False Negatives: {len(errors['false_negative'])}
        - Part Number를 찾지 못한 경우
        - 개선 방향: 데이터 증강, 임계값 조정

        False Positives: {len(errors['false_positive'])}
        - 잘못된 셀을 Part Number로 식별
        - 개선 방향: 규칙 기반 필터 추가

        Wrong Cell: {len(errors['wrong_cell'])}
        - Part Number 형식은 맞지만 다른 셀
        - 개선 방향: 컨텍스트 학습 강화

        Common Error Patterns:
        """

        # 오류 패턴 분석
        patterns = self.find_common_patterns(errors)
        for pattern, count in patterns.items():
            report += f"\n- {pattern}: {count} occurrences"

        return report
```

---

## 6. 로컬 개발 및 실행

### 6.1 환경 설정

#### 패키지 설치
```bash
# requirements.txt
torch>=2.0.0
transformers>=4.30.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
seqeval>=1.2.2
pyyaml>=6.0
tqdm>=4.65.0
jupyter>=1.0.0
matplotlib>=3.7.0
openpyxl>=3.1.0  # Excel 파일 지원

# 대화형 터미널 도구 (Interactive CLI Tools)
colorama>=0.4.6        # 터미널 컬러 출력
inquirer>=3.1.3        # 대화형 프롬프트
rich>=13.5.0           # 고급 터미널 UI
click>=8.1.0           # CLI 프레임워크

# 선택적 (FastAPI 로컬 서버 필요시)
fastapi>=0.100.0
uvicorn>=0.23.0
python-multipart>=0.0.6
```

#### 가상환경 설정 (필수)

**중요: 이 프로젝트는 반드시 venv 가상환경을 사용합니다.**

가상환경을 사용하는 이유:
- 시스템 Python과 프로젝트 의존성 분리
- 패키지 버전 충돌 방지
- 재현 가능한 개발 환경 구축

```bash
# 1. venv 가상환경 생성
python -m venv venv

# 2. 가상환경 활성화

# Windows (명령 프롬프트)
venv\Scripts\activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Linux/Mac
source venv/bin/activate

# 3. 가상환경 활성화 확인
# 프롬프트에 (venv) 표시가 나타나야 함
# 예: (venv) C:\project>

# 4. pip 업그레이드
python -m pip install --upgrade pip

# 5. 패키지 설치
pip install -r requirements.txt

# 6. 설치 확인
pip list
```

**주의사항:**
- 모든 개발 작업 전에 반드시 가상환경을 활성화하세요
- 터미널을 새로 열 때마다 가상환경을 재활성화해야 합니다
- 가상환경을 비활성화하려면: `deactivate`

#### CUDA 설정 (GPU 사용시)
```bash
# PyTorch with CUDA 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# GPU 확인
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 6.2 학습 실행

#### 커맨드라인 스크립트
```python
# scripts/train.py

import argparse
import yaml
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.model.ner_model import BOMPartNumberNER
from src.training.trainer import train_model, BOMDataset
from src.data_preparation.preprocessor import BOMDataPreprocessor
from transformers import AutoTokenizer
import json

def main():
    parser = argparse.ArgumentParser(description='Train Part Number NER Model')
    parser.add_argument('--config', type=str, default='configs/bert_base.yaml',
                       help='Config file path')
    parser.add_argument('--train_data', type=str, default='data/train.json',
                       help='Training data path')
    parser.add_argument('--val_data', type=str, default='data/val.json',
                       help='Validation data path')
    parser.add_argument('--output_dir', type=str, default='models/checkpoint',
                       help='Output directory for model checkpoints')
    args = parser.parse_args()

    # 설정 로드
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 데이터 로드
    with open(args.train_data, 'r') as f:
        train_data = json.load(f)
    with open(args.val_data, 'r') as f:
        val_data = json.load(f)

    # 토크나이저 및 전처리기
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    preprocessor = BOMDataPreprocessor(tokenizer)

    # 데이터셋 생성
    train_dataset = BOMDataset(train_data, preprocessor)
    val_dataset = BOMDataset(val_data, preprocessor)

    # 모델 생성
    model = BOMPartNumberNER(
        model_name=config['model']['name'],
        num_labels=config['model']['num_labels']
    )

    # 학습
    print(f"Starting training with config: {args.config}")
    trainer = train_model(model, train_dataset, val_dataset, args.output_dir)

    print(f"Training completed! Model saved to {args.output_dir}")

if __name__ == '__main__':
    main()
```

#### 실행 예시
```bash
# [필수] venv 가상환경 활성화
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# 기본 학습
python scripts/train.py

# 커스텀 설정으로 학습
python scripts/train.py --config configs/roberta_base.yaml --output_dir models/roberta_v1

# GPU 디바이스 지정
CUDA_VISIBLE_DEVICES=0 python scripts/train.py
```

### 6.3 추론 실행

#### 커맨드라인 추론 스크립트
```python
# scripts/predict.py

import argparse
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.inference.predictor import PartNumberPredictor

def main():
    parser = argparse.ArgumentParser(description='Predict Part Numbers from BOM file')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Input BOM file (CSV or Excel)')
    parser.add_argument('--output_file', type=str, default='output_predictions.csv',
                       help='Output file path')
    parser.add_argument('--confidence_threshold', type=float, default=0.7,
                       help='Minimum confidence threshold')
    args = parser.parse_args()

    # 모델 로드
    print(f"Loading model from {args.model_path}...")
    predictor = PartNumberPredictor(args.model_path)

    # 입력 파일 읽기
    print(f"Reading input file: {args.input_file}")
    if args.input_file.endswith('.csv'):
        df = pd.read_csv(args.input_file, header=None)
    else:
        df = pd.read_excel(args.input_file, header=None)

    # 추론
    print("Running predictions...")
    rows = df.values.tolist()
    results = predictor.batch_predict(rows)

    # 결과 저장
    df['predicted_part_number'] = [r['part_number'] for r in results]
    df['confidence'] = [r['confidence'] for r in results]
    df['needs_review'] = df['confidence'] < args.confidence_threshold

    df.to_csv(args.output_file, index=False)

    # 통계 출력
    total = len(results)
    high_conf = sum(1 for r in results if r['confidence'] >= args.confidence_threshold)
    print(f"\nPrediction completed!")
    print(f"Total rows: {total}")
    print(f"High confidence (>={args.confidence_threshold}): {high_conf} ({high_conf/total*100:.1f}%)")
    print(f"Needs review: {total - high_conf} ({(total-high_conf)/total*100:.1f}%)")
    print(f"Results saved to: {args.output_file}")

if __name__ == '__main__':
    main()
```

#### 실행 예시
```bash
# [필수] venv 가상환경 활성화
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# 기본 추론
python scripts/predict.py --model_path models/best_model --input_file data/new_bom.csv

# 신뢰도 임계값 조정
python scripts/predict.py \
    --model_path models/best_model \
    --input_file data/new_bom.xlsx \
    --output_file results/predictions.csv \
    --confidence_threshold 0.8
```

### 6.4 Jupyter Notebook 활용

#### 대화형 개발 및 분석
```python
# notebooks/01_data_exploration.ipynb

import pandas as pd
import json
from collections import Counter
import matplotlib.pyplot as plt

# 데이터 로드
with open('../data/train.json', 'r') as f:
    data = json.load(f)

# 통계 분석
print(f"Total samples: {len(data)}")

# Part Number 위치 분석
part_number_positions = []
for item in data:
    for idx, label in enumerate(item['labels']):
        if label == 'PART_NUMBER':
            part_number_positions.append(idx)
            break

# 시각화
plt.figure(figsize=(10, 5))
plt.hist(part_number_positions, bins=10)
plt.xlabel('Column Position')
plt.ylabel('Frequency')
plt.title('Part Number Position Distribution')
plt.show()

# Part Number 길이 분석
part_number_lengths = []
for item in data:
    for idx, label in enumerate(item['labels']):
        if label == 'PART_NUMBER':
            part_number_lengths.append(len(item['cells'][idx]))
            break

print(f"Average Part Number length: {sum(part_number_lengths)/len(part_number_lengths):.1f}")
```

```python
# notebooks/02_model_training.ipynb

from transformers import AutoTokenizer
from src.model.ner_model import BOMPartNumberNER
from src.training.trainer import train_model, BOMDataset
from src.data_preparation.preprocessor import BOMDataPreprocessor
import json

# 설정
MODEL_NAME = 'bert-base-uncased'
TRAIN_DATA = '../data/train.json'
VAL_DATA = '../data/val.json'

# 데이터 로드
with open(TRAIN_DATA) as f:
    train_data = json.load(f)
with open(VAL_DATA) as f:
    val_data = json.load(f)

# 전처리
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
preprocessor = BOMDataPreprocessor(tokenizer)

train_dataset = BOMDataset(train_data, preprocessor)
val_dataset = BOMDataset(val_data, preprocessor)

# 모델 생성 및 학습
model = BOMPartNumberNER(model_name=MODEL_NAME, num_labels=3)
trainer = train_model(model, train_dataset, val_dataset, output_dir='../models/checkpoint')

# 평가
eval_results = trainer.evaluate()
print(f"Validation Results: {eval_results}")
```

### 6.5 로컬 API 서버 (선택적)

간단한 로컬 테스트용 API 서버를 실행할 수 있습니다.

```python
# src/api/simple_server.py

from fastapi import FastAPI, UploadFile, File
import pandas as pd
from src.inference.predictor import PartNumberPredictor

app = FastAPI(title="Part Number Extractor - Local")

# 모델 로드
predictor = PartNumberPredictor('./models/best_model')

@app.post("/predict/file")
async def predict_file(file: UploadFile = File(...)):
    """파일 업로드 및 예측"""
    # 파일 읽기
    if file.filename.endswith('.csv'):
        df = pd.read_csv(file.file, header=None)
    else:
        df = pd.read_excel(file.file, header=None)

    # 예측
    rows = df.values.tolist()
    results = predictor.batch_predict(rows)

    return {
        'total_rows': len(results),
        'results': results
    }

@app.get("/")
def home():
    return {"message": "Part Number Extractor API - Running locally"}
```

#### 로컬 서버 실행
```bash
# [필수] venv 가상환경 활성화
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# 서버 시작
uvicorn src.api.simple_server:app --reload --port 8000

# 브라우저에서 접속
# http://localhost:8000/docs  (자동 생성된 API 문서)
```

### 6.6 로깅 및 디버깅

```python
# src/utils/logger.py

import logging
from datetime import datetime
from pathlib import Path

def setup_logger(name, log_dir='logs'):
    """로거 설정"""
    # 로그 디렉토리 생성
    Path(log_dir).mkdir(exist_ok=True)

    # 로거 생성
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 파일 핸들러
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(
        f'{log_dir}/{name}_{timestamp}.log'
    )
    file_handler.setLevel(logging.INFO)

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 포맷터
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# 사용 예시
logger = setup_logger('training')
logger.info('Starting training process...')
```

### 6.7 대화형 터미널 도구 (Interactive CLI Tools)

로컬 개발 효율성을 높이기 위한 대화형 터미널 도구들입니다.

#### 6.7.1 대화형 라벨링 도구

```python
# scripts/interactive_label.py

import sys
from pathlib import Path
import json
from colorama import init, Fore, Style
import readchar

init(autoreset=True)

class InteractiveLabelingTool:
    """대화형 BOM 라벨링 도구"""

    def __init__(self, input_file, output_file='data/labeled.json'):
        self.input_file = input_file
        self.output_file = output_file
        self.labels = ['REFERENCE', 'PART_NUMBER', 'DESCRIPTION',
                      'QUANTITY', 'MANUFACTURER', 'PACKAGE', 'OTHER']
        self.data = []
        self.current_index = 0

    def display_row(self, cells):
        """행 데이터 표시"""
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.YELLOW}Row {self.current_index + 1}")
        print(f"{Fore.CYAN}{'='*80}\n")

        for idx, cell in enumerate(cells):
            print(f"{Fore.GREEN}[{idx}] {Fore.WHITE}{cell}")

    def display_menu(self):
        """라벨 선택 메뉴"""
        print(f"\n{Fore.MAGENTA}Select labels for each cell:")
        for idx, label in enumerate(self.labels, 1):
            print(f"{Fore.YELLOW}{idx}. {label}")
        print(f"\n{Fore.CYAN}Controls:")
        print(f"  n: Next row | p: Previous row | s: Save | q: Quit")

    def label_row(self, cells):
        """단일 행 라벨링"""
        self.display_row(cells)
        self.display_menu()

        row_labels = []
        for cell_idx, cell in enumerate(cells):
            print(f"\n{Fore.GREEN}Cell [{cell_idx}]: {cell}")
            print(f"{Fore.YELLOW}Enter label number (1-{len(self.labels)}): ", end='')

            while True:
                choice = input().strip()
                if choice.isdigit() and 1 <= int(choice) <= len(self.labels):
                    row_labels.append(self.labels[int(choice) - 1])
                    break
                else:
                    print(f"{Fore.RED}Invalid choice. Try again: ", end='')

        return row_labels

    def run(self):
        """라벨링 세션 실행"""
        print(f"{Fore.CYAN}{'='*80}")
        print(f"{Fore.GREEN}BOM Interactive Labeling Tool")
        print(f"{Fore.CYAN}{'='*80}\n")

        # 입력 파일 로드
        import pandas as pd
        if self.input_file.endswith('.csv'):
            df = pd.read_csv(self.input_file, header=None)
        else:
            df = pd.read_excel(self.input_file, header=None)

        rows = df.values.tolist()

        while self.current_index < len(rows):
            cells = [str(cell) for cell in rows[self.current_index]]
            labels = self.label_row(cells)

            # 저장
            self.data.append({
                'row_id': f"{self.current_index:04d}",
                'cells': cells,
                'labels': labels
            })

            self.current_index += 1

            # 진행률 표시
            progress = (self.current_index / len(rows)) * 100
            print(f"\n{Fore.GREEN}Progress: {progress:.1f}% ({self.current_index}/{len(rows)})")

            # 주기적 저장
            if self.current_index % 10 == 0:
                self.save_data()
                print(f"{Fore.CYAN}Auto-saved!")

        # 최종 저장
        self.save_data()
        print(f"\n{Fore.GREEN}Labeling completed! Saved to {self.output_file}")

    def save_data(self):
        """데이터 저장"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Interactive BOM Labeling Tool')
    parser.add_argument('--input', type=str, required=True, help='Input BOM file')
    parser.add_argument('--output', type=str, default='data/labeled.json',
                       help='Output JSON file')
    args = parser.parse_args()

    tool = InteractiveLabelingTool(args.input, args.output)
    tool.run()

if __name__ == '__main__':
    main()
```

**사용 예시:**
```bash
# [필수] venv 가상환경 활성화
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# 기본 사용
python scripts/interactive_label.py --input data/raw/bom_sample.csv

# 출력 파일 지정
python scripts/interactive_label.py --input data/raw/bom_sample.xlsx --output data/train.json
```

#### 6.7.2 대화형 학습 마법사

```python
# scripts/train_interactive.py

import inquirer
from colorama import init, Fore, Style
import yaml
from pathlib import Path

init(autoreset=True)

class TrainingWizard:
    """대화형 학습 설정 마법사"""

    def __init__(self):
        self.config = {}

    def run(self):
        """마법사 실행"""
        print(f"{Fore.CYAN}{'='*80}")
        print(f"{Fore.GREEN}Model Training Wizard")
        print(f"{Fore.CYAN}{'='*80}\n")

        # 1. 모델 선택
        questions = [
            inquirer.List('model',
                message="Select base model",
                choices=[
                    ('BERT-base (Fast, 90-93% accuracy)', 'bert-base-uncased'),
                    ('RoBERTa-base (Balanced, 93-95% accuracy)', 'roberta-base'),
                    ('DeBERTa-v3-base (Best, 95-97% accuracy)', 'microsoft/deberta-v3-base'),
                ],
            ),
        ]
        model_answer = inquirer.prompt(questions)
        self.config['model_name'] = model_answer['model']

        # 2. 학습 파라미터
        questions = [
            inquirer.Text('epochs',
                message="Number of training epochs",
                default="10",
                validate=lambda _, x: x.isdigit() and int(x) > 0,
            ),
            inquirer.Text('batch_size',
                message="Batch size",
                default="16",
                validate=lambda _, x: x.isdigit() and int(x) > 0,
            ),
            inquirer.Text('learning_rate',
                message="Learning rate",
                default="2e-5",
            ),
            inquirer.Confirm('use_gpu',
                message="Use GPU for training?",
                default=True,
            ),
        ]
        train_answers = inquirer.prompt(questions)
        self.config.update(train_answers)

        # 3. 데이터 경로
        questions = [
            inquirer.Text('train_data',
                message="Training data path",
                default="data/train.json",
            ),
            inquirer.Text('val_data',
                message="Validation data path",
                default="data/val.json",
            ),
            inquirer.Text('output_dir',
                message="Model output directory",
                default="models/checkpoint",
            ),
        ]
        data_answers = inquirer.prompt(questions)
        self.config.update(data_answers)

        # 4. 고급 설정
        questions = [
            inquirer.Confirm('advanced',
                message="Configure advanced settings?",
                default=False,
            ),
        ]
        adv_answer = inquirer.prompt(questions)

        if adv_answer['advanced']:
            questions = [
                inquirer.Text('warmup_ratio',
                    message="Warmup ratio",
                    default="0.1",
                ),
                inquirer.Text('weight_decay',
                    message="Weight decay",
                    default="0.01",
                ),
                inquirer.Confirm('fp16',
                    message="Use mixed precision (FP16)?",
                    default=True,
                ),
            ]
            adv_answers = inquirer.prompt(questions)
            self.config.update(adv_answers)

        # 5. 설정 확인
        self.display_config()

        questions = [
            inquirer.Confirm('start',
                message="Start training with these settings?",
                default=True,
            ),
        ]
        confirm = inquirer.prompt(questions)

        if confirm['start']:
            self.save_config()
            self.start_training()
        else:
            print(f"{Fore.YELLOW}Training cancelled.")

    def display_config(self):
        """설정 표시"""
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.GREEN}Training Configuration")
        print(f"{Fore.CYAN}{'='*80}\n")

        for key, value in self.config.items():
            print(f"{Fore.YELLOW}{key:20s}: {Fore.WHITE}{value}")
        print()

    def save_config(self):
        """설정 저장"""
        config_dir = Path('configs/generated')
        config_dir.mkdir(parents=True, exist_ok=True)

        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config_file = config_dir / f'config_{timestamp}.yaml'

        with open(config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

        print(f"{Fore.GREEN}Configuration saved to {config_file}")

    def start_training(self):
        """학습 시작"""
        print(f"\n{Fore.GREEN}Starting training...")
        print(f"{Fore.YELLOW}Run: python scripts/train.py --config {self.config}")

def main():
    wizard = TrainingWizard()
    wizard.run()

if __name__ == '__main__':
    main()
```

**사용 예시:**
```bash
# [필수] venv 가상환경 활성화
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# 대화형 마법사 실행
python scripts/train_interactive.py

# 단계별 프롬프트에 따라 설정 입력
# 설정 자동 저장 및 학습 시작
```

#### 6.7.3 대화형 예측 도구

```python
# scripts/predict_interactive.py

from colorama import init, Fore, Style
import pandas as pd
from src.inference.predictor import PartNumberPredictor
import sys

init(autoreset=True)

class InteractivePredictor:
    """대화형 Part Number 예측 도구"""

    def __init__(self, model_path):
        print(f"{Fore.CYAN}Loading model from {model_path}...")
        self.predictor = PartNumberPredictor(model_path)
        print(f"{Fore.GREEN}Model loaded successfully!\n")

    def predict_single_row(self):
        """단일 행 예측 모드"""
        print(f"{Fore.CYAN}{'='*80}")
        print(f"{Fore.GREEN}Single Row Prediction Mode")
        print(f"{Fore.CYAN}{'='*80}\n")

        while True:
            print(f"{Fore.YELLOW}Enter BOM row (comma-separated cells):")
            print(f"{Fore.CYAN}Example: C29 C33,CC0402KRX7R9BB102,CAP CER 1000PF,9,Yageo")
            print(f"{Fore.MAGENTA}(Type 'q' to quit, 'f' for file mode)\n")

            user_input = input(f"{Fore.WHITE}> ").strip()

            if user_input.lower() == 'q':
                break
            elif user_input.lower() == 'f':
                self.predict_file()
                continue

            # 셀 분리
            cells = [cell.strip() for cell in user_input.split(',')]

            # 예측
            result = self.predictor.predict(cells)

            # 결과 표시
            self.display_result(cells, result)

    def display_result(self, cells, result):
        """예측 결과 표시"""
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.GREEN}Prediction Result")
        print(f"{Fore.CYAN}{'='*80}\n")

        # 각 셀과 라벨 표시
        for idx, cell in enumerate(cells):
            print(f"{Fore.WHITE}[{idx}] {cell}")

        print(f"\n{Fore.YELLOW}Predicted Part Number:")
        if result['part_number']:
            confidence_color = Fore.GREEN if result['confidence'] > 0.8 else Fore.YELLOW
            print(f"{Fore.GREEN}  → {result['part_number']}")
            print(f"{confidence_color}  Confidence: {result['confidence']:.2%}")
        else:
            print(f"{Fore.RED}  → Not found")

        print()

    def predict_file(self):
        """파일 예측 모드"""
        print(f"\n{Fore.YELLOW}Enter file path:")
        file_path = input(f"{Fore.WHITE}> ").strip()

        try:
            # 파일 읽기
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, header=None)
            else:
                df = pd.read_excel(file_path, header=None)

            rows = df.values.tolist()

            print(f"\n{Fore.CYAN}Processing {len(rows)} rows...")

            # 배치 예측
            results = self.predictor.batch_predict(rows)

            # 통계 표시
            self.display_batch_stats(results)

            # 저장 여부
            print(f"\n{Fore.YELLOW}Save results? (y/n):")
            save = input(f"{Fore.WHITE}> ").strip().lower()

            if save == 'y':
                output_file = file_path.replace('.csv', '_predicted.csv').replace('.xlsx', '_predicted.csv')
                df['predicted_part_number'] = [r['part_number'] for r in results]
                df['confidence'] = [r['confidence'] for r in results]
                df.to_csv(output_file, index=False)
                print(f"{Fore.GREEN}Results saved to {output_file}")

        except Exception as e:
            print(f"{Fore.RED}Error: {e}")

    def display_batch_stats(self, results):
        """배치 예측 통계"""
        total = len(results)
        found = sum(1 for r in results if r['part_number'] is not None)
        high_conf = sum(1 for r in results if r['confidence'] > 0.8)
        avg_conf = sum(r['confidence'] for r in results) / total

        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.GREEN}Batch Prediction Statistics")
        print(f"{Fore.CYAN}{'='*80}\n")
        print(f"{Fore.YELLOW}Total rows:          {Fore.WHITE}{total}")
        print(f"{Fore.YELLOW}Part numbers found:  {Fore.WHITE}{found} ({found/total:.1%})")
        print(f"{Fore.YELLOW}High confidence:     {Fore.WHITE}{high_conf} ({high_conf/total:.1%})")
        print(f"{Fore.YELLOW}Average confidence:  {Fore.WHITE}{avg_conf:.2%}")
        print()

    def run(self):
        """메인 실행"""
        print(f"{Fore.CYAN}{'='*80}")
        print(f"{Fore.GREEN}Interactive Part Number Predictor")
        print(f"{Fore.CYAN}{'='*80}\n")

        while True:
            print(f"{Fore.MAGENTA}Select mode:")
            print(f"{Fore.YELLOW}1. Single row prediction")
            print(f"{Fore.YELLOW}2. File prediction")
            print(f"{Fore.YELLOW}3. Quit")

            choice = input(f"\n{Fore.WHITE}> ").strip()

            if choice == '1':
                self.predict_single_row()
            elif choice == '2':
                self.predict_file()
            elif choice == '3':
                print(f"{Fore.GREEN}Goodbye!")
                break
            else:
                print(f"{Fore.RED}Invalid choice. Try again.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Interactive Part Number Predictor')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    args = parser.parse_args()

    predictor = InteractivePredictor(args.model_path)
    predictor.run()

if __name__ == '__main__':
    main()
```

**사용 예시:**
```bash
# [필수] venv 가상환경 활성화
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# 대화형 예측 시작
python scripts/predict_interactive.py --model_path models/best_model

# 모드 선택:
# 1. 단일 행 입력 -> 즉시 결과 확인
# 2. 파일 선택 -> 배치 처리 및 통계
```

#### 6.7.4 기타 대화형 도구

**대화형 데이터 탐색기** (`explore_data.py`)
- 라벨링된 데이터 통계 및 시각화
- Part Number 위치 분포 분석
- 데이터 품질 검사

**대화형 모델 평가기** (`evaluate_interactive.py`)
- 오류 케이스 단계별 검토
- 오류 유형별 필터링 (FN, FP, Wrong Cell)
- 개선 제안 생성

**대화형 프로젝트 대시보드** (`dashboard.py`)
- 프로젝트 전체 상태 모니터링
- 데이터/모델/실험 통합 관리
- 빠른 작업 실행 메뉴

#### 6.7.5 필요 패키지 추가

```bash
# requirements.txt에 추가
colorama>=0.4.6        # 터미널 컬러 출력
inquirer>=3.1.3        # 대화형 프롬프트
rich>=13.5.0           # 고급 터미널 UI
click>=8.1.0           # CLI 프레임워크
tqdm>=4.65.0           # 진행률 표시
```

#### 6.7.6 대화형 도구 우선순위

**Phase 1 (필수 - Week 1-2):**
1. ✅ 대화형 라벨링 도구 - 데이터 준비 핵심
2. ✅ 대화형 학습 마법사 - 초보자 친화적
3. ✅ 대화형 예측 도구 - 실용성 검증

**Phase 2 (유용 - Week 3-4):**
4. 대화형 모델 평가기 - 오류 분석
5. 대화형 배치 처리 모니터 - 실제 사용
6. 대화형 프로젝트 대시보드 - 전체 관리

**Phase 3 (고급 - Week 5+):**
7. 대화형 디버깅 콘솔 - 모델 분석
8. 대화형 모델 비교 도구 - 실험 관리
9. 대화형 데이터 증강 도구 - 데이터 확장
10. 대화형 품질 검증 도구 - QA

---

## 7. 타임라인 (로컬 개발 기준)

### Week 1-2: 환경 설정 및 데이터 준비
- [ ] 개발 환경 설정 (Python, CUDA, 패키지 설치)
- [ ] BOM 데이터 수집 (500+ 샘플)
- [ ] **대화형 라벨링 도구 개발** (`interactive_label.py`)
- [ ] 데이터 라벨링 (대화형 도구 활용)
- [ ] **대화형 데이터 탐색기 개발** (`explore_data.py`)
- [ ] 데이터 증강 파이프라인 구축
- [ ] Train/Val/Test 분리
- [ ] 데이터 탐색 노트북 작성

### Week 3: Baseline 모델 개발
- [ ] BERT-base 모델 구현
- [ ] 데이터 로더 및 전처리 파이프라인
- [ ] 학습 스크립트 작성 (scripts/train.py)
- [ ] **대화형 학습 마법사 개발** (`train_interactive.py`)
- [ ] 평가 메트릭 구현
- [ ] **대화형 모델 평가기 개발** (`evaluate_interactive.py`)
- [ ] Baseline 모델 학습 (대화형 마법사 활용)
- [ ] 초기 평가 및 오류 분석

### Week 4: 모델 최적화
- [ ] 하이퍼파라미터 튜닝 실험
- [ ] 다양한 모델 실험 (RoBERTa, DeBERTa)
- [ ] 데이터 증강 효과 검증
- [ ] 오류 패턴 분석 및 개선
- [ ] 목표 정확도 달성 (95%+)
- [ ] 최종 모델 선정

### Week 5: 추론 및 테스트
- [ ] 추론 스크립트 작성 (scripts/predict.py)
- [ ] **대화형 예측 도구 개발** (`predict_interactive.py`)
- [ ] **대화형 배치 처리 모니터 구현** (`batch_predict.py`)
- [ ] 배치 처리 최적화
- [ ] 다양한 BOM 파일로 테스트 (대화형 도구 활용)
- [ ] 신뢰도 임계값 조정
- [ ] **대화형 프로젝트 대시보드 구축** (`dashboard.py`)
- [ ] 성능 벤치마크 (속도, 정확도)
- [ ] 로컬 API 서버 구축 (선택적)
- [ ] 문서 작성 및 README 업데이트

---

## 8. 필요 리소스 (로컬 개발)

### 인력
- ML Engineer/개발자: 1명 (5주)
- 데이터 라벨러: 본인 또는 팀원 (2주, 파트타임)

### 하드웨어 (로컬 PC/워크스테이션)
```
최소 사양:
- CPU: Intel i5 이상 or AMD Ryzen 5 이상
- RAM: 16GB (최소 8GB)
- GPU: NVIDIA GTX 1060 6GB 이상 (CUDA 지원)
- Storage: SSD 20GB 이상

권장 사양:
- CPU: Intel i7/i9 or AMD Ryzen 7/9
- RAM: 32GB
- GPU: NVIDIA RTX 3060 12GB 이상 or RTX 4060
- Storage: NVMe SSD 50GB

학습 시간 예상:
- GTX 1060: 약 4-6시간/epoch (BERT-base)
- RTX 3060: 약 1-2시간/epoch
- RTX 4080: 약 30분-1시간/epoch
```

### 데이터
- 최소 1,000개의 라벨링된 BOM 행
- 데이터 라벨링: 자체 수행 (비용 $0)
- 로컬 저장소: 10-20GB

### 소프트웨어 (모두 무료)
```
필수:
- Python 3.9+ (무료)
- PyTorch (BSD 라이선스 - 무료)
- Transformers (Apache 2.0 - 무료)
- 사전학습 모델 (BERT/RoBERTa/DeBERTa) - 무료

선택적 (무료 버전):
- Label Studio Community Edition - 라벨링 도구
- Weights & Biases Free Tier - 실험 추적
- Jupyter Lab - 대화형 개발
- VS Code - 코드 에디터
```

### 예상 비용
```
총 비용: $0 (로컬 PC 활용시)

전기세 예상 (GPU 학습):
- 200W GPU × 50시간 = 10kWh
- 약 $1-2 (전기 요금)
```

---

## 9. 리스크 및 대응

### 리스크 1: 데이터 부족
**영향**: 모델 성능 저하
**대응**:
- 데이터 증강 기법 적극 활용
- Few-shot learning 기법 적용
- 합성 데이터 생성

### 리스크 2: 모델 정확도 미달
**영향**: 프로덕션 사용 불가
**대응**:
- 규칙 기반 후처리 추가
- 앙상블 방식 적용
- 사용자 피드백 루프 구축

### 리스크 3: 추론 속도 느림
**영향**: 사용자 경험 저하
**대응**:
- 모델 경량화 (DistilBERT 등)
- 배치 처리 최적화
- 모델 양자화 (INT8)
- GPU 활용 (가능한 경우)
- CPU 추론 최적화 (ONNX Runtime)

### 리스크 4: 새로운 Part Number 형식
**영향**: 기존 모델로 커버 안 됨
**대응**:
- 정기적인 모델 재학습
- 사용자 피드백 기반 업데이트
- 규칙 기반 fallback 구현

---

## 10. 성공 지표 (KPI)

### 모델 성능
- ✅ Part Number 추출 정확도: 95% 이상
- ✅ Token-level F1 Score: 0.93 이상
- ✅ False Positive Rate: 5% 이하

### 시스템 성능 (로컬 환경)
- ✅ 평균 추론 시간: 100ms 이하 (단일 행, GPU)
- ✅ 배치 처리량: 500-1,000 rows/초 (GPU 기준)
- ✅ CPU 추론 시간: 500ms 이하 (단일 행)

### 비즈니스 메트릭
- ✅ 수동 검증 필요 비율: 10% 이하 (낮은 신뢰도 케이스)
- ✅ 작업 시간 절감: 수작업 대비 80% 이상
- ✅ 사용성: 간단한 커맨드로 실행 가능

---

## 11. 다음 단계 (로컬 개발 로드맵)

### 즉시 시작 (Day 1)
1. 로컬 개발 환경 구축
   ```bash
   # 프로젝트 디렉토리 생성
   mkdir part-number-extractor
   cd part-number-extractor

   # [중요] venv 가상환경 생성 (필수)
   python -m venv venv

   # 가상환경 활성화
   # Windows (명령 프롬프트)
   venv\Scripts\activate
   # Windows (PowerShell)
   venv\Scripts\Activate.ps1
   # Linux/Mac
   source venv/bin/activate

   # 활성화 확인 - 프롬프트에 (venv) 표시 확인
   # 예: (venv) C:\part-number-extractor>

   # pip 업그레이드
   python -m pip install --upgrade pip

   # 기본 패키지 설치
   pip install torch transformers pandas jupyter

   # 전체 requirements.txt 작성 후 설치
   pip install -r requirements.txt
   ```

   **⚠️ 중요: 이후 모든 작업은 venv 가상환경이 활성화된 상태에서 진행해야 합니다.**

2. 프로젝트 구조 생성
   ```bash
   mkdir -p {data/{raw,processed},src/{model,training,inference},configs,scripts,notebooks,models,logs}
   ```

3. GPU 설정 확인
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
   ```

4. 데이터 수집 시작
   - 기존 BOM 파일 정리
   - 라벨링 계획 수립

### Phase 1 목표 (Week 2)
- ✅ 라벨링된 데이터 500개 확보
- ✅ 데이터 탐색 및 분석 (Jupyter Notebook)
- ✅ Baseline 모델 학습 완료
- ✅ 초기 정확도 평가 (85% 목표)

### Phase 2 목표 (Week 4)
- ✅ 목표 정확도 달성 (95%)
- ✅ 오류 분석 및 개선
- ✅ 하이퍼파라미터 최적화
- ✅ 최종 모델 선정 및 저장

### Phase 3 목표 (Week 5)
- ✅ 추론 스크립트 완성
- ✅ 다양한 BOM 파일로 테스트
- ✅ 성능 벤치마크
- ✅ 문서화 (README, 사용법)
- ✅ (선택적) 로컬 API 서버 구축

### 향후 발전 방향
1. **모델 개선**
   - Few-shot learning 적용
   - 다국어 지원 (한글 Part Number)
   - 더 큰 모델 실험 (Large 모델)

2. **기능 확장**
   - 다른 필드 추출 (Manufacturer, Package 등)
   - Excel 직접 편집 기능
   - GUI 도구 개발

3. **프로덕션 준비 (필요시)**
   - Docker 컨테이너화
   - 클라우드 배포 (AWS, GCP)
   - API 서버 고도화

---

## 참고 자료

### 논문 및 문서
- BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)
- RoBERTa: A Robustly Optimized BERT Pretraining Approach (Liu et al., 2019)
- DeBERTa: Decoding-enhanced BERT with Disentangled Attention (He et al., 2021)

### 코드 예제
- Hugging Face Transformers: https://github.com/huggingface/transformers
- Token Classification Tutorial: https://huggingface.co/docs/transformers/tasks/token_classification

### 도구
- Label Studio: https://labelstud.io/
- Weights & Biases: https://wandb.ai/
- FastAPI: https://fastapi.tiangolo.com/

---

## 문의 및 지원

프로젝트 관련 문의:
- 이메일: [project-lead@company.com]
- Slack: #part-number-extraction
- 위키: [프로젝트 위키 링크]
