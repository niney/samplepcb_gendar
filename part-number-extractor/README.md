# Part Number Extractor - Transformer-based NER for PCB BOM Data

Transformer 기반 NER(Named Entity Recognition) 모델을 활용하여 PCB BOM 데이터에서 Part Number를 자동으로 추출하는 로컬 개발 프로젝트입니다.

## 🎯 프로젝트 개요

### 핵심 기능
- ✅ 헤더 정보 없이 Part Number 자동 추출
- ✅ Part Number가 랜덤한 열에 위치해도 정확하게 인식
- ✅ 95% 이상의 높은 정확도 목표
- ✅ 로컬 PC에서 학습 및 추론 가능
- ✅ 사용자 친화적인 대화형 CLI 도구

### 기술 스택
- **Python 3.9+** with venv (가상환경)
- **PyTorch 2.0+** - Deep Learning 프레임워크
- **Transformers (Hugging Face)** - BERT/RoBERTa/DeBERTa 모델
- **FastAPI** - API 서버 (선택적)
- **pandas, numpy** - 데이터 처리

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 프로젝트 클론 또는 다운로드
cd part-number-extractor

# ⚠️ 중요: venv 가상환경 생성 (필수!)
python -m venv venv

# 가상환경 활성화
# Windows (명령 프롬프트)
venv\Scripts\activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1
# Linux/Mac
source venv/bin/activate

# (venv) 표시 확인 - 가상환경이 활성화되어 있어야 합니다

# pip 업그레이드
python -m pip install --upgrade pip

# 패키지 설치
pip install -r requirements.txt

# GPU 사용 시 (CUDA 11.8 기준)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. GPU 확인

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 📁 프로젝트 구조

```
part-number-extractor/
├── data/                    # 데이터 디렉토리
│   ├── raw/                 # 원본 BOM 파일
│   ├── processed/           # 전처리된 데이터
│   ├── train.json          # 학습 데이터
│   ├── val.json            # 검증 데이터
│   └── test.json           # 테스트 데이터
│
├── src/                     # 소스 코드
│   ├── data_preparation/   # 데이터 전처리
│   │   ├── preprocessor.py
│   │   ├── data_loader.py
│   │   └── augmentation.py
│   ├── model/              # 모델 정의
│   │   └── ner_model.py
│   ├── training/           # 학습 로직
│   │   └── trainer.py
│   ├── evaluation/         # 평가 메트릭
│   │   └── metrics.py
│   ├── inference/          # 추론 엔진
│   │   └── predictor.py
│   └── utils/              # 유틸리티
│       └── logger.py
│
├── scripts/                # 실행 스크립트
│   ├── train.py           # 학습 실행
│   ├── predict.py         # 추론 실행
│   ├── evaluate.py        # 평가 실행
│   ├── interactive_label.py      # 대화형 라벨링
│   ├── train_interactive.py      # 대화형 학습 마법사
│   └── predict_interactive.py    # 대화형 예측
│
├── configs/               # 설정 파일
│   ├── bert_base.yaml
│   ├── roberta_base.yaml
│   └── deberta_v3.yaml
│
├── models/                # 저장된 모델
├── logs/                  # 학습 로그
├── requirements.txt       # 패키지 의존성
└── README.md
```

---

## 📊 데이터 준비

### 데이터 포맷

라벨링된 데이터는 다음과 같은 JSON 형식을 사용합니다:

```json
[
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
]
```

### 대화형 라벨링 도구 사용

```bash
# 가상환경 활성화 필수!
python scripts/interactive_label.py --input data/raw/bom_sample.csv --output data/labeled.json
```

이 도구는:
- 각 셀에 대해 대화형으로 라벨 선택
- 진행 상황 자동 저장 (10개 행마다)
- 언제든지 중단 후 재개 가능

---

## 🎓 모델 학습

### 방법 1: 대화형 마법사 사용 (추천)

```bash
# 가상환경 활성화 필수!
python scripts/train_interactive.py
```

단계별 프롬프트를 따라 설정하면 자동으로 학습이 시작됩니다.

### 방법 2: 커맨드라인 직접 실행

```bash
# 기본 학습 (BERT-base)
python scripts/train.py \
    --config configs/bert_base.yaml \
    --train_data data/train.json \
    --val_data data/val.json \
    --output_dir models/bert_checkpoint

# RoBERTa 모델로 학습
python scripts/train.py \
    --config configs/roberta_base.yaml \
    --train_data data/train.json \
    --val_data data/val.json \
    --output_dir models/roberta_checkpoint

# 커스텀 설정
python scripts/train.py \
    --model_name bert-base-uncased \
    --epochs 15 \
    --batch_size 8 \
    --train_data data/train.json \
    --val_data data/val.json \
    --output_dir models/custom_model
```

---

## 🔮 Part Number 추출 (추론)

### 방법 1: 대화형 예측 도구 (추천)

```bash
# 가상환경 활성화 필수!
python scripts/predict_interactive.py --model_path models/bert_checkpoint/final_model
```

이 도구는:
- 단일 행 입력 모드: 즉시 결과 확인
- 파일 처리 모드: 배치 처리 및 통계 제공

### 방법 2: 커맨드라인 직접 실행

```bash
# CSV 파일 처리
python scripts/predict.py \
    --model_path models/bert_checkpoint/final_model \
    --input_file data/new_bom.csv \
    --output_file results/predictions.csv \
    --confidence_threshold 0.8

# Excel 파일 처리
python scripts/predict.py \
    --model_path models/bert_checkpoint/final_model \
    --input_file data/new_bom.xlsx \
    --output_file results/predictions.xlsx \
    --confidence_threshold 0.7
```

### 출력 결과

결과 파일에는 다음 정보가 포함됩니다:
- `predicted_part_number`: 추출된 Part Number
- `confidence`: 신뢰도 점수 (0-1)
- `cell_index`: Part Number가 있는 열 인덱스
- `needs_review`: 신뢰도가 임계값 미만인 경우 True

---

## 📈 모델 평가

```bash
# 가상환경 활성화 필수!
python scripts/evaluate.py \
    --model_path models/bert_checkpoint/final_model \
    --test_data data/test.json \
    --output_dir evaluation_results
```

평가 지표:
- **F1 Score**: 토큰 레벨 NER 성능
- **Precision & Recall**: 정밀도와 재현율
- **Part Number Accuracy**: 실제 Part Number 추출 정확도

---

## 🛠️ 모델 선택 가이드

| 모델 | 속도 | 정확도 | GPU 메모리 | 추천 용도 |
|------|------|--------|-----------|----------|
| **BERT-base** | ⚡⚡⚡ | 90-93% | 4GB | 빠른 프로토타입 |
| **RoBERTa-base** | ⚡⚡ | 93-95% | 6GB | 균형잡힌 성능 |
| **DeBERTa-v3-base** | ⚡ | 95-97% | 8GB | 최고 정확도 |

---

## 💻 하드웨어 요구사항

### 최소 사양
- CPU: Intel i5 이상
- RAM: 16GB (최소 8GB)
- GPU: NVIDIA GTX 1060 6GB 이상 (CUDA 지원)
- Storage: SSD 20GB

### 권장 사양
- CPU: Intel i7/i9 or AMD Ryzen 7/9
- RAM: 32GB
- GPU: NVIDIA RTX 3060 12GB 이상
- Storage: NVMe SSD 50GB

---

## 📝 데이터 증강

```python
from src.data_preparation.augmentation import BOMDataAugmenter
from src.data_preparation.data_loader import load_bom_data, save_bom_data

# 데이터 로드
data = load_bom_data('data/train.json')

# 증강기 초기화
augmenter = BOMDataAugmenter()

# 데이터 증강 (1000개 -> 5000개)
augmented_data = augmenter.augment_dataset(
    data,
    target_size=5000,
    methods=['shuffle', 'noise', 'format']
)

# 저장
save_bom_data(augmented_data, 'data/train_augmented.json')
```

---

## 🔧 고급 사용법

### 커스텀 모델 학습

```python
from transformers import AutoTokenizer
from src.model.ner_model import create_model
from src.training.trainer import train_model
from src.data_preparation.data_loader import BOMDataset, load_bom_data
from src.data_preparation.preprocessor import BOMDataPreprocessor
from src.evaluation.metrics import compute_metrics

# 데이터 로드
train_data = load_bom_data('data/train.json')
val_data = load_bom_data('data/val.json')

# 토크나이저 및 전처리
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
preprocessor = BOMDataPreprocessor(tokenizer)

# 데이터셋 생성
train_dataset = BOMDataset(train_data, preprocessor)
val_dataset = BOMDataset(val_data, preprocessor)

# 모델 생성
model = create_model('roberta-base', num_labels=3)

# 학습
trainer = train_model(
    model,
    train_dataset,
    val_dataset,
    compute_metrics,
    output_dir='models/my_model'
)
```

---

## 🐛 트러블슈팅

### 가상환경 문제

```bash
# 가상환경이 활성화되지 않은 경우
# 프롬프트에 (venv)가 표시되어야 합니다

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 가상환경 비활성화
deactivate
```

### CUDA 에러

```bash
# CUDA 버전 확인
python -c "import torch; print(torch.version.cuda)"

# CUDA 사용 불가 시 CPU로 학습
# configs/*.yaml 파일에서 fp16: false 로 설정
```

### 메모리 부족

```bash
# batch_size 줄이기
python scripts/train.py --batch_size 8

# gradient_accumulation 사용 (설정 파일에서)
gradient_accumulation_steps: 4
```

---

## 📊 성공 지표 (KPI)

### 모델 성능
- ✅ Part Number 추출 정확도: **95% 이상**
- ✅ Token-level F1 Score: **0.93 이상**
- ✅ False Positive Rate: **5% 이하**

### 시스템 성능
- ✅ GPU 추론 시간: **<100ms** (단일 행)
- ✅ CPU 추론 시간: **<500ms** (단일 행)
- ✅ 배치 처리량: **500-1,000 rows/초** (GPU)

---

## 📚 추가 문서

- [데이터 준비 가이드](docs/data_preparation.md)
- [모델 학습 가이드](docs/training_guide.md)
- [API 문서](docs/api_documentation.md)

---

## 🤝 기여

이 프로젝트는 로컬 개발용으로 설계되었습니다. 개선 사항이나 버그 리포트는 이슈로 등록해주세요.

---

## 📄 라이선스

MIT License

---

## 🙏 감사의 말

- Hugging Face Transformers
- PyTorch
- seqeval

---

## 📧 문의

프로젝트 관련 문의: [your-email@example.com]

---

**⚠️ 중요한 주의사항:**

1. **가상환경 사용은 필수입니다!** 모든 작업 전에 `venv` 활성화를 확인하세요.
2. 터미널을 새로 열 때마다 가상환경을 재활성화해야 합니다.
3. 대규모 모델 학습 시 충분한 디스크 공간을 확보하세요.
4. GPU 메모리가 부족할 경우 batch_size를 줄이세요.

**Happy Extracting! 🚀**
