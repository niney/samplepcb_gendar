# Part Number Extractor - 빠른 시작 가이드

## 1단계: 환경 설정 (5분)

### venv 가상환경 생성 및 활성화

```bash
# 프로젝트 디렉토리로 이동
cd part-number-extractor

# venv 가상환경 생성
python -m venv venv

# 가상환경 활성화
# Windows (명령 프롬프트)
venv\Scripts\activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Linux/Mac
source venv/bin/activate

# ✓ 프롬프트에 (venv) 표시 확인
# 예: (venv) C:\part-number-extractor>
```

### 패키지 설치

```bash
# pip 업그레이드
python -m pip install --upgrade pip

# 필수 패키지 설치
pip install -r requirements.txt

# GPU 사용 시 (CUDA 11.8 기준)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# GPU 확인
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 2단계: 샘플 데이터 생성 (1분)

```bash
# 샘플 데이터 생성
python examples/create_sample_data.py

# 확인
ls data/
# 출력: sample_train.json
```

---

## 3단계: 데이터 라벨링 (실제 프로젝트 시작)

### 옵션 A: 대화형 라벨링 도구 (추천)

```bash
# BOM CSV/Excel 파일 준비 (data/raw/ 폴더에 배치)

# 대화형 라벨링 시작
python scripts/interactive_label.py \
    --input data/raw/your_bom.csv \
    --output data/labeled.json

# 단계별 프롬프트를 따라 각 셀에 라벨 지정
# - 1: REFERENCE
# - 2: PART_NUMBER  ← 중요!
# - 3: DESCRIPTION
# - 4: QUANTITY
# - 5: MANUFACTURER
# - 6: PACKAGE
# - 7: OTHER
```

### 옵션 B: 샘플 데이터로 테스트

```bash
# 샘플 데이터를 train/val/test로 분할
python scripts/split_data.py \
    --input data/sample_train.json \
    --output_dir data

# 확인
ls data/
# 출력: train.json, val.json, test.json
```

---

## 4단계: 모델 학습

### 방법 1: 대화형 마법사 (초보자 추천)

```bash
python scripts/train_interactive.py
```

프롬프트에서:
1. 모델 선택: `BERT-base` (빠른 학습)
2. Epochs: `10`
3. Batch size: `16` (메모리 부족 시 `8`)
4. Learning rate: `2e-5`
5. 데이터 경로 확인
6. 시작!

### 방법 2: 커맨드라인

```bash
# BERT-base로 빠른 학습
python scripts/train.py \
    --config configs/bert_base.yaml \
    --epochs 30 \
    --batch_size 4
```

학습 진행 상황:
- `logs/` 폴더에 학습 로그 생성
- `models/checkpoint/` 에 체크포인트 저장
- 학습 완료 후 `final_model/` 생성

---

## 5단계: Part Number 추출

### 방법 1: 대화형 예측 (사용자 친화적)

```bash
python scripts/predict_interactive.py \
    --model_path models/checkpoint/final_model

# 모드 선택:
# 1. 단일 행 입력 - 즉시 테스트
# 2. 파일 처리 - 실제 BOM 파일 처리
```

### 방법 2: 커맨드라인 (배치 처리)

```bash
python scripts/predict.py \
    --model_path models/my_first_model/final_model \
    --input_file data/raw/new_bom.csv \
    --output_file results/predictions.csv \
    --confidence_threshold 0.7
```

결과 확인:
- `results/predictions.csv` 열기
- 열: `predicted_part_number`, `confidence`, `cell_index`, `needs_review`

---

## 6단계: 모델 평가

```bash
python scripts/evaluate.py \
    --model_path models/my_first_model/final_model \
    --test_data data/test.json \
    --output_dir evaluation_results
```

평가 지표:
- **F1 Score**: NER 토큰 레벨 성능
- **Part Number Accuracy**: 실제 추출 정확도
- **Precision/Recall**: 정밀도/재현율

---

## 일반적인 워크플로우

### 신규 프로젝트 시작

```bash
# 1. 가상환경 활성화
source venv/bin/activate  # or venv\Scripts\activate

# 2. BOM 파일 준비
# data/raw/ 폴더에 배치

# 3. 라벨링
python scripts/interactive_label.py --input data/raw/bom.csv --output data/labeled.json

# 4. 데이터 분할
python scripts/split_data.py --input data/labeled.json

# 5. 학습
python scripts/train_interactive.py

# 6. 예측
python scripts/predict_interactive.py --model_path models/.../final_model
```

### 기존 프로젝트 재개

```bash
# 가상환경 재활성화 (터미널 재시작 시)
source venv/bin/activate

# 학습 재개 (체크포인트에서)
python scripts/train.py \
    --config configs/bert_base.yaml \
    --train_data data/train.json \
    --val_data data/val.json \
    --output_dir models/my_model  # 기존 경로
```

---

## 트러블슈팅

### 1. "ModuleNotFoundError"

```bash
# 가상환경이 활성화되어 있는지 확인
# 프롬프트에 (venv) 표시가 있어야 함

# 패키지 재설치
pip install -r requirements.txt
```

### 2. CUDA 메모리 부족

```bash
# batch_size 줄이기
python scripts/train.py --batch_size 8

# 또는 설정 파일 수정
# configs/bert_base.yaml:
#   batch_size: 8
```

### 3. CPU에서 학습

```bash
# fp16 비활성화
# configs/bert_base.yaml:
#   fp16: false
```

---

## 다음 단계

1. **더 많은 데이터 수집**: 1,000+ 샘플 목표
2. **데이터 증강**: 기존 데이터 5배 확장
3. **고급 모델 시도**: RoBERTa, DeBERTa
4. **하이퍼파라미터 튜닝**: Learning rate, epochs 조정
5. **앙상블**: 여러 모델 결합

---

## 도움말

- 문의: README.md 참조
- 자세한 가이드: `docs/` 폴더
- 예제: `examples/` 폴더

**Happy Extracting! 🚀**
