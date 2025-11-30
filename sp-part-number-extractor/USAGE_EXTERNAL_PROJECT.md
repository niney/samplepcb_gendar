# 외부 프로젝트에서 Part Number Extractor 사용하기

이 문서는 `sp-part-number-extractor`를 다른 Python 프로젝트에서 사용하는 방법을 설명합니다.

---

## 📋 목차

1. [설치 방법](#-설치-방법)
   - [방법 1: pip install -e (권장)](#방법-1-pip-install--e-권장)
   - [방법 2: Wheel 패키지 배포](#방법-2-wheel-패키지-배포)
   - [방법 3: sys.path 추가 (설치 없이)](#방법-3-syspath-추가-설치-없이)
2. [코드 사용법](#-코드-사용법)
   - [기본 사용](#기본-사용)
   - [단일 행 예측](#단일-행-예측)
   - [배치 예측 (여러 행)](#배치-예측-여러-행)
   - [CSV/Excel 파일 처리](#csvexcel-파일-처리)
   - [신뢰도 기반 필터링](#신뢰도-기반-필터링)
3. [래퍼 클래스 예시](#-래퍼-클래스-예시)
4. [방법 비교표](#-방법-비교표)

---

## 📦 설치 방법

### 방법 1: pip install -e (권장)

**editable 모드**로 설치하면 소스 코드 수정 시 자동 반영됩니다.

#### 같은 가상환경인 경우

```powershell
cd d:\work\workspace_other\samplepcb_gendar\sp-part-number-extractor
pip install -e .
```

#### 다른 가상환경인 경우

```powershell
# 1. 외부 프로젝트의 가상환경 활성화
cd d:\your\external\project
.\venv\Scripts\activate

# 2. sp-part-number-extractor 경로로 설치
pip install -e d:\work\workspace_other\samplepcb_gendar\sp-part-number-extractor
```

#### 확인

```powershell
pip list | findstr part-number
# 출력: sp-part-number-extractor    1.0.0    d:\work\...\sp-part-number-extractor
```

#### 제거

```powershell
pip uninstall sp-part-number-extractor
# Proceed (Y/n)? 물으면 Y 입력
```

---

### 방법 2: Wheel 패키지 배포

배포 가능한 `.whl` 파일을 생성하여 설치합니다.

#### Step 1: Wheel 생성

```powershell
# sp-part-number-extractor 디렉토리에서
cd d:\work\workspace_other\samplepcb_gendar\sp-part-number-extractor
.\venv\Scripts\activate

# 빌드 도구 설치
pip install wheel build

# wheel 생성
python -m build
```

생성된 파일:
```
dist/
├── sp_part_number_extractor-1.0.0-py3-none-any.whl
└── sp_part_number_extractor-1.0.0.tar.gz
```

#### Step 2: 외부 프로젝트에서 설치

```powershell
# 외부 프로젝트 가상환경 활성화
cd d:\your\external\project
.\venv\Scripts\activate

# wheel 파일로 설치
pip install d:\work\workspace_other\samplepcb_gendar\sp-part-number-extractor\dist\sp_part_number_extractor-1.0.0-py3-none-any.whl
```

#### 업데이트 시

```powershell
# 1. 새 wheel 생성
cd d:\work\workspace_other\samplepcb_gendar\sp-part-number-extractor
python -m build

# 2. 재설치
pip install --force-reinstall dist\sp_part_number_extractor-1.0.0-py3-none-any.whl
```

#### 제거

```powershell
pip uninstall sp-part-number-extractor
# Proceed (Y/n)? 물으면 Y 입력
```

---

### 방법 3: sys.path 추가 (설치 없이)

패키지 설치 없이 직접 경로를 추가하는 방법입니다.

#### 의존성 설치 (필수)

```powershell
pip install torch transformers pandas numpy scikit-learn pyyaml tqdm safetensors
```

#### 코드에서 경로 추가

```python
import sys
sys.path.append("d:/work/workspace_other/samplepcb_gendar/sp-part-number-extractor")

# 이후 정상적으로 import 가능
from src.inference.predictor import SpPartNumberPredictor
```

---

## 💻 코드 사용법

### 기본 사용

```python
from src.inference.predictor import SpPartNumberPredictor

# 학습된 모델 경로
MODEL_PATH = "d:/work/workspace_other/samplepcb_gendar/sp-part-number-extractor/models/checkpoint/final_model"

# Predictor 초기화
predictor = SpPartNumberPredictor(MODEL_PATH)
```

---

### 단일 행 예측

```python
from src.inference.predictor import SpPartNumberPredictor

predictor = SpPartNumberPredictor("path/to/model")

# BOM 한 행 데이터 (셀 리스트)
row = ["R15 R16", "RC0402FR-0710KL", "RES 10K OHM 1%", "2", "Yageo", "0402"]

# 예측
result = predictor.predict(row)

# 결과 출력
print(f"입력: {row}")
print(f"Part Number: {result['part_number']}")
print(f"신뢰도: {result['confidence']:.2%}")
print(f"셀 위치: {result['cell_index']}")
```

**출력 예시:**
```
입력: ['R15 R16', 'RC0402FR-0710KL', 'RES 10K OHM 1%', '2', 'Yageo', '0402']
Part Number: RC0402FR-0710KL
신뢰도: 95.32%
셀 위치: 1
```

---

### 배치 예측 (여러 행)

```python
from src.inference.predictor import SpPartNumberPredictor

predictor = SpPartNumberPredictor("path/to/model")

# 여러 행 데이터
rows = [
    ["C1 C2", "CC0402KRX7R9BB102", "CAP CER 1000PF", "2", "Yageo"],
    ["R1", "RC0402FR-07100KL", "RES 100K", "1", "Yageo"],
    ["U1", "STM32F103C8T6", "MCU ARM", "1", "STMicroelectronics"],
    ["D1 D2 D3", "BAT54S", "DIODE SCHOTTKY", "3", "ON Semi"],
]

# 배치 예측
results = predictor.batch_predict(rows, batch_size=32)

# 결과 출력
for i, (row, result) in enumerate(zip(rows, results)):
    print(f"Row {i+1}: {result['part_number']:25} (confidence: {result['confidence']:.2%})")
```

**출력 예시:**
```
Row 1: CC0402KRX7R9BB102         (confidence: 94.21%)
Row 2: RC0402FR-07100KL          (confidence: 96.15%)
Row 3: STM32F103C8T6             (confidence: 92.87%)
Row 4: BAT54S                    (confidence: 91.43%)
```

---

### CSV/Excel 파일 처리

```python
import pandas as pd
from src.inference.predictor import SpPartNumberPredictor

# 모델 로드
predictor = SpPartNumberPredictor("path/to/model")

# CSV 파일 읽기
df = pd.read_csv("input_bom.csv", header=None)

# 빈 값 처리 및 문자열 변환
rows = df.fillna('').astype(str).values.tolist()

# 예측
results = predictor.batch_predict(rows, batch_size=32)

# 결과를 DataFrame에 추가
df['predicted_part_number'] = [r['part_number'] for r in results]
df['confidence'] = [r['confidence'] for r in results]

# 결과 저장
df.to_csv("output_with_predictions.csv", index=False)

# 통계 출력
print(f"총 행: {len(results)}")
print(f"Part Number 발견: {sum(1 for r in results if r['part_number'])}")
print(f"평균 신뢰도: {sum(r['confidence'] for r in results) / len(results):.2%}")
```

#### Excel 파일 처리

```python
# Excel 파일 읽기 (openpyxl 필요: pip install openpyxl)
df = pd.read_excel("input_bom.xlsx", header=None)

# 이후 동일한 처리
rows = df.fillna('').astype(str).values.tolist()
results = predictor.batch_predict(rows)
```

---

### 신뢰도 기반 필터링

```python
from src.inference.predictor import SpPartNumberPredictor

predictor = SpPartNumberPredictor("path/to/model")

row = ["C1", "CC0402KRX7R9BB102", "CAP", "1"]

# 신뢰도 임계값 적용
result = predictor.predict_with_threshold(row, confidence_threshold=0.8)

if result['is_confident']:
    print(f"확정: {result['part_number']}")
else:
    print(f"검토 필요: {result['part_number']} (신뢰도: {result['confidence']:.2%})")
```

---

## 🔧 래퍼 클래스 예시

프로젝트에서 편리하게 사용하기 위한 래퍼 클래스입니다.

```python
"""
bom_extractor.py - BOM Part Number 추출 래퍼 클래스
"""

import pandas as pd
from typing import List, Dict, Optional
from src.inference.predictor import SpPartNumberPredictor


class BOMPartNumberExtractor:
    """BOM 파일에서 Part Number를 추출하는 유틸리티 클래스"""
    
    def __init__(
        self, 
        model_path: str, 
        confidence_threshold: float = 0.7,
        device: Optional[str] = None
    ):
        """
        Args:
            model_path: 학습된 모델 디렉토리 경로
            confidence_threshold: 최소 신뢰도 임계값 (0.0 ~ 1.0)
            device: 'cuda' 또는 'cpu' (None이면 자동 선택)
        """
        self.predictor = SpPartNumberPredictor(model_path, device=device)
        self.threshold = confidence_threshold
    
    def extract_from_row(self, row: List[str]) -> Dict:
        """
        단일 행에서 Part Number 추출
        
        Args:
            row: BOM 행의 셀 데이터 리스트
            
        Returns:
            {
                'part_number': str or None,
                'confidence': float,
                'cell_index': int or None,
                'is_reliable': bool,
                'needs_review': bool
            }
        """
        result = self.predictor.predict(row)
        return {
            'part_number': result['part_number'],
            'confidence': result['confidence'],
            'cell_index': result['cell_index'],
            'is_reliable': result['confidence'] >= self.threshold,
            'needs_review': result['confidence'] < self.threshold
        }
    
    def extract_from_rows(
        self, 
        rows: List[List[str]], 
        batch_size: int = 32
    ) -> List[Dict]:
        """
        여러 행에서 Part Number 추출
        
        Args:
            rows: BOM 행 리스트
            batch_size: 배치 크기
            
        Returns:
            결과 딕셔너리 리스트
        """
        results = self.predictor.batch_predict(rows, batch_size=batch_size)
        
        return [
            {
                'part_number': r['part_number'],
                'confidence': r['confidence'],
                'cell_index': r['cell_index'],
                'is_reliable': r['confidence'] >= self.threshold,
                'needs_review': r['confidence'] < self.threshold
            }
            for r in results
        ]
    
    def extract_from_file(
        self, 
        file_path: str, 
        has_header: bool = False,
        sheet_name: int = 0
    ) -> pd.DataFrame:
        """
        파일에서 Part Number 추출
        
        Args:
            file_path: CSV 또는 Excel 파일 경로
            has_header: 헤더 행 존재 여부
            sheet_name: Excel 시트 인덱스 (0부터 시작)
            
        Returns:
            예측 결과가 추가된 DataFrame
        """
        # 파일 읽기
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, header=0 if has_header else None)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(
                file_path, 
                header=0 if has_header else None,
                sheet_name=sheet_name
            )
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {file_path}")
        
        # 예측
        rows = df.fillna('').astype(str).values.tolist()
        results = self.extract_from_rows(rows)
        
        # 결과 컬럼 추가
        df['_part_number'] = [r['part_number'] for r in results]
        df['_confidence'] = [r['confidence'] for r in results]
        df['_cell_index'] = [r['cell_index'] for r in results]
        df['_needs_review'] = [r['needs_review'] for r in results]
        
        return df
    
    def get_statistics(self, results: List[Dict]) -> Dict:
        """
        추출 결과 통계
        
        Args:
            results: extract_from_rows의 결과
            
        Returns:
            통계 딕셔너리
        """
        total = len(results)
        found = sum(1 for r in results if r['part_number'])
        reliable = sum(1 for r in results if r['is_reliable'])
        needs_review = sum(1 for r in results if r['needs_review'] and r['part_number'])
        
        avg_confidence = sum(r['confidence'] for r in results) / total if total > 0 else 0
        
        return {
            'total_rows': total,
            'part_numbers_found': found,
            'reliable_predictions': reliable,
            'needs_review': needs_review,
            'average_confidence': avg_confidence,
            'detection_rate': found / total if total > 0 else 0,
            'reliability_rate': reliable / total if total > 0 else 0
        }


# 사용 예시
if __name__ == "__main__":
    # 초기화
    extractor = BOMPartNumberExtractor(
        model_path="d:/path/to/models/checkpoint/final_model",
        confidence_threshold=0.8
    )
    
    # 단일 행 처리
    result = extractor.extract_from_row(
        ["C1 C2", "GRM155R71C104KA88D", "CAP 0.1UF", "2", "Murata"]
    )
    print(f"Part Number: {result['part_number']}")
    print(f"Reliable: {result['is_reliable']}")
    
    # 파일 처리
    df = extractor.extract_from_file("input_bom.csv")
    df.to_csv("output_bom.csv", index=False)
    
    # 통계 확인
    rows = df.fillna('').astype(str).values.tolist()
    results = extractor.extract_from_rows(rows)
    stats = extractor.get_statistics(results)
    
    print(f"\n=== 추출 통계 ===")
    print(f"총 행: {stats['total_rows']}")
    print(f"Part Number 발견: {stats['part_numbers_found']}")
    print(f"신뢰할 수 있는 예측: {stats['reliable_predictions']}")
    print(f"검토 필요: {stats['needs_review']}")
    print(f"평균 신뢰도: {stats['average_confidence']:.2%}")
```

---

## 📊 방법 비교표

| 항목 | 방법 1 (pip -e) | 방법 2 (wheel) | 방법 3 (sys.path) |
|------|----------------|----------------|-------------------|
| **설치 복잡도** | ⭐ 쉬움 | ⭐⭐ 중간 | ⭐ 쉬움 |
| **소스 수정 반영** | ✅ 자동 | ❌ 재빌드 필요 | ✅ 자동 |
| **배포 용이성** | ❌ 경로 의존 | ✅ 독립 배포 가능 | ❌ 경로 의존 |
| **의존성 관리** | ✅ 자동 | ✅ 자동 | ❌ 수동 설치 |
| **권장 상황** | 개발 중 | 프로덕션 배포 | 빠른 테스트 |

---

## ⚠️ 주의사항

### 모델 경로
- 학습된 모델 디렉토리 경로를 정확히 지정해야 합니다.
- 상대 경로보다 **절대 경로** 권장

```python
# 권장
model_path = "d:/work/workspace_other/samplepcb_gendar/sp-part-number-extractor/models/checkpoint/final_model"

# 비권장 (현재 디렉토리에 따라 오류 가능)
model_path = "models/checkpoint/final_model"
```

### GPU 사용
- GPU가 있으면 자동으로 사용됩니다.
- CPU만 사용하려면:

```python
predictor = SpPartNumberPredictor(model_path, device="cpu")
```

### Python 버전
- Python 3.8 이상 필요

### 필수 의존성
```
torch>=1.9.0
transformers>=4.20.0
pandas>=1.3.0
numpy>=1.20.0
safetensors>=0.3.0
```

---

## 📞 문제 해결

### ImportError: No module named 'src'

**원인:** 패키지가 설치되지 않음

**해결:**
```powershell
pip install -e d:\work\workspace_other\samplepcb_gendar\sp-part-number-extractor
```

### 모델 로드 실패

**원인:** 모델 경로가 잘못되었거나 모델 파일이 없음

**확인:**
```python
from pathlib import Path
model_path = "your/model/path"
print(f"경로 존재: {Path(model_path).exists()}")
print(f"파일 목록: {list(Path(model_path).glob('*'))}")
```

### CUDA out of memory

**해결:** CPU 모드로 실행
```python
predictor = SpPartNumberPredictor(model_path, device="cpu")
```

---

## 📝 라이선스

MIT License
