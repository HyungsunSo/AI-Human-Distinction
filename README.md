# 🤖 AI-Human Distinction: Stylistic & Deep Learning Dashboard

본 프로젝트는 BERT 기반의 딥러닝 탐지와 EDA 기반의 통계적 문체 분석을 결합하여, 한국어 텍스트의 AI 생성 여부를 다각도로 판별하는 통합 대시보드 시스템입니다.

---

## 🏗️ 시스템 아키텍처

시스템은 두 가지 핵심 엔진의 하이브리드 방식으로 동작합니다.

### 1. BERT Deep Learning Engine (Contextual)
- **Hierarchical Analysis**: 문단 단위로 쪼개어 각각 AI 확률을 계산한 뒤, 가장 의심스러운 문단을 선정합니다.
- **Explainability (LIME)**: 선정된 문단 내에서 어떤 단어가 AI 또는 사람의 특징인지 토큰 레벨에서 시각화합니다.
- **Reliability (Deletion Test)**: 핵심 단어를 제거했을 때 모델의 점수 하락폭을 측정하여 분석 결과의 신뢰도를 검증합니다.

### 2. Stylistic Fingerprint Engine (Statistical)
- **Real Stats Based**: `open/train_with_all_features.csv`에서 추출한 **실제 데이터의 5대 핵심 지표(쉼표 밀도, 문장 길이, 조사/어미 밀도, 어휘 다양성)** 통계치를 사용합니다.
- **Comparative Distribution**: 전체 Human/AI 데이터 분포 내에서 현재 입력된 텍스트가 어느 위치에 있는지 p-value와 함께 보여줍니다.

---

## 📡 API Endpoints (Backend)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | 텍스트 입력 및 전체 분석 결과(딥러닝 + 문체) 반환 |
| `/checkpoints` | GET | 사용 가능한 모델 체크포인트 목록 조회 |
| `/checkpoints/load` | POST | 특정 가중치 모델 로드 |

### /analyze 응답 구조 예시
```json
{
  "prediction": "AI",
  "confidence": 0.92,
  "paragraphs": [...],
  "top_paragraph": {...},
  "lime_result": {"tokens": [...]},
  "deletion_test": {"reliability": "high", ...},
  "meta_analysis": {
    "features": [
      {
        "display_name": "쉼표 밀도",
        "value": 0.75,
        "p_value": 0.03,
        "interpretation": "AI 패턴에 가깝습니다."
      }
    ]
  }
}
```

---

## 📊 데이터 소스 및 신뢰성 안내
- **더미 데이터 여부**: `meta_analyzer.py`에 정의된 상수는 EDA 노트북을 통해 `train_with_all_features.csv`에서 직접 산출한 **실제 통계값**입니다.
- **모델 신뢰도**: LIME 엔진과 삭제 테스트를 통해 인공지능이 "왜" 그렇게 판단했는지에 대한 논리적 근거를 제공합니다.

---

## 🚀 시작하기
- **백엔드**: `cd backend && uvicorn main:app --reload`
- **프론트엔드**: `cd frontend && python3 -m http.server 3000` (OR VSCode Live Server)
