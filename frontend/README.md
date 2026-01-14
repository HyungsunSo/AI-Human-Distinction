# AI Text Detector - Frontend

AI/Human 텍스트 분류 대시보드 프론트엔드

## 기술 스택

- **HTML5** + **Vanilla CSS** + **Vanilla JavaScript**
- **Chart.js** - 차트 시각화

## 파일 구조

```
frontend/
├── index.html          # 메인 HTML
├── styles/
│   └── main.css        # 스타일시트
└── js/
    ├── api.js          # API 클라이언트
    ├── app.js          # 메인 앱 로직
    ├── charts.js       # Chart.js 래퍼
    └── lime.js         # LIME 시각화
```

---

## API 입출력 명세

### Backend URL
```
http://localhost:8000
```

### 1. 체크포인트 목록 조회

**Endpoint:** `GET /checkpoints`

**Response:**
```json
{
  "checkpoints": ["best_model.pt", "epoch3.pt"]
}
```

---

### 2. 체크포인트 로드

**Endpoint:** `POST /checkpoints/load`

**Request:**
```json
{
  "checkpoint_name": "best_model.pt"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Loaded best_model.pt"
}
```

---

### 3. 텍스트 분석 (메인 API)

**Endpoint:** `POST /analyze`

**Request:**
```json
{
  "text": "분석할 텍스트..."
}
```

**Response:**
```json
{
  "prediction": "AI",
  "confidence": 0.87,
  "paragraphs": [
    {
      "index": 0,
      "text": "첫 번째 문단...",
      "ai_prob": 0.92,
      "importance": 0.15
    }
  ],
  "top_paragraph": {
    "index": 0,
    "text": "가장 AI 확률 높은 문단",
    "ai_prob": 0.92
  },
  "lime_result": {
    "tokens": [
      { "word": "인공지능", "score": 0.12 },
      { "word": "자연스럽게", "score": -0.08 }
    ]
  },
  "deletion_test": {
    "original_prob": 0.87,
    "modified_prob": 0.56,
    "drop": 0.31,
    "reliability": "high",
    "removed_tokens": ["인공지능", "기술"]
  }
}
```

---

## 실행 방법

1. 백엔드 서버 실행 (8000 포트)
2. `index.html`을 브라우저에서 열기
3. 텍스트 입력 → Analyze 클릭

## Mock 모드

개발 시 백엔드 없이 테스트:
```javascript
// js/api.js
const USE_MOCK_DATA = true;  // Mock 데이터 사용
```
