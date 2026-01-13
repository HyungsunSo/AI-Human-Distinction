# 🤖 GGUF 모델 사용 가이드

## 📌 사용 가능한 모델

현재 시스템에 다운로드된 모델:

1. **HyperCLOVAX 0.5B** (Q4_K_M)
   - 경로: `~/.cache/huggingface/hub/HyperCLOVAX-GGUF/hyperclovax-seed-text-instruct-0.5b-q4_k_m.gguf`
   - 크기: 412MB
   - 특징: 한국어 특화

2. **Qwen3 0.6B** (Q4_K_M)
   - 경로: `~/.cache/huggingface/hub/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf`
   - 크기: 378MB
   - 특징: 다국어 지원

3. **Qwen3 1.7B** (Q8_0)
   - 경로: `~/.cache/huggingface/hub/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q8_0.gguf`
   - 크기: 1.7GB
   - 특징: 더 강력한 성능

## 🚀 사용 방법

### 1. 모델 선택

`hyperclova_generation.ipynb`에서 다음 라인을 수정:

```python
MODEL_CHOICE = "hyperclova"  # 옵션: "hyperclova", "qwen-0.6b", "qwen-1.7b"
```

### 2. 출력 파일명

선택한 모델에 따라 자동으로 파일명이 설정됩니다:

- `hyperclova` → `hyperclova_synthetic_pairs.csv`
- `qwen-0.6b` → `qwen06b_synthetic_pairs.csv`
- `qwen-1.7b` → `qwen17b_synthetic_pairs.csv`

### 3. 노트북 실행

Jupyter Notebook에서 셀을 순서대로 실행하면 됩니다.

## 📊 모델 비교

| 모델 | 크기 | 양자화 | 속도 | 품질 | 추천 용도 |
|------|------|--------|------|------|-----------|
| HyperCLOVA 0.5B | 412MB | Q4_K_M | 빠름 | 중간 | 한국어 특화 작업 |
| Qwen3 0.6B | 378MB | Q4_K_M | 빠름 | 중간 | 빠른 테스트 |
| Qwen3 1.7B | 1.7GB | Q8_0 | 중간 | 높음 | 최고 품질 필요 시 |

## 💡 팁

- **빠른 테스트**: Qwen3 0.6B 추천
- **한국어 품질 우선**: HyperCLOVA 0.5B 추천  
- **최고 품질**: Qwen3 1.7B 추천 (단, 느림)
