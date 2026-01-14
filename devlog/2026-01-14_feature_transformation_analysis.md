# 2026-01-14 Feature Transformation Analysis

## 📊 문체 분석 피처 변환 실험 결과

EDA 기반 문체 분석에서 사용하는 5가지 핵심 피처의 분포 특성을 분석하고, **Log 변환 적용 여부**를 결정했습니다.

---

## 분석 배경

- **문제점**: 일부 피처(문장 길이, 어휘 반복률 등)가 긴 꼬리 분포(Long-tail)를 가져 정규분포 가정 기반의 p-value 계산이 부정확할 수 있음
- **목표**: 왜도(Skewness)를 줄여 정규성을 개선하고, 더 정확한 통계적 비교 수행

---

## 피처별 분석 결과

| 피처명 | Human 왜도 | AI 왜도 | KS 분리도 | Log 변환 후 왜도(Human) | 결정 |
|--------|-----------|---------|-----------|-------------------------|------|
| `sent_len_median` | **27.09** ⚠️ | 1.56 | 0.234 | 0.83 ✅ | **Log 변환 적용** |
| `comma_density` | **2.32** ⚠️ | 2.18 | 0.338 | 0.20 ✅ | **Log 변환 적용** |
| `particle_per_100char` | -0.58 | -0.26 | 0.023 | -2.64 (악화) | 원본 유지 |
| `ending_per_100char` | 0.36 | 0.56 | 0.095 | 변화 미미 | 원본 유지 |
| `repeat_ratio_mean` | **6.57** ⚠️ | 2.61 | 0.182 | 5.00 (약간 개선) | **Log 변환 적용** |

---

## 주요 인사이트

### 1. Log 변환이 KS 분리도를 높이진 않음
- 변환 전후 KS 통계량은 거의 동일
- **그러나** 왜도가 줄어들어 정규분포 가정이 더 타당해짐 → p-value 정확도 향상

### 2. 상대값(문서 길이 대비) vs Z-score
- 문서 길이로 나누는 상대값은 해석이 어려워짐
- **전역 분포 기준 Z-score가 더 직관적**: "평균에서 몇 표준편차 떨어졌는가"

### 3. 변환 후 새로운 통계치 산출
변환된 분포에서의 Mean/Std를 다시 계산하여 `meta_analyzer.py`에 적용:

```python
# Log1p 변환 적용 피처 (from train_with_all_features.csv)
'sent_len_median': {'human': {'mean': 2.642, 'std': 0.248}, 'ai': {'mean': 2.529, 'std': 0.201}}
'comma_density':   {'human': {'mean': 0.709, 'std': 0.266}, 'ai': {'mean': 0.509, 'std': 0.255}}
'repeat_ratio_mean': {'human': {'mean': 0.016, 'std': 0.015}, 'ai': {'mean': 0.011, 'std': 0.010}}
```

---

## 적용 결과

### Backend 변경 (`meta_analyzer.py` v2)
- `FEATURE_CONFIG`에 `transform: 'log' | 'raw'` 필드 추가
- `_transform_value()` 메서드로 피처별 변환 적용
- 변환된 분포 기준의 새로운 통계치 사용

### Frontend 변경 (`charts.js`)
- 차트 렌더링 시 log 피처는 `Math.log1p(value)`로 변환
- 분포 범위를 ±3σ로 확장

---

## 테스트 결과 예시

```
입력: AI 스타일 문서 (4문단)
결과: 5개 지표 중 1개 AI 패턴, 4개 Human 패턴
      - 조사 밀도만 AI에 가까움 (p=0.166)
      - 문장 길이, 어휘 반복률은 Human 분포에서 벗어남
```

---

## 다음 단계 (Optional)

1. **Box-Cox 변환** 탐색: log보다 더 유연한 변환
2. **Robust Z-score** 적용: 중앙값/IQR 기반으로 이상치에 강건한 표준화
3. **피처 가중치** 도입: KS 분리도가 높은 피처에 더 높은 중요도 부여

---

*분석 도구: Python, pandas, scipy.stats*
*데이터 소스: `open/train_with_all_features.csv`*
