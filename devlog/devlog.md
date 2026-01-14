# 🛠️ Dev Log & Idea Sketch

## 📅 2026-01-12: 모델링 전략 수립 (Phase 1)

### 1. 🎯 핵심 문제 정의 (Problem Definition)

* **Label Granularity Mismatch**:
  * **Train Data**: `Full Text` 단위의 라벨 (0 or 1). `generated=1`인 경우에도 일부 문단은 사람(Human)이 썼을 가능성이 있음 (Label Noise 존재).
  * **Test Data**: `Paragraph` 단위의 확률 예측.
* **Goal**: 문단(Paragraph) 단위의 정밀한 AI/Human 판별 모델 구축.

### 2. 💡 접근 전략 (Strategy): Synthetic Data & 2-Stage Modeling

#### 2.1. 데이터셋 구축 아이디어 (Synthetic Dataset Construction)

* **가설**: Human 데이터(`generated=0`)의 문단은 100% 사람이 쓴 것으므로 신뢰할 수 있는 "Source"이다.
* **생성 모델(GenAI) 활용**:
  * Source(Human Paragraph)를 LLM(e.g., HyperCLOVA X, GPT-4, or Open Source LLM)에 주입.
  * 다양한 프롬프트로 변형 생성 (AI 데이터 확보):
    1. **Re-writing**: "이 문단을 AI 스타일로 다시 써줘."
    2. **Summarization**: "이 내용을 요약해줘."
    3. **Expansion**: "이 내용을 이어 써줘."
  * **결과**: (Human Para, 0) vs (Generated AI Para, 1)의 완벽한 문단 단위 쌍(Pair) 데이터셋 확보 가능.

#### 2.2. 모델 구조 아이디어 (2nd Order Structure)

* 단순 분류기를 넘어선 2단계 접근법 제안:
  * **Stage 1 (Generator/Teacher)**:
    * Human 데이터를 기반으로 고품질의 AI 문단 생성 (Data Augmentation).
  * **Stage 2 (Discriminator/Student)**:
    * 생성된 Synthetic Dataset으로 1차 문단 판별기 학습 (BERT/RoBERTa 등).
  * **Stage 3 (Refinement/Pseudo-labeling)**:
    * 학습된 1차 판별기를 사용하여, **기존 Train Data(`generated=1`)의 문단들을 검수**.
    * 기존 데이터 중 "진짜 AI 같은 문단"만 필터링(Denoising)하거나 가중치를 부여하여 모델 재학습 (Self-training).

### 3. 📝 To-Do List

- [ ] 문단 단위 분리(Split) 전처리 로직 구현.
- [ ] 생성 모델(LLM) 선정 및 프롬프트 엔지니어링 (AI 스타일 모방).
- [ ] 베이스라인 모델(BERT/RoBERTa) 선정 (`klue/roberta-large` 등).

---

## 📅 2026-01-12: DeepSeek-R1 방법론 적용 아이디어 (Phase 2)

### 🔬 DeepSeek-R1 핵심 방법론 요약

([참고](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-reasoning-llms))

1. **Cold Start**: 소량의 고품질 데이터로 SFT
2. **RL (강화학습)**: Rule-based Reward (정확도, 형식)로 모델 행동 유도
3. **Rejection Sampling**: 생성된 데이터 중 품질 좋은 것만 선별 → Synthetic Dataset 구축
4. **Distillation**: 큰 Teacher 모델의 지식을 작은 Student 모델에 전이

### 💡 우리 문제에 적용 (추상적 레벨)

| DeepSeek-R1 단계                    | 우리 적용                                                                            |
| ----------------------------------- | ------------------------------------------------------------------------------------ |
| **Cold Start (SFT)**          | Human 100% 확신 데이터(`generated=0`)의 문단으로 **Synthetic AI 문단 생성**  |
| **Rejection Sampling**        | 생성된 AI 문단 중**품질 좋은 것만 필터링** (Reward Model or Rule-based)        |
| **RL with Rule-based Reward** | 판별기(Discriminator)에 `정확도 Reward` + `일관성 Reward` 설계하여 Self-Training |
| **Distillation**              | LLM(Teacher)이 생성한 Synthetic Data로**작은 BERT(Student) 학습**              |

### 🔄 Proposed Pipeline (DeepSeek-Inspired 2-Stage)

```
[Phase 1: Data Generation]
Human 문단 (Label=0)
        │
        ▼
   LLM (Generator) ─────────> AI 스타일 문단 생성
        │
        ▼
   Rejection Sampling ────────> 품질 좋은 AI 문단 선별 (Rule or Reward Model)
        │
        ▼
Synthetic Dataset: (Human Para, 0) vs (AI Para, 1)


[Phase 2: Model Training]
        │
        ▼
   BERT/RoBERTa (Discriminator) 학습
        │
        ▼
   기존 Train Data (generated=1) Pseudo-labeling / Denoising
        │
        ▼
   Self-Training / Refinement
```

### 핵심 포인트

* **Generator가 Discriminator를 돕는 구조**: 생성 모델이 만들어낸 데이터로 판별 모델을 학습.
* **Rule-based Reward**: 생성된 AI 문단이 "AI 같은지" 판단하는 규칙 (e.g., perplexity, vocabulary diversity, 반복 패턴 등).
* **Distillation 효과**: LLM의 지식(AI 스타일)을 작은 BERT가 학습.

---

## 📅 2026-01-12: Feature Analysis 결과 (통계적 지문)

KoGPT2를 이용해 Human(100개) vs AI(100개) 샘플의 통계적 특징을 분석함.

### 📊 주요 발견 (Key Insights)

![1768191196975](image/devlog/1768191196975.png)


perplexity  mean_entropy  logprob_std  low_prob_ratio  trigram_rep_ratio  
0  194.258610      4.549955     3.549926        0.494118           0.011236
1   52.300789      3.020914     3.616106        0.356863           0.005882
2  113.846155      4.574265     3.174602        0.458824           0.011390
3   88.632998      4.332357     2.948958        0.380392           0.003782
4   76.841650      4.189429     2.833590        0.392157           0.000000

   bigram_rep_ratio  generated  label
0          0.049407          0  Human
1          0.030488          0  Human
2          0.025822          0  Human
3          0.024100          1     AI
4          0.006667          1     AI

1. **Perplexity (복잡도/예측불가능성)**

* **AI (95.6) < Human (126.9)**
* AI 텍스트가 훨씬 **"매끄럽고 예측 가능함"**. Human 텍스트는 의외의 단어 선택이나 독창적인 표현으로 인해 Perplexity가 높음.
* ➡️ **Rule-based Filter 1**: `Perplexity < 110` 인 경우 "AI스러움"으로 판단 가능.

2. **Logprob Consistency (확률 일관성)**

   * **AI (3.03) < Human (3.34)** (Logprob Std)
   * AI는 생성 시 확률 분포가 비교적 일관됨. 사람은 문장마다 확신/불확신 편차가 큼.
3. **Repetition (반복성)**

   * **AI (0.023) < Human (0.030)** (Bigram Repetition)
   * 의외로 **사람이 반복을 더 많이 함**. (특정 주제 강조, 관용구 사용 등).
   * AI는 Diverse decoding(Sampling) 덕분에 오히려 반복을 회피하는 경향이 있음.
   * ➡️ "반복이 많다고 무조건 AI는 아님" (오히려 그 반대일 수 있음).

### 🚀 적용 전략: Rejection Sampling Criteria

LLM으로 Synthetic Data 생성 후, 다음 조건을 만족하는 데이터만 **"High-Quality AI Data"**로 채택하여 학습에 사용:

1. **Low Perplexity**: `Perplexity`가 낮을수록 (예: 100 이하) AI 특징이 강함.
2. **Low Logprob Std**: 확률 변동성이 낮은 샘플.

---

## 📝 Feature Extraction Code Analysis (Dimensional Breakdown)

`get_lm_features` 함수 내부 로직 및 텐서 차원(Shape) 변화 분석.
(가정: `batch_size=1`, `seq_len=50`, `vocab_size=51200`)

### 1. Input Processing

```python
inputs = tokenizer(text, return_tensors="pt", ...).to(device)
# inputs['input_ids'] Shape: (1, 50) -> (Batch, Seq_Len)
```

### 2. Model Forward

```python
outputs = model(**inputs, labels=inputs["input_ids"])
logits = outputs.logits[:, :-1, :] 
# Raw Logits: (1, 50, 51200)
# Sliced Logits: (1, 49, 51200) -> (Batch, Seq_Len-1, Vocab_Size)
# (마지막 토큰은 맞출 정답이 없으므로 제외)

labels = inputs["input_ids"][:, 1:] 
# Labels: (1, 49) -> (Batch, Seq_Len-1)
# (첫 토큰은 예측 대상이 아니므로 제외)
```

### 3. Feature Calculation (Line-by-Line)

#### Perplexity (복잡도)

```python
loss = outputs.loss.item()      # Scalar (e.g., 4.5)
perplexity = np.exp(loss)       # Scalar (e.g., 90.01)
```

* 전체 Loss를 지수화하여 "평균 헷갈림 정도(Branching Factor)"를 측정.

#### Token-level Probabilities

```python
probs = torch.softmax(logits, dim=-1) 
# Shape: (1, 49, 51200)
# (Vocab 차원에 대해 확률합 1.0으로 정규화)
```

#### Mean Entropy (불확실성)

```python
entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
# 1. probs * log(probs): (1, 49, 51200)
# 2. Sum over dim=-1: (1, 49) -> 각 토큰별 엔트로피 값
mean_entropy = entropy.mean().item() # Scalar (평균)
```

#### Logprob of Actual Tokens (정답 확률)

```python
log_probs = torch.log(probs + 1e-9) 
# Shape: (1, 49, 51200)

actual_logprobs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
# 1. labels.unsqueeze(-1): (1, 49, 1) -> 인덱싱을 위해 차원 확장
# 2. gather: (1, 49, 51200)에서 정답 인덱스 위치의 값만 추출 -> (1, 49, 1)
# 3. squeeze: (1, 49) -> 다시 2차원으로 복귀
```

#### Statistics (일관성 및 저확률 빈도)

```python
logprob_std = actual_logprobs.std().item() 
# Scalar (로그 확률들의 표준편차 -> 일관성 지표)

low_prob_ratio = (actual_logprobs < -5).float().mean().item()
# 1. (actual_logprobs < -5): (1, 49) Bool Tensor (True/False)
# 2. .float(): (1, 49) Float Tensor (1.0/0.0)
# 3. .mean(): Scalar (비율)
```
