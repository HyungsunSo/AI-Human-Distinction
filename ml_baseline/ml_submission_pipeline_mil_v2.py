"""
MIL (Multiple Instance Learning) v2 - Iterative Refinement
==========================================================
1. EDA 기반 정밀 피처 추출
2. Stage 1 (Initial Paragraph Model): 문서 라벨로 학습
3. Label Cleaning (Iterative): 
   - AI 문서 내에서 점수가 낮은(Human-like) 문단들의 라벨을 0으로 보정
4. Stage 2 (Clean Paragraph Model): 정제된 라벨로 재학습
5. Meta-Model (Pooling): 
   - 문서별 통계치 풀링 후 최종 문서 점수 산출
6. Inference: 
   - 문단 점수와 문서 점수 결합 (순위 보존형 앙상블)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import re
from tqdm import tqdm
import warnings
import os

warnings.filterwarnings('ignore')

# 기본 경로
BASE_DIR = '/Users/youngjinson/멋사1/AI-Human-Distinction'
OPEN_DIR = os.path.join(BASE_DIR, 'open')
OUTPUT_DIR = os.path.join(BASE_DIR, 'ml_baseline')

# 기능어 패턴 (조사, 어미) - EDA 기반
PARTICLES = ['은', '는', '이', '가', '을', '를', '에', '에서', '으로', '로', '와', '과', '의', '도', '만', '까지', '부터', '에게', '한테', '께']
ENDINGS = ['다', '며', '고', '지만', '는데', '면서', '지', '니', '라', '자', '려고', '도록', '듯이', '처럼']

def extract_upgraded_features(text):
    """EDA 기반 정밀 메타 피처 추출"""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {
            'sent_len_median': 0, 'sent_len_p90': 0, 'sent_len_std': 0, 'sent_len_cv': 0,
            'comma_density': 0, 'particle_density': 0, 'ending_density': 0,
            'repeat_ratio': 0, 'ttr': 1, 'text_len': 0, 'n_words': 0
        }
    
    text = text.strip()
    text_len = len(text)
    words = text.split()
    n_words = len(words)
    
    # 문장 분할 및 통계
    sentences = [s.strip() for s in re.split(r'[.!?。]\s*', text) if s.strip()]
    sent_lengths = [len(s) for s in sentences] if sentences else [0]
    
    # 어휘 다양성
    unique_words = set(words)
    repeat_ratio = 1 - (len(unique_words) / n_words) if n_words > 0 else 0
    ttr = len(unique_words) / n_words if n_words > 0 else 1
    
    # 정규화 기준 (100자당)
    norm = text_len / 100 if text_len > 0 else 1
    
    # 구두점 및 기능어
    comma_cnt = text.count(',') + text.count('，')
    particle_cnt = sum(text.count(p) for p in PARTICLES)
    ending_cnt = sum(text.count(e) for e in ENDINGS)
    
    # 피처 셋
    feats = {
        'sent_len_median': np.median(sent_lengths),
        'sent_len_p90': np.percentile(sent_lengths, 90) if len(sent_lengths) >= 2 else np.median(sent_lengths),
        'sent_len_std': np.std(sent_lengths) if len(sent_lengths) > 1 else 0,
        'comma_density': comma_cnt / norm,
        'particle_density': particle_cnt / norm,
        'ending_density': ending_cnt / norm,
        'repeat_ratio': repeat_ratio,
        'ttr': ttr,
        'text_len': text_len,
        'n_words': n_words,
        'avg_word_len': text_len / n_words if n_words > 0 else 0
    }
    feats['sent_len_cv'] = feats['sent_len_std'] / (feats['sent_len_median'] + 1e-6)
    
    return feats

# =============================================================================
# 1. 데이터 준비
# =============================================================================
print("📂 [1/6] 데이터 로딩 및 피처 추출...")
train_df = pd.read_csv(os.path.join(OPEN_DIR, 'train.csv'))
test_df = pd.read_csv(os.path.join(OPEN_DIR, 'test.csv'))

# Train 문단 분리 & 피처 추출
train_paras = []
for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Train parsing"):
    full_text = str(row['full_text'])
    paras = [p.strip() for p in full_text.split('\n') if p.strip()]
    for i, p in enumerate(paras):
        f = extract_upgraded_features(p)
        f.update({'doc_idx': idx, 'p_idx': i, 'generated': row['generated']})
        train_paras.append(f)
train_para_df = pd.DataFrame(train_paras)

# Test 피처 추출
test_paras = []
for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Test parsing"):
    f = extract_upgraded_features(row['paragraph_text'])
    f.update({'ID': row['ID'], 'title': row['title'], 'p_idx': row['paragraph_index']})
    test_paras.append(f)
test_para_df = pd.DataFrame(test_paras)

# =============================================================================
# 2. Stage 1: Initial Para Model (Noisy Label)
# =============================================================================
print("\n🚀 [2/6] Stage 1 학습 (Noisy Label)...")
features = [c for c in train_para_df.columns if c not in ['doc_idx', 'p_idx', 'generated']]
X_para = train_para_df[features]
y_para = train_para_df['generated']

gkf = GroupKFold(n_splits=5)
para_model = HistGradientBoostingClassifier(max_iter=300, random_state=42)

oof_para_scores = np.zeros(len(train_para_df))
for tr_idx, val_idx in gkf.split(X_para, y_para, groups=train_para_df['doc_idx']):
    para_model.fit(X_para.iloc[tr_idx], y_para.iloc[tr_idx])
    oof_para_scores[val_idx] = para_model.predict_proba(X_para.iloc[val_idx])[:, 1]

# =============================================================================
# 3. Step 2: Label Cleaning (Iterative)
# =============================================================================
print("\n🧹 [3/6] Label Cleaning (AI 문서 내 Human 문단 식별)...")
train_para_df['initial_score'] = oof_para_scores

# AI 문서(generated=1)이면서 점수가 낮은 문단은 사실 Human일 확률이 높음
# 임계값: 0.2 (하위 점수는 Human으로 간주)
clean_y = train_para_df['generated'].copy()
# AI 문서 내에서 점수가 극히 낮은 문단들 필터링
noise_mask = (train_para_df['generated'] == 1) & (train_para_df['initial_score'] < 0.2)
clean_y[noise_mask] = 0
print(f"   - 정제된 문단 수: {noise_mask.sum():,} (AI -> Human 보정)")

# =============================================================================
# 4. Stage 3: Cleaned Para Model Re-train
# =============================================================================
print("\n🚀 [4/6] Stage 3 학습 (Cleaned Label)...")
para_model_clean = HistGradientBoostingClassifier(max_iter=300, random_state=42)
para_model_clean.fit(X_para, clean_y)

# =============================================================================
# 5. Stage 4: Meta-Model & Pooling
# =============================================================================
print("\n🚀 [5/6] Meta-Model 학습 (Pooling)...")
train_para_df['clean_score'] = para_model_clean.predict_proba(X_para)[:, 1]

doc_meta = train_para_df.groupby('doc_idx')['clean_score'].agg([
    ('max_val', 'max'), ('mean_val', 'mean'), ('q90_val', lambda x: np.percentile(x, 90)), ('std_val', 'std')
]).fillna(0)

doc_meta['actual_label'] = train_df['generated']
meta_model = HistGradientBoostingClassifier(max_iter=100, random_state=42)
meta_model.fit(doc_meta.drop('actual_label', axis=1), doc_meta['actual_label'])

# =============================================================================
# 6. Inference & Submission
# =============================================================================
print("\n🔮 [6/6] Inference 및 최종 Submission 생성...")
X_test = test_para_df[features]
test_para_df['raw_score'] = para_model_clean.predict_proba(X_test)[:, 1]

# Title(문서) 레벨로 묶어서 Meta 점수 산출
test_doc_meta = test_para_df.groupby('title')['raw_score'].agg([
    ('max_val', 'max'), ('mean_val', 'mean'), ('q90_val', lambda x: np.percentile(x, 90)), ('std_val', 'std')
]).fillna(0)

test_doc_meta['doc_score'] = meta_model.predict_proba(test_doc_meta)[:, 1]

# 최종 점수 산출 (문단 점수 위주, 문서 점수 보조)
# *중의 사항*: 문단 ID가 타겟이므로 문서 점수가 너무 각 문단을 지배하면 안 됨.
test_final = test_para_df.merge(test_doc_meta[['doc_score']], on='title', how='left')
# 문단 점수와 문서 점수의 곱/가중치 합 (순위 보존)
test_final['final_prob'] = (test_final['raw_score'] * 0.8) + (test_final['doc_score'] * 0.2)

submission = pd.DataFrame({
    'ID': test_para_df['ID'],
    'generated': test_final['final_prob']
})

out_path = os.path.join(OUTPUT_DIR, 'submission_mil_v2_iterative.csv')
submission.to_csv(out_path, index=False)
print(f"\n✅ 완료! {out_path}")
print(f"📊 분포 요약:\n{submission['generated'].describe()}")
