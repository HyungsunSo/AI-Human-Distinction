"""
MIL (Multiple Instance Learning) 기반 AI/Human 구분 Submission 파이프라인
========================================================================
1. Stage 1 (Paragraph Training): 
   - train.csv를 문단 단위로 쪼개어 학습 (Label = 문서 Label, Noisy Labeling)
2. Stage 2 (Scoring & Pooling): 
   - 학습된 문단 모델로 train.csv 내 모든 문단 점수 산출
   - 문서별로 문단 점수들의 통계치(Max, Mean, Std, Top-K)를 피처로 생성
3. Stage 3 (Meta-Classification):
   - 문서별 풀링 점수를 피처로 하여 최종 문서 분류기 학습
4. Inference:
   - test.csv 문단별 점수 산출
   - title(문서)별로 묶어 Meta-Model을 통한 점수 보정 (Optional but helpful)
   - 최종 Submission 생성 (확률값)
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

# 기능어 패턴 (조사, 어미)
PARTICLES = ['은', '는', '이', '가', '을', '를', '에', '에서', '으로', '로', '와', '과', '의', '도', '만', '까지', '부터', '에게', '한테', '께']
ENDINGS = ['다', '며', '고', '지만', '는데', '면서', '지', '니', '라', '자', '려고', '도록', '듯이', '처럼']

# =============================================================================
# 1. 피처 추출 엔진
# =============================================================================

def extract_features(text):
    """단일 텍스트(문단)에서 메타 피처 추출"""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {
            'sent_len_median': 0, 'sent_len_std': 0, 'comma_density': 0, 
            'repeat_ratio': 0, 'ttr': 1, 'particle_density': 0, 
            'ending_density': 0, 'text_len': 0, 'n_words': 0
        }
    
    text = text.strip()
    text_len = len(text)
    words = text.split()
    n_words = len(words)
    
    # 문장 분할 및 길이
    sentences = [s.strip() for s in re.split(r'[.!?。]\s*', text) if s.strip()]
    sent_lengths = [len(s) for s in sentences] if sentences else [0]
    
    # 어휘 다양성
    unique_words = set(words)
    repeat_ratio = 1 - (len(unique_words) / n_words) if n_words > 0 else 0
    ttr = len(unique_words) / n_words if n_words > 0 else 1
    
    # 밀도 피처
    comma_cnt = text.count(',') + text.count('，')
    particle_cnt = sum(text.count(p) for p in PARTICLES)
    ending_cnt = sum(text.count(e) for e in ENDINGS)
    
    norm = text_len / 100 if text_len > 0 else 1
    
    return {
        'sent_len_median': np.median(sent_lengths),
        'sent_len_std': np.std(sent_lengths) if len(sent_lengths) > 1 else 0,
        'comma_density': comma_cnt / norm,
        'repeat_ratio': repeat_ratio,
        'ttr': ttr,
        'particle_density': particle_cnt / norm,
        'ending_density': ending_cnt / norm,
        'text_len': text_len,
        'n_words': n_words
    }

# =============================================================================
# 2. 데이터 준비
# =============================================================================

print("📂 데이터 로딩 및 문단 분할...")
train_df = pd.read_csv(os.path.join(OPEN_DIR, 'train.csv'))
test_df = pd.read_csv(os.path.join(OPEN_DIR, 'test.csv'))

# Train 문단 분리
train_paras = []
for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Train Paragraph Parsing"):
    full_text = str(row['full_text'])
    paras = [p.strip() for p in full_text.split('\n') if p.strip()]
    for i, p in enumerate(paras):
        feat = extract_features(p)
        feat['doc_idx'] = idx
        feat['generated'] = row['generated']
        train_paras.append(feat)

train_para_df = pd.DataFrame(train_paras)

# Test 피처 추출
test_paras = []
for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Test Feature Extraction"):
    feat = extract_features(row['paragraph_text'])
    feat['ID'] = row['ID']
    feat['title'] = row['title']
    test_paras.append(feat)

test_para_df = pd.DataFrame(test_paras)

# =============================================================================
# 3. Stage 1: Paragraph Model Training (Noisy Label)
# =============================================================================

print("\n🚀 Stage 1: Paragraph-level Training...")
X_para = train_para_df.drop(['doc_idx', 'generated'], axis=1)
y_para = train_para_df['generated']

# GroupKFold (동일 문서의 문단이 섞이지 않도록)
gkf = GroupKFold(n_splits=5)
para_model = HistGradientBoostingClassifier(max_iter=300, random_state=42)

oof_para_preds = np.zeros(len(train_para_df))
for train_idx, val_idx in gkf.split(X_para, y_para, groups=train_para_df['doc_idx']):
    X_tr, X_val = X_para.iloc[train_idx], X_para.iloc[val_idx]
    y_tr, y_val = y_para.iloc[train_idx], y_para.iloc[val_idx]
    para_model.fit(X_tr, y_tr)
    oof_para_preds[val_idx] = para_model.predict_proba(X_val)[:, 1]

print(f"✅ Para-level OOF AUC: {roc_auc_score(y_para, oof_para_preds):.4f}")

# Re-train on full paragraph data
para_model.fit(X_para, y_para)

# =============================================================================
# 4. Stage 2 & 3: Pooling & Meta-Model
# =============================================================================

print("\n🚀 Stage 2 & 3: Pooling & Meta-Model Training...")
train_para_df['para_score'] = oof_para_preds

# 문서별 점수 풀링
doc_meta_feats = train_para_df.groupby('doc_idx')['para_score'].agg([
    ('max_score', 'max'),
    ('mean_score', 'mean'),
    ('std_score', 'std'),
    ('q75_score', lambda x: np.percentile(x, 75) if len(x)>0 else 0),
    ('min_score', 'min')
]).fillna(0)

doc_meta_feats['actual_label'] = train_df['generated']

# Meta-Model 학습 (문서 레벨)
X_meta = doc_meta_feats.drop('actual_label', axis=1)
y_meta = doc_meta_feats['actual_label']

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
meta_model = HistGradientBoostingClassifier(max_iter=100, random_state=42)

meta_cv_scores = []
for train_idx, val_idx in skf.split(X_meta, y_meta):
    X_tr, X_val = X_meta.iloc[train_idx], X_meta.iloc[val_idx]
    y_tr, y_val = y_meta.iloc[train_idx], y_meta.iloc[val_idx]
    meta_model.fit(X_tr, y_tr)
    val_probs = meta_model.predict_proba(X_val)[:, 1]
    meta_cv_scores.append(roc_auc_score(y_val, val_probs))

print(f"✅ Doc-level Meta AUC: {np.mean(meta_cv_scores):.4f}")

# Re-train meta model
meta_model.fit(X_meta, y_meta)

# =============================================================================
# 5. Inference & Score Refinement
# =============================================================================

print("\n🔮 Inference & Final Refinement...")
X_test_para = test_para_df.drop(['ID', 'title'], axis=1)
test_para_df['raw_score'] = para_model.predict_proba(X_test_para)[:, 1]

# Title별(문서별) 풀링 및 Meta-Model 적용
test_doc_groups = test_para_df.groupby('title')['raw_score'].agg([
    ('max_score', 'max'),
    ('mean_score', 'mean'),
    ('std_score', 'std'),
    ('q75_score', lambda x: np.percentile(x, 75) if len(x)>0 else 0),
    ('min_score', 'min')
]).fillna(0)

test_doc_groups['doc_refine_score'] = meta_model.predict_proba(test_doc_groups)[:, 1]

# 문단 점수 보정: 
# (문단 자체 점수) 와 (해당 문단이 속한 문서의 전체 점수)를 앙상블
test_para_df = test_para_df.merge(test_doc_groups[['doc_refine_score']], on='title', how='left')

# 최종 점수: 문단 점수와 문서 점수의 결합
# 문단 점수가 높으면서 속한 문서도 AI일 확률이 높을 때 시너지
test_para_df['final_score'] = (test_para_df['raw_score'] * 0.7) + (test_para_df['doc_refine_score'] * 0.3)

# =============================================================================
# 6. Submission 생성
# =============================================================================

submission = pd.DataFrame({
    'ID': test_para_df['ID'],
    'generated': test_para_df['final_score']
})

output_path = os.path.join(OUTPUT_DIR, 'submission_mil_refined.csv')
submission.to_csv(output_path, index=False)

print(f"\n✅ 완료! Submission 저장됨: {output_path}")
print(f"📊 최종 확률 분포: {submission['generated'].describe()}")
