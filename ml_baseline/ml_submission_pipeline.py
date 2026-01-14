"""
ML 기반 AI/Human 구분 Submission 파이프라인
===========================================
1. train.csv에서 문단별 메타 피처 추출 (문서 → 문단 분할)
2. 문단 피처를 문서 레벨로 풀링
3. HistGradientBoosting 학습
4. test.csv 문단별 피처 추출 후 예측
5. sample_submission.csv 형식으로 저장
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import re
from collections import Counter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. 메타 피처 추출 함수들
# =============================================================================

# 기능어 패턴 (조사, 어미)
PARTICLES = ['은', '는', '이', '가', '을', '를', '에', '에서', '으로', '로', '와', '과', '의', '도', '만', '까지', '부터', '에게', '한테', '께']
ENDINGS = ['다', '며', '고', '지만', '는데', '면서', '지', '니', '라', '자', '려고', '도록', '듯이', '처럼']

def extract_paragraph_features(text):
    """문단에서 메타 피처 추출"""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {
            'sent_len_median': 0, 'sent_len_p90': 0, 'sent_len_std': 0, 'sent_len_cv': 0,
            'comma_density': 0, 'repeat_ratio': 0, 'ttr': 1,
            'particle_per_100char': 0, 'ending_per_100char': 0, 'funcword_per_100char': 0,
            'para_len': 0, 'n_sentences': 0, 'excl_cnt': 0, 'quest_cnt': 0
        }
    
    text = text.strip()
    para_len = len(text)
    
    # 문장 분할
    sentences = [s.strip() for s in re.split(r'[.!?。]\s*', text) if s.strip()]
    if len(sentences) == 0:
        sentences = [text]
    
    sent_lengths = [len(s) for s in sentences]
    
    # 문장 길이 통계
    sent_len_median = np.median(sent_lengths) if sent_lengths else 0
    sent_len_p90 = np.percentile(sent_lengths, 90) if len(sent_lengths) >= 2 else sent_len_median
    sent_len_std = np.std(sent_lengths) if len(sent_lengths) >= 2 else 0
    sent_len_cv = sent_len_std / (sent_len_median + 1e-6)
    
    # 쉼표 밀도 (100자당)
    comma_cnt = text.count(',') + text.count('，')
    comma_density = comma_cnt / (para_len / 100) if para_len > 0 else 0
    
    # 구두점 카운트
    excl_cnt = text.count('!')
    quest_cnt = text.count('?')
    
    # 반복 비율 & TTR (어휘 다양성)
    words = text.split()
    if len(words) > 0:
        unique_words = set(words)
        repeat_ratio = 1 - (len(unique_words) / len(words))
        ttr = len(unique_words) / len(words)
    else:
        repeat_ratio = 0
        ttr = 1
    
    # 기능어 밀도
    particle_cnt = sum(text.count(p) for p in PARTICLES)
    ending_cnt = sum(text.count(e) for e in ENDINGS)
    
    particle_per_100char = particle_cnt / (para_len / 100) if para_len > 0 else 0
    ending_per_100char = ending_cnt / (para_len / 100) if para_len > 0 else 0
    funcword_per_100char = particle_per_100char + ending_per_100char
    
    return {
        'sent_len_median': sent_len_median,
        'sent_len_p90': sent_len_p90,
        'sent_len_std': sent_len_std,
        'sent_len_cv': sent_len_cv,
        'comma_density': comma_density,
        'repeat_ratio': repeat_ratio,
        'ttr': ttr,
        'particle_per_100char': particle_per_100char,
        'ending_per_100char': ending_per_100char,
        'funcword_per_100char': funcword_per_100char,
        'para_len': para_len,
        'n_sentences': len(sentences),
        'excl_cnt': excl_cnt,
        'quest_cnt': quest_cnt
    }

def pool_paragraph_features(para_features_list):
    """문단 피처들을 문서 레벨로 풀링 (mean, std, max, min)"""
    if len(para_features_list) == 0:
        return {}
    
    df = pd.DataFrame(para_features_list)
    pooled = {}
    
    for col in df.columns:
        pooled[f'{col}_mean'] = df[col].mean()
        pooled[f'{col}_std'] = df[col].std() if len(df) > 1 else 0
        pooled[f'{col}_max'] = df[col].max()
        pooled[f'{col}_min'] = df[col].min()
        pooled[f'{col}_median'] = df[col].median()
    
    # 추가 문서 레벨 피처
    pooled['n_paragraphs'] = len(para_features_list)
    pooled['total_len'] = sum(pf['para_len'] for pf in para_features_list)
    
    return pooled

# =============================================================================
# 2. Train 데이터 처리
# =============================================================================
print("📂 Train 데이터 로딩...")
train_df = pd.read_csv('/Users/youngjinson/멋사1/AI-Human-Distinction/open/train.csv')
print(f"Train 문서 수: {len(train_df):,}")

print("\n🔧 Train 피처 추출 (문서 → 문단 분할 → 풀링)...")
train_features = []


for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Train 피처 추출"):
    full_text = row['full_text']
    if not isinstance(full_text, str):
        full_text = ""
    
    # 문단 분할 (줄바꿈 기준)
    paragraphs = [p.strip() for p in full_text.split('\n') if p.strip()]
    
    # 각 문단에서 피처 추출
    para_features_list = [extract_paragraph_features(p) for p in paragraphs]
    
    # 문서 레벨로 풀링
    if len(para_features_list) > 0:
        doc_features = pool_paragraph_features(para_features_list)
    else:
        doc_features = pool_paragraph_features([extract_paragraph_features("")])
    
    doc_features['generated'] = row['generated']
    train_features.append(doc_features)

train_feat_df = pd.DataFrame(train_features)
print(f"✅ Train 피처 추출 완료: {train_feat_df.shape}")

# =============================================================================
# 3. Test 데이터 처리
# =============================================================================
print("\n📂 Test 데이터 로딩...")
test_df = pd.read_csv('/Users/youngjinson/멋사1/AI-Human-Distinction/open/test.csv')
print(f"Test 문단 수: {len(test_df):,}")

print("\n🔧 Test 피처 추출 (문단별)...")
test_features = []

for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Test 피처 추출"):
    para_text = row['paragraph_text']
    features = extract_paragraph_features(para_text)
    features['ID'] = row['ID']
    features['title'] = row['title']
    features['paragraph_index'] = row['paragraph_index']
    test_features.append(features)

test_feat_df = pd.DataFrame(test_features)
print(f"✅ Test 피처 추출 완료: {test_feat_df.shape}")

# =============================================================================
# 4. 학습 및 예측
# =============================================================================
print("\n" + "="*60)
print("🚀 모델 학습")
print("="*60)

# 피처 컬럼 선택 (숫자 컬럼만)
feature_cols = [c for c in train_feat_df.columns if c != 'generated' and train_feat_df[c].dtype in ['float64', 'int64']]
print(f"피처 수: {len(feature_cols)}")

X_train = train_feat_df[feature_cols].copy()
y_train = train_feat_df['generated'].copy()

# 무한값/결측치 처리
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_train = X_train.fillna(0)

# 5-Fold CV로 학습
print("\n📊 5-Fold Cross Validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
oof_preds = np.zeros(len(X_train))

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    model = HistGradientBoostingClassifier(
        max_iter=200, 
        max_depth=8, 
        learning_rate=0.05, 
        random_state=42
    )
    model.fit(X_tr, y_tr)
    
    val_proba = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_proba
    
    auc = roc_auc_score(y_val, val_proba)
    cv_scores.append(auc)
    print(f"  Fold {fold+1}: AUC = {auc:.4f}")

print(f"\n📊 CV 평균 AUC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores)*2:.4f})")

# 전체 데이터로 최종 모델 학습
print("\n🎯 최종 모델 학습...")
final_model = HistGradientBoostingClassifier(
    max_iter=200, 
    max_depth=8, 
    learning_rate=0.05, 
    random_state=42
)
final_model.fit(X_train, y_train)

# =============================================================================
# 5. Test 예측 (문단별 → 문서 풀링 필요 없이 문단별 예측)
# =============================================================================
print("\n🔮 Test 예측...")

# Test 피처 준비
test_feature_cols = [c for c in feature_cols if c in test_feat_df.columns]
missing_cols = [c for c in feature_cols if c not in test_feat_df.columns]

# 풀링 피처 (_mean, _std 등)는 test에 없으므로, 문단 단일 피처 사용
# 문단 피처를 그대로 사용 (mean 컬럼은 해당 문단의 값으로 대체)
X_test = pd.DataFrame()

# 기본 문단 피처들
basic_features = ['sent_len_median', 'sent_len_p90', 'sent_len_std', 'sent_len_cv',
                  'comma_density', 'repeat_ratio', 'ttr', 
                  'particle_per_100char', 'ending_per_100char', 'funcword_per_100char',
                  'para_len', 'n_sentences', 'excl_cnt', 'quest_cnt']

for col in feature_cols:
    if col in test_feat_df.columns:
        X_test[col] = test_feat_df[col]
    else:
        # 풀링 피처의 경우 기본 피처에서 매핑
        base_col = col.replace('_mean', '').replace('_std', '').replace('_max', '').replace('_min', '').replace('_median', '')
        if base_col in test_feat_df.columns:
            if '_std' in col:
                X_test[col] = 0  # 단일 문단이므로 std=0
            elif '_mean' in col or '_median' in col or '_max' in col or '_min' in col:
                X_test[col] = test_feat_df[base_col]
        elif col == 'n_paragraphs':
            X_test[col] = 1  # 문단 단위이므로 1
        elif col == 'total_len':
            X_test[col] = test_feat_df['para_len']
        else:
            X_test[col] = 0

X_test = X_test.replace([np.inf, -np.inf], np.nan)
X_test = X_test.fillna(0)

# 예측
test_proba = final_model.predict_proba(X_test)[:, 1]

# =============================================================================
# 6. Submission 파일 생성
# =============================================================================
print("\n📝 Submission 파일 생성...")

submission = pd.DataFrame({
    'ID': test_feat_df['ID'],
    'generated': test_proba  # 확률값으로 출력
})

# 저장
output_path = '/Users/youngjinson/멋사1/AI-Human-Distinction/ml_baseline/submission_ml_baseline.csv'
submission.to_csv(output_path, index=False)

print(f"\n✅ 저장 완료: {output_path}")
print(f"   평균 AI 확률: {test_proba.mean():.4f}")
print(f"   확률 분포: min={test_proba.min():.4f}, max={test_proba.max():.4f}")

# 샘플 확인
print("\n📋 예측 샘플:")
print(submission.head(10))
