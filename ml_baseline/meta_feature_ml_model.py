"""
Meta Feature 기반 AI/Human 구분 ML 모델
- 텍스트 임베딩 없이 EDA에서 발견한 메타 피처만 사용
- 빠른 학습과 해석 가능성에 초점
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. 데이터 로드 및 피처 선택
# =============================================================================
print("📂 데이터 로딩 중...")
df = pd.read_csv('/Users/youngjinson/멋사1/AI-Human-Distinction/open/train_with_all_features.csv')

print(f"전체 데이터: {len(df):,}개")
print(f"컬럼: {df.columns.tolist()}")

# EDA 분석 결과 기반 메타 피처 선택
META_FEATURES = [
    # 반복 표현 / 어휘 다양성
    'repeat_ratio_mean',      # Human > AI (어휘 반복 많음)
    'repeat_ratio_p90',
    'ttr_doc',                # Type-Token Ratio
    
    # 기능어 밀도 (조사, 어미)
    'particle_per_100char',   # Human > AI
    'ending_per_100char',     # Human > AI  
    'funcword_per_100char',   # Human > AI
    
    # 문서/문장 길이 관련
    'doc_len',                # 문서 전체 길이
    'sent_len_median',        # 문장 중앙값 (Human > AI)
    'sent_len_p90',           # 상위 10% 문장 길이
    'sent_len_std',           # 표준편차 (Human 변동성 큼)
    
    # 구두점 사용
    'comma_density',          # 100자당 쉼표 (Human > AI)
    
    # 문단 구조
    'n_paragraphs',           # 문단 수
]

# clipped 버전도 추가 (극단값 영향 제거)
CLIPPED_FEATURES = [
    'repeat_ratio_mean_clipped',
    'ttr_doc_clipped',
    'particle_per_100char_clipped',
    'ending_per_100char_clipped',
    'funcword_per_100char_clipped',
]

# 사용할 피처 최종 리스트
FEATURES_TO_USE = [f for f in META_FEATURES + CLIPPED_FEATURES if f in df.columns]
print(f"\n✅ 사용할 피처 ({len(FEATURES_TO_USE)}개): {FEATURES_TO_USE}")

# =============================================================================
# 2. 피처 엔지니어링 - 추가 파생 피처
# =============================================================================
print("\n🔧 피처 엔지니어링...")

# 문장 길이 변동계수 (CV = std / median)
if 'sent_len_std' in df.columns and 'sent_len_median' in df.columns:
    df['sent_len_cv'] = df['sent_len_std'] / (df['sent_len_median'] + 1e-6)
    FEATURES_TO_USE.append('sent_len_cv')

# 기능어 비율 (조사 vs 어미)
if 'particle_per_100char' in df.columns and 'ending_per_100char' in df.columns:
    df['particle_ending_ratio'] = df['particle_per_100char'] / (df['ending_per_100char'] + 1e-6)
    FEATURES_TO_USE.append('particle_ending_ratio')

# 문서 길이 대비 문단 수 비율
if 'n_paragraphs' in df.columns and 'doc_len' in df.columns:
    df['para_density'] = df['n_paragraphs'] / (df['doc_len'] + 1e-6) * 1000  # 1000자당 문단 수
    FEATURES_TO_USE.append('para_density')

print(f"✅ 파생 피처 추가 후 총 {len(FEATURES_TO_USE)}개 피처")

# =============================================================================
# 3. 데이터 준비
# =============================================================================
# 결측치 처리
X = df[FEATURES_TO_USE].copy()
y = df['generated'].copy()

# 무한값 처리
X = X.replace([np.inf, -np.inf], np.nan)

# 결측치 중앙값 대체
for col in X.columns:
    if X[col].isna().sum() > 0:
        X[col].fillna(X[col].median(), inplace=True)

print(f"\n📊 피처 통계:")
print(X.describe().T)

# Train/Test 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n📊 Train: {len(X_train):,} / Test: {len(X_test):,}")
print(f"클래스 분포 - Human(0): {(y_train==0).sum():,} / AI(1): {(y_train==1).sum():,}")

# 스케일링 (로지스틱 회귀용)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# 4. 모델 학습 및 비교
# =============================================================================
print("\n" + "="*60)
print("🚀 모델 학습 시작")
print("="*60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=100, max_depth=6, learning_rate=0.1, random_state=42),
}

results = {}

for name, model in models.items():
    print(f"\n📌 {name}")
    
    # 스케일링 필요 여부에 따라 데이터 선택
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    
    # 평가
    auc = roc_auc_score(y_test, y_proba)
    results[name] = {
        'model': model,
        'auc': auc,
        'y_pred': y_pred,
        'y_proba': y_proba
    }
    
    print(f"   ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))

# =============================================================================
# 5. 최고 모델 선택 및 상세 분석
# =============================================================================
best_model_name = max(results, key=lambda x: results[x]['auc'])
best_result = results[best_model_name]

print("\n" + "="*60)
print(f"🏆 최고 모델: {best_model_name} (AUC: {best_result['auc']:.4f})")
print("="*60)

# 혼동 행렬
print("\n📊 혼동 행렬:")
cm = confusion_matrix(y_test, best_result['y_pred'])
print(f"           Predicted")
print(f"           Human  AI")
print(f"Actual Human  {cm[0,0]:>5}  {cm[0,1]:>5}")
print(f"       AI     {cm[1,0]:>5}  {cm[1,1]:>5}")

# 피처 중요도 (트리 기반 모델인 경우)
if best_model_name == 'Random Forest':
    print(f"\n📈 {best_model_name} 피처 중요도:")
    importance = best_result['model'].feature_importances_
    feat_importance = pd.DataFrame({
        'feature': FEATURES_TO_USE,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    for idx, row in feat_importance.head(10).iterrows():
        print(f"   {row['feature']:<30} : {row['importance']:.4f}")

elif best_model_name == 'HistGradientBoosting':
    from sklearn.inspection import permutation_importance
    print(f"\n📈 {best_model_name} Permutation 피처 중요도:")
    perm_importance = permutation_importance(best_result['model'], X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    feat_importance = pd.DataFrame({
        'feature': FEATURES_TO_USE,
        'importance': perm_importance.importances_mean
    }).sort_values('importance', ascending=False)
    
    for idx, row in feat_importance.head(10).iterrows():
        print(f"   {row['feature']:<30} : {row['importance']:.4f}")

# 로지스틱 회귀 계수 (해석용)
if 'Logistic Regression' in results:
    print(f"\n📈 Logistic Regression 계수 (절대값 기준 정렬):")
    lr_model = results['Logistic Regression']['model']
    coef_df = pd.DataFrame({
        'feature': FEATURES_TO_USE,
        'coef': lr_model.coef_[0]
    })
    coef_df['abs_coef'] = coef_df['coef'].abs()
    coef_df = coef_df.sort_values('abs_coef', ascending=False)
    
    for idx, row in coef_df.head(10).iterrows():
        direction = "→ AI" if row['coef'] > 0 else "→ Human"
        print(f"   {row['feature']:<30} : {row['coef']:>+.4f} {direction}")

# =============================================================================
# 6. Cross-Validation 최종 검증
# =============================================================================
print("\n" + "="*60)
print("🔄 5-Fold Cross-Validation")
print("="*60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name in ['Random Forest', 'HistGradientBoosting']:
    model = models[name]
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    print(f"{name:<20}: AUC = {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

print("\n✅ 완료!")
