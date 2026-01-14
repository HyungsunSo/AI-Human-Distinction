"""
Meta-Feature Analyzer for AI Text Detection
===========================================
EDA 결과를 기반으로 입력된 텍스트의 문체적 특징을 
Human vs AI 분포와 통계적으로 비교합니다.
"""

import numpy as np
import re
from scipy.stats import norm
from .schemas import MetaAnalysisResult, MetaFeatureResult, FeatureStats

# train_with_all_features.csv에서 추출한 전역 통계치 (EDA 결과)
GLOBAL_STATS = {
    'sent_len_median': {
        'display_name': '문장 길이 중앙값',
        'human': {'mean': 13.545, 'std': 5.670},
        'ai': {'mean': 11.802, 'std': 2.751}
    },
    'comma_density': {
        'display_name': '쉼표 밀도 (100자당)',
        'human': {'mean': 1.107, 'std': 0.602},
        'ai': {'mean': 0.721, 'std': 0.493}
    },
    'particle_per_100char': {
        'display_name': '조사 밀도 (100자당)',
        'human': {'mean': 10.607, 'std': 1.528},
        'ai': {'mean': 10.639, 'std': 1.364}
    },
    'ending_per_100char': {
        'display_name': '어미 밀도 (100자당)',
        'human': {'mean': 3.560, 'std': 0.747},
        'ai': {'mean': 3.428, 'std': 0.682}
    },
    'repeat_ratio_mean': {
        'display_name': '어휘 반복률',
        'human': {'mean': 0.016, 'std': 0.015},
        'ai': {'mean': 0.011, 'std': 0.010}
    }
}

# 한국어 기능어 패턴
PARTICLES = ['은', '는', '이', '가', '을', '를', '에', '에서', '으로', '로', '와', '과', '의', '도', '만', '까지', '부터', '에게', '한테', '께']
ENDINGS = ['다', '며', '고', '지만', '는데', '면서', '지', '니', '라', '자', '려고', '도록', '듯이', '처럼']

class MetaAnalyzer:
    def extract_features(self, text: str) -> dict:
        """입력 텍스트에서 메타 피처 추출"""
        text = text.strip()
        if not text:
            return {f: 0 for f in GLOBAL_STATS.keys()}

        text_len = len(text)
        words = text.split()
        n_words = len(words)
        
        # 문장 분할 및 통계
        sentences = [s.strip() for s in re.split(r'[.!?。]\s*', text) if s.strip()]
        sent_lengths = [len(s) for s in sentences] if sentences else [0]
        
        # 어휘 다양성 (반복률)
        unique_words = set(words)
        repeat_ratio = 1 - (len(unique_words) / n_words) if n_words > 0 else 0
        
        # 정규화 기준 (100자당)
        norm_val = text_len / 100 if text_len > 0 else 1
        
        # 구두점 및 기능어
        comma_cnt = text.count(',') + text.count('，')
        particle_cnt = sum(text.count(p) for p in PARTICLES)
        ending_cnt = sum(text.count(e) for e in ENDINGS)
        
        return {
            'sent_len_median': np.median(sent_lengths),
            'comma_density': comma_cnt / norm_val,
            'particle_per_100char': particle_cnt / norm_val,
            'ending_per_100char': ending_cnt / norm_val,
            'repeat_ratio_mean': repeat_ratio
        }

    def analyze(self, text: str) -> MetaAnalysisResult:
        """통계적 비교 및 p-value 산출"""
        current_features = self.extract_features(text)
        results = []
        ai_signals = 0
        human_signals = 0
        
        for feat_name, stats in GLOBAL_STATS.items():
            val = current_features[feat_name]
            h_mean, h_std = stats['human']['mean'], stats['human']['std']
            a_mean, a_std = stats['ai']['mean'], stats['ai']['std']
            
            # p-value 계산 (Human 분포 하에서 현재 값이 관측될 확률)
            # z-score 기준
            z_human = (val - h_mean) / (h_std + 1e-6)
            p_val = 2 * (1 - norm.cdf(abs(z_human))) # Two-tailed
            
            # 해석 생성
            closer_to = "AI" if abs(val - a_mean) < abs(val - h_mean) else "Human"
            if closer_to == "AI": ai_signals += 1
            else: human_signals += 1
            
            interpretation = f"이 피처는 {closer_to} 그룹의 평균치에 더 가깝습니다."
            if p_val < 0.05:
                interpretation += f" (통계적으로 {h_mean:.2f}인 Human 분포에서 벗어나 있습니다)"

            results.append(MetaFeatureResult(
                feature_name=feat_name,
                display_name=stats['display_name'],
                value=float(val),
                human_stats=FeatureStats(mean=h_mean, std=h_std),
                ai_stats=FeatureStats(mean=a_mean, std=a_std),
                p_value=float(p_val),
                interpretation=interpretation
            ))
            
        overall = f"문체 분석 결과, 주요 {len(GLOBAL_STATS)}개 지표 중 {ai_signals}개가 AI 패턴을, {human_signals}개가 Human 패턴을 보입니다."
        
        return MetaAnalysisResult(
            features=results,
            overall_interpretation=overall
        )

# Singleton instance
meta_analyzer = MetaAnalyzer()
