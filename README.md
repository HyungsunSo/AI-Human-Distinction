# 프로젝트 개요

### 프로젝트 주제
AI가 작성한 글과 Human이 작성한 글에 대해 AI모델이 어떤 근거에서 AI인지 판단하는 모델을 구축한 프로젝트입니다.(AI-Human-Distinction)

### 데이터 및 프로젝트 출처
[DACON 2025 SW중심대학 디지털 경진대회 : AI부문](https://dacon.io/competitions/official/236473/overview/description)

<br>

---

# 프로젝트 설명

### 프로젝트 결과


### 프로젝트 구조

```
.
├── colab_notebooks/ # Google Colab 실험 및 분석 노트북
│   ├── bert_paragraph_classifier.ipynb
│   ├── explainability_analysis_colab.ipynb
│   ├── gpt_oss_synthetic_pairs.ipynb
│   ├── paragraph_maxpool_colab.ipynb
│   ├── style_trajectory_analysis.ipynb
│   └── README.md
│
├── data/ # 학습 및 평가 데이터
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── data_generation/ # 데이터 생성 관련 노트북
│   ├── ai_generation.ipynb
│   ├── hyperclova_generation.ipynb
│   ├── paragraph_splitting.ipynb
│   └── MODEL_GUIDE.md
│
├── outputs/ # 모델 출력 및 결과물
│
├── app.py # 애플리케이션 엔트리포인트
├── main.py # 메인 실행 스크립트
├── train.py # 모델 학습 로직
├── dataset.py # Dataset / DataLoader 정의
├── utils.py # 공용 유틸리티 함수
├── exaone.py # EXAONE 관련 로직
├── note.ipynb # 임시 실험용 노트북
├── note.py # 임시 테스트 스크립트
│
├── config.yaml # 전체 설정 파일
├── README.md  # 프로젝트 개요
├── .gitignore
└── .github/ # GitHub 설정
    ├── ISSUE_TEMPLATE/
    └── PULL_REQUEST_TEMPLATE.md
```

## 기술 스택
- Python
- Machine Learning / Deep Learning

## 업데이트
- 2026-01-12: 프로젝트 README 업데이트
