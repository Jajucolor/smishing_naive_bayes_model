# 한국어 스미싱 문자 탐지 - 나이브 베이즈 분류기 프로젝트

## 주제
나이브 베이즈 분류기를 이용한 한국어 스미싱 문자 탐지 프로그램

## 목표
한국어 문자 메시지를 입력하면 해당 문자가 `일반 문자`인지 `피싱/스미싱 의심 문자`인지 확률적으로 분류한다.

## 사용할 데이터셋
- Hugging Face: `meal-bbang/Korean_message`
- 주요 열
  - `content`: 문자 메시지 내용
  - `class`: 라벨
    - 1: 일반 문자
    - 2: 피싱/스미싱 문자

## 설치
```bash
pip install -r requirements.txt
```

## 실행 순서

### 1단계: 데이터셋 구조 확인
```bash
python scripts/inspect_dataset.py
```

### 2단계: 모델 학습
```bash
python scripts/train_model.py
```

학습이 끝나면 다음 파일이 생성된다.

```text
models/smishing_nb_model.joblib
results/evaluation.txt
```

### 3단계: 직접 문자 입력 예측
```bash
python scripts/predict_message.py
```

### 4단계: 수학 원리 직접 계산 예시
```bash
python scripts/manual_naive_bayes_example.py
```

### 5단계: 결과 분석
```bash
python scripts/analyze_results.py
```

분석이 끝나면 다음 파일이 생성된다.

```text
results/misclassified_examples.csv
results/top_features.txt
results/result_analysis.txt
```

