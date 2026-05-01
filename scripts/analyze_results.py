"""
5단계: 모델 결과 분석

실행:
    python scripts/05_analyze_results.py

먼저 2단계 학습을 실행해야 한다.
    python scripts/02_train_model.py

이 파일이 하는 일:
1. 같은 방식으로 데이터셋을 불러오고 테스트 데이터를 만든다.
2. 저장된 모델로 테스트 데이터를 예측한다.
3. 잘못 분류한 예시를 CSV로 저장한다.
4. 스미싱 판단에 큰 영향을 준 단어와 일반 문자 판단에 큰 영향을 준 단어를 출력한다.
5. 보고서에 참고할 수 있는 분석 결과를 TXT로 저장한다.
"""

from pathlib import Path

import joblib
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


DATASET_NAME = "meal-bbang/Korean_message"

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "smishing_nb_model.joblib"
RESULT_DIR = ROOT_DIR / "results"

MISCLASSIFIED_PATH = RESULT_DIR / "misclassified_examples.csv"
TOP_FEATURES_PATH = RESULT_DIR / "top_features.txt"
ANALYSIS_PATH = RESULT_DIR / "result_analysis.txt"

LABEL_NAME = {
    1: "일반 문자",
    2: "피싱/스미싱 문자",
}


def load_and_clean_data() -> pd.DataFrame:
    ds = load_dataset(DATASET_NAME)
    df = ds["train"].to_pandas()

    df = df[["content", "class"]].copy()
    df = df.dropna(subset=["content", "class"])
    df["content"] = df["content"].astype(str).str.strip()
    df["class"] = df["class"].astype(int)
    df = df[df["content"] != ""]
    df = df[df["class"].isin([1, 2])]

    return df


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}\n"
            "먼저 python scripts/02_train_model.py 를 실행하세요."
        )

    return joblib.load(MODEL_PATH)


def get_test_data(df: pd.DataFrame):
    x = df["content"]
    y = df["class"]

    _, x_test, _, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return x_test, y_test


def save_misclassified_examples(x_test, y_test, y_pred, y_prob) -> pd.DataFrame:
    rows = []

    for text, actual, pred, prob in zip(x_test, y_test, y_pred, y_prob):
        actual = int(actual)
        pred = int(pred)

        if actual != pred:
            prob_map = {
                int(class_label): float(p)
                for class_label, p in prob.items()
            }
            rows.append(
                {
                    "content": text,
                    "actual_label": actual,
                    "actual_name": LABEL_NAME.get(actual, "알 수 없음"),
                    "predicted_label": pred,
                    "predicted_name": LABEL_NAME.get(pred, "알 수 없음"),
                    "prob_normal": prob_map.get(1, 0.0),
                    "prob_smishing": prob_map.get(2, 0.0),
                }
            )

    mis_df = pd.DataFrame(rows)
    mis_df.to_csv(MISCLASSIFIED_PATH, index=False, encoding="utf-8-sig")
    return mis_df


def extract_top_features(model, top_n: int = 30) -> str:
    """
    나이브 베이즈가 각 분류에서 중요하게 본 단어를 확인한다.

    CountVectorizer의 단어 목록과 MultinomialNB의 feature_log_prob_를 사용한다.
    feature_log_prob_는 각 분류에서 특정 단어가 등장할 로그 확률이다.
    """
    vectorizer = model.named_steps["vectorizer"]
    nb = model.named_steps["nb"]

    feature_names = vectorizer.get_feature_names_out()
    class_labels = nb.classes_

    lines = []
    lines.append("나이브 베이즈 주요 단어 분석")
    lines.append("=" * 60)
    lines.append(
        "아래 단어들은 각 분류에서 등장할 확률이 상대적으로 높게 학습된 단어들이다."
    )
    lines.append("단어 자체가 무조건 위험하다는 뜻은 아니며, 데이터셋 안에서의 통계적 경향을 의미한다.")
    lines.append("")

    for class_index, class_label in enumerate(class_labels):
        class_label = int(class_label)
        class_name = LABEL_NAME.get(class_label, str(class_label))

        log_probs = nb.feature_log_prob_[class_index]
        top_indices = log_probs.argsort()[-top_n:][::-1]

        lines.append(f"[{class_label}: {class_name}에서 확률이 높은 단어]")
        for rank, idx in enumerate(top_indices, start=1):
            word = feature_names[idx]
            score = log_probs[idx]
            lines.append(f"{rank:02d}. {word}  (log probability: {score:.4f})")
        lines.append("")

    text = "\n".join(lines)
    TOP_FEATURES_PATH.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1] 데이터셋 불러오기")
    df = load_and_clean_data()
    print(f"전체 데이터 개수: {len(df)}")

    print("\n[2] 저장된 모델 불러오기")
    model = load_model()
    print(f"모델: {MODEL_PATH}")

    print("\n[3] 테스트 데이터 구성")
    x_test, y_test = get_test_data(df)
    print(f"테스트 데이터 개수: {len(x_test)}")

    print("\n[4] 예측")
    y_pred = model.predict(x_test)

    classes = list(model.classes_)
    raw_probs = model.predict_proba(x_test)
    y_prob = [
        {
            int(class_label): prob
            for class_label, prob in zip(classes, probs)
        }
        for probs in raw_probs
    ]

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        labels=[1, 2],
        target_names=["일반 문자", "피싱/스미싱 문자"],
        digits=4,
    )
    matrix = confusion_matrix(y_test, y_pred, labels=[1, 2])

    print(f"\n정확도: {accuracy:.4f}")
    print("\n분류 리포트")
    print(report)
    print("\n혼동행렬")
    print(matrix)

    print("\n[5] 오분류 예시 저장")
    mis_df = save_misclassified_examples(x_test, y_test, y_pred, y_prob)
    print(f"오분류 개수: {len(mis_df)}")
    print(f"저장 위치: {MISCLASSIFIED_PATH}")

    print("\n[6] 주요 단어 분석")
    top_features_text = extract_top_features(model, top_n=30)
    print(f"저장 위치: {TOP_FEATURES_PATH}")

    print("\n[7] 분석 요약 저장")
    lines = []
    lines.append("한국어 스미싱 문자 탐지 모델 결과 분석")
    lines.append("=" * 60)
    lines.append(f"데이터셋: {DATASET_NAME}")
    lines.append(f"테스트 데이터 개수: {len(x_test)}")
    lines.append(f"정확도: {accuracy:.4f}")
    lines.append(f"오분류 개수: {len(mis_df)}")
    lines.append("")
    lines.append("[혼동행렬]")
    lines.append("행: 실제값, 열: 예측값")
    lines.append("라벨 순서: 1=일반 문자, 2=피싱/스미싱 문자")
    lines.append(str(matrix))
    lines.append("")
    lines.append("[분류 리포트]")
    lines.append(report)
    lines.append("")
    lines.append("[해석 가이드]")
    lines.append(
        "- 정확도는 전체 테스트 문자 중 모델이 라벨을 맞힌 비율이다."
    )
    lines.append(
        "- 혼동행렬의 왼쪽 위는 실제 일반 문자를 일반 문자로 맞힌 개수이다."
    )
    lines.append(
        "- 오른쪽 아래는 실제 스미싱 문자를 스미싱 문자로 맞힌 개수이다."
    )
    lines.append(
        "- 실제 스미싱인데 일반 문자로 예측한 경우는 보안상 더 위험한 오류이다."
    )
    lines.append(
        "- 실제 일반 문자인데 스미싱으로 예측한 경우는 사용자 불편을 만들 수 있는 오류이다."
    )
    lines.append("")
    lines.append("[생성 파일]")
    lines.append(f"- {MISCLASSIFIED_PATH}")
    lines.append(f"- {TOP_FEATURES_PATH}")

    ANALYSIS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"저장 위치: {ANALYSIS_PATH}")

    print("\n완료")


if __name__ == "__main__":
    main()
