"""
2단계: 나이브 베이즈 모델 학습

실행:
    python scripts/02_train_model.py
"""

from pathlib import Path

import joblib
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


DATASET_NAME = "meal-bbang/Korean_message"

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "smishing_nb_model.joblib"
RESULT_PATH = ROOT_DIR / "results" / "evaluation.txt"

LABEL_NAME = {
    1: "일반 문자",
    2: "피싱/스미싱 문자",
}


def load_and_clean_data() -> pd.DataFrame:
    print(f"[1] 데이터셋 불러오는 중: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME)
    df = ds["train"].to_pandas()

    required_cols = {"content", "class"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"필수 열이 없습니다: {missing}")

    df = df[["content", "class"]].copy()
    df = df.dropna(subset=["content", "class"])
    df["content"] = df["content"].astype(str).str.strip()
    df["class"] = df["class"].astype(int)

    df = df[df["content"] != ""]
    df = df[df["class"].isin([1, 2])]

    if len(df) == 0:
        raise ValueError("학습 가능한 데이터가 없습니다.")

    return df


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "vectorizer",
                CountVectorizer(
                    token_pattern=r"(?u)\b\w+\b",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=30000,
                ),
            ),
            ("nb", MultinomialNB(alpha=1.0)),
        ]
    )


def save_evaluation_text(
    accuracy: float,
    report: str,
    matrix,
    train_size: int,
    test_size: int,
    label_counts: pd.Series,
) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("한국어 스미싱 문자 탐지 - 나이브 베이즈 모델 평가 결과")
    lines.append("=" * 60)
    lines.append(f"데이터셋: {DATASET_NAME}")
    lines.append(f"학습 데이터 개수: {train_size}")
    lines.append(f"테스트 데이터 개수: {test_size}")
    lines.append("")
    lines.append("[라벨 분포]")
    for label, count in label_counts.sort_index().items():
        lines.append(f"{label} ({LABEL_NAME.get(int(label), '알 수 없음')}): {count}개")
    lines.append("")
    lines.append(f"[정확도] {accuracy:.4f}")
    lines.append("")
    lines.append("[분류 리포트]")
    lines.append(report)
    lines.append("")
    lines.append("[혼동행렬]")
    lines.append("행: 실제값, 열: 예측값")
    lines.append("라벨 순서: 1=일반 문자, 2=피싱/스미싱 문자")
    lines.append(str(matrix))

    RESULT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = load_and_clean_data()

    print("\n[2] 데이터 기본 정보")
    print(f"전체 데이터 개수: {len(df)}")
    print("라벨 분포:")
    print(df["class"].value_counts().sort_index())

    x = df["content"]
    y = df["class"]

    print("\n[3] 학습용/테스트용 데이터 분리")
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(f"학습 데이터: {len(x_train)}개")
    print(f"테스트 데이터: {len(x_test)}개")

    print("\n[4] 모델 생성 및 학습")
    model = build_model()
    model.fit(x_train, y_train)

    print("\n[5] 테스트 데이터 예측")
    y_pred = model.predict(x_test)

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
    print("행: 실제값, 열: 예측값")
    print("라벨 순서: 1=일반 문자, 2=피싱/스미싱 문자")
    print(matrix)

    print("\n[6] 모델 저장")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"저장 완료: {MODEL_PATH}")

    print("\n[7] 평가 결과 저장")
    save_evaluation_text(
        accuracy=accuracy,
        report=report,
        matrix=matrix,
        train_size=len(x_train),
        test_size=len(x_test),
        label_counts=df["class"].value_counts(),
    )
    print(f"저장 완료: {RESULT_PATH}")

    print("\n[8] 예시 문자 예측")
    examples = [
        "오늘 저녁 7시에 회의 있습니다.",
        "고객님의 계좌가 정지 예정입니다. 본인인증 링크를 확인하세요.",
        "택배 배송 주소 오류로 반송 예정입니다. 아래 링크에서 주소를 수정하세요.",
        "내일 점심 같이 먹을래?",
    ]

    probs = model.predict_proba(examples)
    preds = model.predict(examples)
    class_order = model.classes_

    for text, pred, prob in zip(examples, preds, probs):
        print("-" * 60)
        print(f"문자: {text}")
        print(f"예측: {pred} ({LABEL_NAME.get(int(pred), '알 수 없음')})")
        for class_label, p in zip(class_order, prob):
            print(f"  P({LABEL_NAME.get(int(class_label), class_label)} | 문자) = {p:.4f}")


if __name__ == "__main__":
    main()
