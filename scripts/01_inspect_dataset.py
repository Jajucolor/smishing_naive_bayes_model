"""
1단계: meal-bbang/Korean_message 데이터셋 구조 확인

실행:
    python scripts/01_inspect_dataset.py
"""

from datasets import load_dataset


DATASET_NAME = "meal-bbang/Korean_message"


def shorten(text: str, max_len: int = 80) -> str:
    text = str(text).replace("\n", " ").strip()
    return text if len(text) <= max_len else text[:max_len] + "..."


def main() -> None:
    print(f"[1] 데이터셋 불러오는 중: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME)

    print("\n[2] 사용 가능한 split")
    print(ds)

    train = ds["train"]
    df = train.to_pandas()

    print("\n[3] 열 이름")
    print(list(df.columns))

    required_cols = {"content", "class"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"필수 열이 없습니다: {missing}")

    print("\n[4] 전체 데이터 개수")
    print(len(df))

    print("\n[5] 결측치 개수")
    print(df[["content", "class"]].isna().sum())

    print("\n[6] 라벨 분포")
    label_map = {1: "일반 문자", 2: "피싱/스미싱 문자"}
    label_counts = df["class"].value_counts().sort_index()
    for label, count in label_counts.items():
        print(f"{label} ({label_map.get(int(label), '알 수 없음')}): {count}개")

    print("\n[7] 문자 길이 통계")
    lengths = df["content"].astype(str).str.len()
    print(lengths.describe())

    print("\n[8] 샘플 5개")
    sample_df = df.sample(n=min(5, len(df)), random_state=42)
    for _, row in sample_df.iterrows():
        label = int(row["class"])
        print("-" * 60)
        print(f"라벨: {label} ({label_map.get(label, '알 수 없음')})")
        print(f"내용: {shorten(row['content'])}")


if __name__ == "__main__":
    main()
