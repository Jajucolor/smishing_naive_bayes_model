"""
3단계: 직접 입력한 문자 메시지 예측

실행:
    python scripts/03_predict_message.py
"""

from pathlib import Path

import joblib


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "smishing_nb_model.joblib"

LABEL_NAME = {
    1: "일반 문자",
    2: "피싱/스미싱 문자",
}

EXIT_WORDS = {"q", "quit", "exit", "종료"}


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}\n"
            "먼저 다음 명령어를 실행하세요.\n"
            "python scripts/02_train_model.py"
        )

    return joblib.load(MODEL_PATH)


def predict_one(model, message: str) -> None:
    message = message.strip()

    if not message:
        print("빈 문자는 예측할 수 없습니다.")
        return

    pred = model.predict([message])[0]
    probs = model.predict_proba([message])[0]
    class_order = model.classes_

    print("\n[입력 문자]")
    print(message)

    print("\n[예측 결과]")
    print(f"{LABEL_NAME.get(int(pred), '알 수 없음')}")

    print("\n[예측 확률]")
    for class_label, prob in zip(class_order, probs):
        print(f"{LABEL_NAME.get(int(class_label), class_label)}: {prob:.4f}")

    smishing_prob = 0.0
    normal_prob = 0.0

    for class_label, prob in zip(class_order, probs):
        if int(class_label) == 1:
            normal_prob = prob
        elif int(class_label) == 2:
            smishing_prob = prob

    print("\n[해석]")
    if int(pred) == 2:
        print(
            f"이 문자는 스미싱 의심 확률이 {smishing_prob:.2%}로 더 높게 계산되었습니다."
        )
    else:
        print(
            f"이 문자는 일반 문자 확률이 {normal_prob:.2%}로 더 높게 계산되었습니다."
        )

    print("-" * 60)


def main() -> None:
    print("한국어 스미싱 문자 탐지 프로그램")
    print("=" * 60)
    print("나이브 베이즈 분류기를 이용해 문자를 분류합니다.")
    print("종료하려면 q, quit, exit, 종료 중 하나를 입력하세요.")
    print("=" * 60)

    model = load_model()
    print(f"모델 불러오기 완료: {MODEL_PATH}")

    while True:
        message = input("\n문자를 입력하세요: ").strip()

        if message.lower() in EXIT_WORDS:
            print("프로그램을 종료합니다.")
            break

        predict_one(model, message)


if __name__ == "__main__":
    main()
