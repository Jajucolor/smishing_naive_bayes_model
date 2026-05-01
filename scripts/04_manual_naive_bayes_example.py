"""
4단계: 보고서용 나이브 베이즈 직접 계산 예시

실행:
    python scripts/04_manual_naive_bayes_example.py
"""

from collections import Counter, defaultdict
import math


DATA = [
    ("오늘 회의 시간 확인", "일반"),
    ("내일 점심 약속 확인", "일반"),
    ("숙제 제출 시간 확인", "일반"),
    ("계좌 정지 링크 확인", "스미싱"),
    ("본인 인증 링크 접속", "스미싱"),
    ("배송 오류 링크 확인", "스미싱"),
]

TEST_MESSAGE = "계좌 링크 확인"


def tokenize(text: str) -> list[str]:
    return text.split()


def train_naive_bayes(data):
    class_doc_counts = Counter()
    class_word_counts = defaultdict(Counter)
    class_total_words = Counter()
    vocabulary = set()

    for text, label in data:
        words = tokenize(text)
        class_doc_counts[label] += 1

        for word in words:
            class_word_counts[label][word] += 1
            class_total_words[label] += 1
            vocabulary.add(word)

    total_docs = len(data)

    return {
        "class_doc_counts": class_doc_counts,
        "class_word_counts": class_word_counts,
        "class_total_words": class_total_words,
        "vocabulary": vocabulary,
        "total_docs": total_docs,
    }


def prior_probability(label: str, model_info) -> float:
    return model_info["class_doc_counts"][label] / model_info["total_docs"]


def likelihood_probability(word: str, label: str, model_info, alpha: int = 1) -> float:
    count = model_info["class_word_counts"][label][word]
    total_words = model_info["class_total_words"][label]
    vocab_size = len(model_info["vocabulary"])

    return (count + alpha) / (total_words + alpha * vocab_size)


def calculate_score(words: list[str], label: str, model_info) -> float:
    log_score = math.log(prior_probability(label, model_info))

    for word in words:
        log_score += math.log(likelihood_probability(word, label, model_info))

    return log_score


def softmax_from_log_scores(log_scores: dict[str, float]) -> dict[str, float]:
    max_log = max(log_scores.values())
    exp_scores = {
        label: math.exp(score - max_log)
        for label, score in log_scores.items()
    }
    total = sum(exp_scores.values())

    return {
        label: score / total
        for label, score in exp_scores.items()
    }


def main() -> None:
    model_info = train_naive_bayes(DATA)

    print("나이브 베이즈 직접 계산 예시")
    print("=" * 60)

    print("\n[1] 예시 학습 데이터")
    for text, label in DATA:
        print(f"- {text} -> {label}")

    print("\n[2] 분류별 문장 수")
    for label, count in model_info["class_doc_counts"].items():
        print(f"{label}: {count}개")

    print("\n[3] 전체 단어 목록")
    print(sorted(model_info["vocabulary"]))
    print(f"전체 단어 종류 수: {len(model_info['vocabulary'])}")

    print("\n[4] 사전확률 P(분류)")
    for label in model_info["class_doc_counts"]:
        print(f"P({label}) = {prior_probability(label, model_info):.4f}")

    test_words = tokenize(TEST_MESSAGE)
    print("\n[5] 예측할 문자")
    print(TEST_MESSAGE)
    print(f"단어: {test_words}")

    print("\n[6] 조건부확률 P(단어 | 분류), 라플라스 스무딩 적용")
    for label in model_info["class_doc_counts"]:
        print(f"\n분류 = {label}")
        for word in test_words:
            prob = likelihood_probability(word, label, model_info)
            count = model_info["class_word_counts"][label][word]
            print(f"P({word} | {label}) = {prob:.4f}  (등장 횟수: {count})")

    print("\n[7] 로그 점수 계산")
    log_scores = {}
    for label in model_info["class_doc_counts"]:
        score = calculate_score(test_words, label, model_info)
        log_scores[label] = score
        print(f"{label}: {score:.4f}")

    print("\n[8] 점수를 확률처럼 정규화한 결과")
    probs = softmax_from_log_scores(log_scores)
    for label, prob in probs.items():
        print(f"P({label} | {TEST_MESSAGE}) ≈ {prob:.4f}")

    pred = max(probs, key=probs.get)
    print("\n[9] 최종 예측")
    print(f"'{TEST_MESSAGE}' 문자는 '{pred}'으로 분류된다.")


if __name__ == "__main__":
    main()
