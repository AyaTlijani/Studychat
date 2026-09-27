from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "results" / "diagnostic_scores.csv"
OUTPUT_PATH = BASE_DIR / "results" / "critical_thinking_scores.csv"


def min_max(series):
    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(0.0, index=series.index)

    return (series - minimum) / (maximum - minimum)


def main():

    print("=" * 60)
    print("CRITICAL THINKING EVIDENCE")
    print("=" * 60)

    if not INPUT_PATH.exists():
        print("\nERROR: diagnostic_scores.csv was not found.")
        print(INPUT_PATH)
        return

    df = pd.read_csv(INPUT_PATH)

    print(f"\nLoaded interactions: {len(df):,}")


    # ========================================================
    # 1. QUESTIONING
    # ========================================================

    # Questions can indicate that the student is not simply
    # accepting information without examination.

    questioning = min_max(
        df["question_word_count"] + df["question_mark"]
    )


    # ========================================================
    # 2. VERIFICATION
    # ========================================================

    # Checking, testing, validating, or confirming a result.

    verification = min_max(
        df["verification_word_count"]
    )


    # ========================================================
    # 3. JUSTIFICATION / EXPLANATION
    # ========================================================

    # Requests for reasoning or explanation can provide evidence
    # that the student wants to understand why an answer is valid.

    justification = min_max(
        df["explanation_word_count"]
    )


    # ========================================================
    # 4. CONTEXT / COMPARISON
    # ========================================================

    # Some interaction types provide evidence that the student
    # is examining a result in context.

    context = df["label_group"].map({
        "verification": 1.0,
        "contextual_questions": 0.9,
        "provide_context": 0.8,
        "conceptual_questions": 0.6,
        "editing_request": 0.5,
        "writing_request": 0.1,
        "off_topic": 0.0,
        "misc": 0.0
    }).fillna(0.0)


    # ========================================================
    # 5. CRITICAL THINKING EVIDENCE
    # ========================================================

    # Critical thinking emphasizes examination of information:
    #
    # - questioning
    # - verification
    # - justification
    # - contextual examination

    critical_score = (
        0.30 * questioning +
        0.35 * verification +
        0.20 * justification +
        0.15 * context
    )

    critical_score = critical_score.clip(0, 1)


    # ========================================================
    # 6. SAVE COMPONENTS
    # ========================================================

    df["critical_questioning"] = questioning
    df["critical_verification"] = verification
    df["critical_justification"] = justification
    df["critical_context"] = context

    df["critical_score"] = critical_score


    # ========================================================
    # 7. SAVE
    # ========================================================

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nCritical-thinking evidence calculated.")

    print("\nObservable indicators:")
    print("- Questioning")
    print("- Verification")
    print("- Justification / explanation")
    print("- Contextual examination")

    print(f"\nSaved to:")
    print(OUTPUT_PATH)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

