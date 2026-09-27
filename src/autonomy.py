
from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "results" / "nlp_features.csv"
OUTPUT_PATH = BASE_DIR / "results" / "autonomy_scores.csv"


def min_max(series):
    """
    Normalize a feature between 0 and 1.
    """
    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(0.0, index=series.index)

    return (series - minimum) / (maximum - minimum)


def main():

    print("=" * 60)
    print("AUTONOMY EVIDENCE")
    print("=" * 60)

    if not INPUT_PATH.exists():
        print("\nERROR: nlp_features.csv was not found.")
        print(INPUT_PATH)
        return

    df = pd.read_csv(INPUT_PATH)

    print(f"\nLoaded interactions: {len(df):,}")


    # ========================================================
    # 1. OBSERVABLE AUTONOMY INDICATORS
    # ========================================================

    # Asking for explanations / understanding
    explanation = min_max(
        df["explanation_word_count"]
    )

    # Asking questions
    questions = min_max(
        df["question_word_count"] + df["question_mark"]
    )

    # Modifying or refining something
    editing = min_max(
        df["editing_word_count"]
    )

    # Providing context / an existing attempt
    context = df["label_group"].map({
        "provide_context": 1.0,
        "contextual_questions": 1.0,
        "editing_request": 0.8,
        "conceptual_questions": 0.7,
        "verification": 0.6,
        "writing_request": 0.2,
        "off_topic": 0.0,
        "misc": 0.0
    }).fillna(0.0)


    # ========================================================
    # 2. DIRECT DELEGATION
    # ========================================================

    # Asking the assistant to simply perform the task
    # provides weaker evidence of autonomy.

    delegation = min_max(
        df["delegation_word_count"]
    )


    # ========================================================
    # 3. AUTONOMY EVIDENCE
    # ========================================================

    # Autonomy is primarily about active engagement:
    #
    # - trying to understand
    # - asking meaningful questions
    # - providing context
    # - modifying/refining work
    #
    # Direct delegation reduces the evidence.

    positive_evidence = (
        0.30 * explanation +
        0.25 * questions +
        0.25 * editing +
        0.20 * context
    )

    autonomy_score = (
        positive_evidence -
        0.20 * delegation
    )

    autonomy_score = autonomy_score.clip(0, 1)


    # ========================================================
    # 4. SAVE COMPONENTS
    # ========================================================

    df["autonomy_explanation"] = explanation
    df["autonomy_questions"] = questions
    df["autonomy_editing"] = editing
    df["autonomy_context"] = context
    df["autonomy_delegation"] = delegation

    df["autonomy_score"] = autonomy_score


    # ========================================================
    # 5. SAVE
    # ========================================================

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nAutonomy evidence calculated.")

    print("\nObservable indicators:")
    print("- Explanation / understanding")
    print("- Questioning")
    print("- Editing / refinement")
    print("- Providing context")
    print("- Direct delegation (negative evidence)")

    print(f"\nSaved to:")
    print(OUTPUT_PATH)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
