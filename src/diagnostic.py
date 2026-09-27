
from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "results" / "autonomy_scores.csv"
OUTPUT_PATH = BASE_DIR / "results" / "diagnostic_scores.csv"


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
    print("DIAGNOSTIC / PROBLEM-SOLVING EVIDENCE")
    print("=" * 60)

    if not INPUT_PATH.exists():
        print("\nERROR: autonomy_scores.csv was not found.")
        print(INPUT_PATH)
        return

    df = pd.read_csv(INPUT_PATH)

    print(f"\nLoaded interactions: {len(df):,}")


    # ========================================================
    # 1. PROBLEM IDENTIFICATION
    # ========================================================

    # Words such as error, bug, issue, problem, failed, debug...
    # provide direct textual evidence that the student is
    # dealing with a problem.

    problem = min_max(
        df["diagnostic_word_count"]
    )


    # ========================================================
    # 2. QUESTIONING / INVESTIGATION
    # ========================================================

    # Questions can indicate that the student is investigating
    # why something happened or how to solve it.

    investigation = min_max(
        df["question_word_count"] + df["question_mark"]
    )


    # ========================================================
    # 3. EXPLANATION / CONTEXT
    # ========================================================

    # Providing context is particularly useful for diagnosing
    # a problem because it gives information about the situation.

    context = df["label_group"].map({
        "provide_context": 1.0,
        "contextual_questions": 1.0,
        "verification": 0.8,
        "editing_request": 0.7,
        "conceptual_questions": 0.5,
        "writing_request": 0.1,
        "off_topic": 0.0,
        "misc": 0.0
    }).fillna(0.0)


    # ========================================================
    # 4. MODIFICATION / ATTEMPTED SOLUTION
    # ========================================================

    # Editing or modifying something can indicate that the
    # student is actively trying to solve the problem.

    modification = min_max(
        df["editing_word_count"]
    )


    # ========================================================
    # 5. VERIFICATION
    # ========================================================

    # Checking whether a change or solution worked is useful
    # evidence of a problem-solving process.

    verification = min_max(
        df["verification_word_count"]
    )


    # ========================================================
    # 6. DIAGNOSTIC EVIDENCE
    # ========================================================

    # Unlike Autonomy, Diagnostic evidence emphasizes:
    #
    # - identifying a problem
    # - investigating it
    # - providing useful context
    # - attempting a modification
    # - checking the result

    diagnostic_score = (
        0.25 * problem +
        0.20 * investigation +
        0.20 * context +
        0.20 * modification +
        0.15 * verification
    )

    diagnostic_score = diagnostic_score.clip(0, 1)


    # ========================================================
    # 7. SAVE COMPONENTS
    # ========================================================

    df["diagnostic_problem"] = problem
    df["diagnostic_investigation"] = investigation
    df["diagnostic_context"] = context
    df["diagnostic_modification"] = modification
    df["diagnostic_verification"] = verification

    df["diagnostic_score"] = diagnostic_score


    # ========================================================
    # 8. SAVE
    # ========================================================

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nDiagnostic evidence calculated.")

    print("\nObservable indicators:")
    print("- Problem identification")
    print("- Investigation")
    print("- Providing relevant context")
    print("- Modification / attempted solution")
    print("- Verification")

    print(f"\nSaved to:")
    print(OUTPUT_PATH)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

