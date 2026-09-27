from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

RESULTS_DIR = Path("results")

INPUT_FILE = RESULTS_DIR / "nlp_features.csv"


# ============================================================
# HELPERS
# ============================================================

def describe_feature(df, feature):
    """Print distribution and student-level coverage for one feature."""

    print("\n" + "=" * 70)
    print(f"FEATURE: {feature}")
    print("=" * 70)

    if feature not in df.columns:
        print("NOT FOUND")
        return

    values = pd.to_numeric(df[feature], errors="coerce").fillna(0)

    print("\nInteraction-level distribution:")
    print(values.describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95]).round(4))

    print(f"\nInteractions > 0: {(values > 0).sum()} / {len(values)}")
    print(f"Interactions > 0 rate: {(values > 0).mean():.2%}")

    if "userId" in df.columns:
        student_values = (
            pd.DataFrame({
                "userId": df["userId"],
                "value": values
            })
            .groupby("userId")["value"]
            .agg(
                interactions="count",
                total="sum",
                mean="mean",
                evidence_count=lambda x: (x > 0).sum(),
                evidence_rate=lambda x: (x > 0).mean()
            )
        )

        print("\nStudent-level distribution of evidence rate:")
        print(
            student_values["evidence_rate"]
            .describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95])
            .round(4)
        )

        print(
            f"\nStudents with any evidence: "
            f"{(student_values['evidence_count'] > 0).sum()} / "
            f"{len(student_values)}"
        )

        print(
            f"Students with NO evidence: "
            f"{(student_values['evidence_count'] == 0).sum()}"
        )


def print_label_distribution(df):
    if "label_group" not in df.columns:
        return

    print("\n" + "=" * 70)
    print("LABEL GROUP DISTRIBUTION")
    print("=" * 70)

    counts = df["label_group"].value_counts()
    percentages = df["label_group"].value_counts(normalize=True) * 100

    result = pd.DataFrame({
        "count": counts,
        "percentage": percentages.round(2)
    })

    print(result)


def print_similarity_distribution(df):
    similarity_features = [
        "autonomy_similarity",
        "diagnostic_similarity",
        "critical_similarity"
    ]

    print("\n" + "=" * 70)
    print("SEMANTIC SIMILARITY DISTRIBUTIONS")
    print("=" * 70)

    for feature in similarity_features:
        if feature not in df.columns:
            continue

        values = pd.to_numeric(df[feature], errors="coerce")

        print(f"\n{feature}")
        print(
            values.describe(
                percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
            ).round(4)
        )


def print_student_activity(df):
    if "userId" not in df.columns:
        return

    print("\n" + "=" * 70)
    print("STUDENT ACTIVITY")
    print("=" * 70)

    activity = df.groupby("userId").size()

    print(
        activity.describe(
            percentiles=[0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        )
    )

    print(f"\nStudents: {activity.shape[0]}")
    print(f"Interactions: {len(df)}")


def print_correlations(df, features):
    existing = [f for f in features if f in df.columns]

    if len(existing) < 2:
        return

    print("\n" + "=" * 70)
    print("FEATURE CORRELATIONS")
    print("=" * 70)

    numeric = df[existing].apply(pd.to_numeric, errors="coerce")

    print(numeric.corr().round(2))


# ============================================================
# MAIN
# ============================================================

def main():

    if not INPUT_FILE.exists():
        print(f"ERROR: File not found: {INPUT_FILE}")
        print("Run the NLP step first.")
        return

    df = pd.read_csv(INPUT_FILE)

    print("=" * 70)
    print("STUDYCHAT — EVIDENCE INSPECTION")
    print("=" * 70)

    print(f"\nFile: {INPUT_FILE}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    if "userId" in df.columns:
        print(f"Students: {df['userId'].nunique():,}")

    print_label_distribution(df)
    print_student_activity(df)
    print_similarity_distribution(df)

    features = [
        # Autonomy
        "explanation_word_count",
        "question_word_count",
        "editing_word_count",
        "delegation_word_count",

        # Diagnostic / problem solving
        "diagnostic_word_count",
        "verification_word_count",

        # General linguistic evidence
        "word_count",
        "question_mark",
    ]

    for feature in features:
        describe_feature(df, feature)

    print_correlations(
        df,
        features + [
            "autonomy_similarity",
            "diagnostic_similarity",
            "critical_similarity"
        ]
    )

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)
    print("\nDo NOT calculate competency scores yet.")
    print("We will build them from these distributions.")


if __name__ == "__main__":
    main()