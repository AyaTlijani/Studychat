from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

RESULTS_DIR = Path("results")

INPUT_FILE = RESULTS_DIR / "nlp_features.csv"

STUDENT_OUTPUT = RESULTS_DIR / "student_profiles.csv"
PHASE_OUTPUT = RESULTS_DIR / "competency_by_phase.csv"


# ============================================================
# HELPERS
# ============================================================

def binary(df, column):
    """
    Convert an existing indicator into binary evidence.

    > 0 means the indicator was detected in the interaction.
    """
    if column not in df.columns:
        return pd.Series(0, index=df.index, dtype=int)

    values = pd.to_numeric(df[column], errors="coerce").fillna(0)

    return (values > 0).astype(int)


def label_is(df, value):
    """
    Check whether label_group matches a specific behavioral label.
    """
    if "label_group" not in df.columns:
        return pd.Series(0, index=df.index, dtype=int)

    return (df["label_group"] == value).astype(int)


def combine(*signals):
    """
    An interaction contains evidence if at least one of the
    competency-specific signals is present.
    """
    result = signals[0].copy()

    for signal in signals[1:]:
        result = ((result == 1) | (signal == 1)).astype(int)

    return result.astype(int)


def percentile(series):
    """
    Relative position inside the StudyChat cohort.

    IMPORTANT:
    This is NOT a competency percentage.
    """
    return series.rank(method="average", pct=True) * 100


def level_from_percentile(value):
    if value >= 75:
        return "Strong"
    elif value >= 25:
        return "Moderate"
    else:
        return "Developing"


# ============================================================
# AUTONOMY
# ============================================================

def autonomy_evidence(df):
    """
    Observable autonomy evidence.

    Positive evidence:
        - conceptual / explanation request
        - providing context
        - editing/refining something

    Negative evidence:
        - direct delegation to the LLM

    A question alone is NOT considered autonomy evidence.
    """

    conceptual = label_is(df, "conceptual_questions")

    explanation = binary(
        df,
        "explanation_word_count"
    )

    context = label_is(
        df,
        "provide_context"
    )

    editing = label_is(
        df,
        "editing_request"
    )

    evidence = combine(
        conceptual,
        explanation,
        context,
        editing
    )

    delegation = binary(
        df,
        "delegation_word_count"
    )

    return evidence, delegation


# ============================================================
# DIAGNOSTIC / PROBLEM-SOLVING
# ============================================================

def diagnostic_evidence(df):
    """
    Observable diagnostic / problem-solving evidence.

    Evidence is based on behaviors directly related to diagnosing
    or resolving a problem:

        - diagnostic language
        - providing contextual information
        - editing/modifying something
        - verification

    Generic questions are NOT used.
    """

    diagnostic = binary(
        df,
        "diagnostic_word_count"
    )

    context = label_is(
        df,
        "provide_context"
    )

    editing = label_is(
        df,
        "editing_request"
    )

    verification = label_is(
        df,
        "verification"
    )

    evidence = combine(
        diagnostic,
        context,
        editing,
        verification
    )

    return evidence


# ============================================================
# CRITICAL THINKING
# ============================================================

def critical_evidence(df):
    """
    Observable critical-thinking / evaluation evidence.

    Evidence focuses on checking or challenging information:

        - explicit verification interactions
        - verification language
        - requests for explanation/justification

    A generic question is NOT enough.
    """

    verification_label = label_is(
        df,
        "verification"
    )

    verification_language = binary(
        df,
        "verification_word_count"
    )

    explanation = binary(
        df,
        "explanation_word_count"
    )

    evidence = combine(
        verification_label,
        verification_language,
        explanation
    )

    return evidence


# ============================================================
# STUDENT AGGREGATION
# ============================================================

def aggregate(df, evidence_column, negative_column=None):
    """
    Calculate transparent student-level evidence measures.

    evidence_rate:
        evidence-bearing interactions / analyzed interactions

    This is an observable behavior rate, NOT a competency percentage.
    """

    grouped = df.groupby("userId")

    result = grouped.size().rename("interaction_count").to_frame()

    result["evidence_count"] = grouped[evidence_column].sum()

    result["evidence_rate"] = (
        result["evidence_count"] /
        result["interaction_count"]
    )

    if negative_column:
        result["negative_count"] = grouped[negative_column].sum()

        result["negative_rate"] = (
            result["negative_count"] /
            result["interaction_count"]
        )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    if not INPUT_FILE.exists():
        print(f"ERROR: {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE)

    print("=" * 70)
    print("STUDYCHAT — COMPETENCY EVIDENCE ANALYSIS")
    print("=" * 70)

    print(f"\nInteractions: {len(df):,}")
    print(f"Students: {df['userId'].nunique():,}")

    # --------------------------------------------------------
    # BUILD INTERACTION-LEVEL EVIDENCE
    # --------------------------------------------------------

    autonomy, delegation = autonomy_evidence(df)

    diagnostic = diagnostic_evidence(df)

    critical = critical_evidence(df)

    df["autonomy_evidence"] = autonomy
    df["autonomy_delegation"] = delegation

    df["diagnostic_evidence"] = diagnostic

    df["critical_evidence"] = critical

    # --------------------------------------------------------
    # STUDENT-LEVEL AGGREGATION
    # --------------------------------------------------------

    auto = aggregate(
        df,
        "autonomy_evidence",
        "autonomy_delegation"
    )

    diag = aggregate(
        df,
        "diagnostic_evidence"
    )

    crit = aggregate(
        df,
        "critical_evidence"
    )

    # Rename columns

    auto = auto.rename(columns={
        "interaction_count": "autonomy_interactions",
        "evidence_count": "autonomy_evidence_count",
        "evidence_rate": "autonomy_evidence_rate",
        "negative_count": "autonomy_delegation_count",
        "negative_rate": "autonomy_delegation_rate"
    })

    diag = diag.rename(columns={
        "interaction_count": "diagnostic_interactions",
        "evidence_count": "diagnostic_evidence_count",
        "evidence_rate": "diagnostic_evidence_rate"
    })

    crit = crit.rename(columns={
        "interaction_count": "critical_interactions",
        "evidence_count": "critical_evidence_count",
        "evidence_rate": "critical_evidence_rate"
    })

    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------

    profiles = (
        auto
        .join(diag, how="outer")
        .join(crit, how="outer")
        .reset_index()
    )

    # --------------------------------------------------------
    # RELATIVE POSITION
    # --------------------------------------------------------

    profiles["autonomy_percentile"] = percentile(
        profiles["autonomy_evidence_rate"]
    )

    profiles["diagnostic_percentile"] = percentile(
        profiles["diagnostic_evidence_rate"]
    )

    profiles["critical_percentile"] = percentile(
        profiles["critical_evidence_rate"]
    )

    # --------------------------------------------------------
    # LEVELS
    # --------------------------------------------------------

    profiles["autonomy_level"] = (
        profiles["autonomy_percentile"]
        .apply(level_from_percentile)
    )

    profiles["diagnostic_level"] = (
        profiles["diagnostic_percentile"]
        .apply(level_from_percentile)
    )

    profiles["critical_level"] = (
        profiles["critical_percentile"]
        .apply(level_from_percentile)
    )

    # --------------------------------------------------------
    # STRONGEST / DEVELOPMENT
    # --------------------------------------------------------

    percentile_columns = [
        "autonomy_percentile",
        "diagnostic_percentile",
        "critical_percentile"
    ]

    profiles["strongest_competency"] = (
        profiles[percentile_columns]
        .idxmax(axis=1)
        .map({
            "autonomy_percentile": "Autonomy",
            "diagnostic_percentile": "Diagnostic",
            "critical_percentile": "Critical Thinking"
        })
    )

    profiles["development_area"] = (
        profiles[percentile_columns]
        .idxmin(axis=1)
        .map({
            "autonomy_percentile": "Autonomy",
            "diagnostic_percentile": "Diagnostic",
            "critical_percentile": "Critical Thinking"
        })
    )

    # --------------------------------------------------------
    # COHORT INDEX + RANK
    # --------------------------------------------------------
    #
    # This is only a relative comparison index.
    # It is NOT a competency score.
    #
    # It is useful for sorting candidates in the dashboard.

    profiles["cohort_index"] = (
        profiles[percentile_columns]
        .mean(axis=1)
    )

    profiles["cohort_rank"] = (
        profiles["cohort_index"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    # --------------------------------------------------------
    # TOTAL ACTIVITY
    # --------------------------------------------------------

    profiles["total_interactions"] = (
        profiles[
            [
                "autonomy_interactions",
                "diagnostic_interactions",
                "critical_interactions"
            ]
        ]
        .max(axis=1)
    )

    profiles["total_evidence"] = (
        profiles[
            [
                "autonomy_evidence_count",
                "diagnostic_evidence_count",
                "critical_evidence_count"
            ]
        ]
        .sum(axis=1)
    )

    # --------------------------------------------------------
    # SAVE STUDENT PROFILES
    # --------------------------------------------------------

    profiles.to_csv(
        STUDENT_OUTPUT,
        index=False
    )

    # ========================================================
    # PHASE ANALYSIS
    # ========================================================

    if "phase" in df.columns:

        phase = (
            df.groupby(["userId", "phase"])
            .agg(
                autonomy_evidence=(
                    "autonomy_evidence",
                    "sum"
                ),
                autonomy_interactions=(
                    "autonomy_evidence",
                    "count"
                ),
                diagnostic_evidence=(
                    "diagnostic_evidence",
                    "sum"
                ),
                diagnostic_interactions=(
                    "diagnostic_evidence",
                    "count"
                ),
                critical_evidence=(
                    "critical_evidence",
                    "sum"
                ),
                critical_interactions=(
                    "critical_evidence",
                    "count"
                )
            )
            .reset_index()
        )

        phase["autonomy_rate"] = (
            phase["autonomy_evidence"] /
            phase["autonomy_interactions"]
        )

        phase["diagnostic_rate"] = (
            phase["diagnostic_evidence"] /
            phase["diagnostic_interactions"]
        )

        phase["critical_rate"] = (
            phase["critical_evidence"] /
            phase["critical_interactions"]
        )

        phase.to_csv(
            PHASE_OUTPUT,
            index=False
        )

    # ========================================================
    # REPORT
    # ========================================================

    print("\n" + "=" * 70)
    print("STUDENT-LEVEL EVIDENCE")
    print("=" * 70)

    for name, column in [
        ("Autonomy", "autonomy_evidence_rate"),
        ("Diagnostic", "diagnostic_evidence_rate"),
        ("Critical Thinking", "critical_evidence_rate")
    ]:

        print(f"\n{name}")

        print(
            profiles[column]
            .describe(
                percentiles=[
                    0.10,
                    0.25,
                    0.50,
                    0.75,
                    0.90
                ]
            )
            .round(4)
        )

    print("\n" + "=" * 70)
    print("LEVEL COUNTS")
    print("=" * 70)

    for name, column in [
        ("Autonomy", "autonomy_level"),
        ("Diagnostic", "diagnostic_level"),
        ("Critical Thinking", "critical_level")
    ]:

        print(f"\n{name}")
        print(
            profiles[column]
            .value_counts()
            .reindex(
                ["Strong", "Moderate", "Developing"],
                fill_value=0
            )
        )

    print("\n" + "=" * 70)
    print("FILES CREATED")
    print("=" * 70)

    print(f"\n{STUDENT_OUTPUT}")
    print(PHASE_OUTPUT)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()