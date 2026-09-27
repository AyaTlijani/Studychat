import json
from pathlib import Path

import pandas as pd


# =========================
# Configuration
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "StudyChat.jsonl"
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_PATH = RESULTS_DIR / "preprocessed.csv"


# =========================
# Load JSONL
# =========================

def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: could not read line {line_number}")

    return pd.DataFrame(rows)


# =========================
# Extract LLM label
# =========================

def extract_label(value):
    """
    Extract the label and split it into:
    - label_group
    - label_detail

    Example:
    contextual_questions>Code Explanation

    becomes:
    label_group   = contextual_questions
    label_detail  = Code Explanation
    """

    if isinstance(value, dict):
        value = value.get("label", "")

    if not isinstance(value, str):
        return "", ""

    parts = value.split(">", 1)

    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()

    return value.strip(), ""


# =========================
# Preprocess
# =========================

def preprocess(df):

    # Keep only rows with a usable student prompt
    df = df.copy()

    df["prompt"] = df["prompt"].fillna("").astype(str).str.strip()

    df = df[df["prompt"] != ""].copy()

    # Parse timestamp
    df["timestamp"] = pd.to_numeric(
        df["timestamp"],
        errors="coerce"
    )

    df["datetime"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        errors="coerce"
    )

    # Extract label information
    labels = df["llm_label"].apply(extract_label)

    df["label_group"] = labels.apply(lambda x: x[0])
    df["label_detail"] = labels.apply(lambda x: x[1])

    # Convert interaction counts to numeric
    df["interactionCount"] = pd.to_numeric(
        df["interactionCount"],
        errors="coerce"
    )

    df["chatTotalInteractionCount"] = pd.to_numeric(
        df["chatTotalInteractionCount"],
        errors="coerce"
    )

    # Sort chronologically
    df = df.sort_values(
        ["userId", "datetime"]
    ).reset_index(drop=True)

    # Create temporal phase within each student's activity
    df["student_interaction_index"] = (
        df.groupby("userId").cumcount() + 1
    )

    df["student_total_interactions"] = (
        df.groupby("userId")["prompt"].transform("count")
    )

    df["progress_ratio"] = (
        df["student_interaction_index"]
        / df["student_total_interactions"]
    )

    df["phase"] = pd.cut(
        df["progress_ratio"],
        bins=[0, 0.33, 0.66, 1.0],
        labels=["early", "middle", "late"],
        include_lowest=True
    )

    return df


# =========================
# Main
# =========================

def main():

    print("=" * 60)
    print("STUDYCHAT PREPROCESSING")
    print("=" * 60)

    print(f"\nInput: {DATA_PATH}")

    if not DATA_PATH.exists():
        print("\nERROR: StudyChat.jsonl was not found.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_jsonl(DATA_PATH)

    print(f"Loaded interactions: {len(df):,}")

    df = preprocess(df)

    # Columns needed by the rest of the project
    columns = [
        "prompt",
        "response",
        "topic",
        "timestamp",
        "datetime",
        "chatId",
        "userId",
        "interactionCount",
        "chatTotalInteractionCount",
        "label_group",
        "label_detail",
        "semester",
        "student_interaction_index",
        "student_total_interactions",
        "progress_ratio",
        "phase"
    ]

    # Keep only columns that actually exist
    columns = [c for c in columns if c in df.columns]

    df = df[columns]

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nProcessed interactions: {len(df):,}")
    print(f"Students: {df['userId'].nunique():,}")
    print(f"Conversations: {df['chatId'].nunique():,}")

    print("\n--- Label groups ---")
    print(
        df["label_group"]
        .value_counts()
        .to_string()
    )

    print("\n--- Phases ---")
    print(
        df["phase"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(f"\nSaved to:")
    print(OUTPUT_PATH)

    print("\n" + "=" * 60)
    print("Preprocessing complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()