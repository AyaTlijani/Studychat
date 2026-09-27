import json
from pathlib import Path
from collections import Counter

import pandas as pd


# =========================
# Configuration
# =========================

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "StudyChat.jsonl"


# =========================
# Load dataset
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
# Inspection
# =========================

def main():
    print("=" * 60)
    print("STUDYCHAT DATASET INSPECTION")
    print("=" * 60)

    print(f"\nFile: {DATA_PATH}")

    if not DATA_PATH.exists():
        print("\nERROR: StudyChat.jsonl was not found.")
        print("Expected location:")
        print(DATA_PATH)
        return

    df = load_jsonl(DATA_PATH)

    print(f"\nNumber of interactions: {len(df):,}")
    print(f"Number of columns: {len(df.columns)}")

    print("\n--- Columns ---")
    for column in df.columns:
        print(f"- {column}")

    print("\n--- Missing values ---")
    missing = df.isna().sum()
    for column, count in missing.items():
        if count > 0:
            print(f"- {column}: {count:,}")

    if "userId" in df.columns:
        print(f"\nNumber of students: {df['userId'].nunique():,}")

    if "chatId" in df.columns:
        print(f"Number of conversations: {df['chatId'].nunique():,}")

    if "semester" in df.columns:
        print("\n--- Semesters ---")
        print(df["semester"].value_counts(dropna=False).to_string())

    if "topic" in df.columns:
        print("\n--- Topics ---")
        print(df["topic"].value_counts(dropna=False).sort_index().to_string())

    if "llm_label" in df.columns:
        print("\n--- LLM labels ---")

        labels = []

        for value in df["llm_label"].dropna():
            if isinstance(value, dict):
                label = value.get("label")
                if label:
                    labels.append(label)
            else:
                labels.append(str(value))

        for label, count in Counter(labels).most_common():
            print(f"- {label}: {count:,}")

    if "timestamp" in df.columns:
        print("\n--- Timestamp ---")

        dates = pd.to_numeric(df["timestamp"], errors="coerce")

        if dates.notna().any():
            converted = pd.to_datetime(
                dates,
                unit="ms",
                errors="coerce"
            )

            print(f"Earliest: {converted.min()}")
            print(f"Latest:   {converted.max()}")

    print("\n--- Example row ---")
    print(df.iloc[0].to_dict())

    print("\n" + "=" * 60)
    print("Inspection complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()