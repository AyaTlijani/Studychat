from pathlib import Path
import re

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# Configuration
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "results" / "preprocessed.csv"
OUTPUT_PATH = BASE_DIR / "results" / "nlp_features.csv"


# =========================
# Competency descriptions
# =========================

AUTONOMY_TEXTS = [
    "The student tries to understand a concept rather than simply asking for the answer.",
    "The student asks for explanations and reasoning to develop their own understanding.",
    "The student independently explores how or why something works.",
]

DIAGNOSTIC_TEXTS = [
    "The student identifies errors, problems, or unexpected results.",
    "The student investigates the cause of a problem and asks how to fix it.",
    "The student uses explanations and evidence to solve a problem.",
]

CRITICAL_TEXTS = [
    "The student questions whether an answer or result is correct.",
    "The student verifies, tests, or checks an answer or result.",
    "The student challenges an explanation or considers whether another solution is possible.",
]


# =========================
# Text cleaning
# =========================

def clean_text(text):
    text = str(text).lower()

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================
# Linguistic features
# =========================

def count_words(text, words):
    """
    Count how many target words appear as complete words.
    """
    count = 0

    for word in words:
        pattern = rf"\b{re.escape(word)}\b"
        count += len(re.findall(pattern, text))

    return count


def count_phrases(text, phrases):
    """
    Count how many target phrases appear in the text.
    """
    count = 0

    for phrase in phrases:
        if phrase in text:
            count += 1

    return count


def extract_linguistic_features(text):

    question_words = [
        "why",
        "how",
        "what",
        "which",
        "can",
        "could",
        "should",
        "is",
        "are",
        "does",
        "do"
    ]

    explanation_words = [
        "explain",
        "understand",
        "meaning",
        "reason",
        "why"
    ]

    verification_words = [
        "check",
        "verify",
        "correct",
        "validate",
        "test",
        "confirm",
        "sure",
        "accurate"
    ]

    diagnostic_words = [
        "error",
        "bug",
        "issue",
        "problem",
        "wrong",
        "fail",
        "failed",
        "debug",
        "fix",
        "incorrect"
    ]

    delegation_phrases = [
        "write",
        "give me",
        "generate",
        "provide",
        "make",
        "create",
        "code for me"
    ]

    editing_words = [
        "edit",
        "modify",
        "change",
        "improve",
        "rewrite",
        "correct"
    ]

    return {
        "word_count": len(text.split()),

        "question_mark": int("?" in text),

        "question_word_count": count_words(
            text,
            question_words
        ),

        "explanation_word_count": count_words(
            text,
            explanation_words
        ),

        "verification_word_count": count_words(
            text,
            verification_words
        ),

        "diagnostic_word_count": count_words(
            text,
            diagnostic_words
        ),

        "delegation_word_count": count_phrases(
            text,
            delegation_phrases
        ),

        "editing_word_count": count_words(
            text,
            editing_words
        )
    }


# =========================
# Semantic similarity
# =========================

def calculate_similarity(
    prompt_embeddings,
    reference_embeddings
):
    """
    Calculate the average cosine similarity
    between a prompt and the competency descriptions.

    We use the average rather than only the maximum
    so that one unusually high similarity does not
    completely determine the evidence.
    """

    similarities = cosine_similarity(
        prompt_embeddings,
        reference_embeddings
    )

    return similarities.mean(axis=1)


# =========================
# Main NLP pipeline
# =========================

def main():

    print("=" * 60)
    print("STUDYCHAT NLP FEATURE EXTRACTION")
    print("=" * 60)

    if not INPUT_PATH.exists():
        print("\nERROR: preprocessed.csv was not found.")
        print(INPUT_PATH)
        return

    # -------------------------
    # Load data
    # -------------------------

    df = pd.read_csv(INPUT_PATH)

    print(f"\nLoaded interactions: {len(df):,}")

    # -------------------------
    # Clean prompts
    # -------------------------

    prompts = (
        df["prompt"]
        .fillna("")
        .apply(clean_text)
        .tolist()
    )

    # -------------------------
    # Load Sentence Transformer
    # -------------------------

    print("\nLoading sentence embedding model...")

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    print("Model loaded.")

    # -------------------------
    # Create embeddings
    # -------------------------

    print("\nCreating prompt embeddings...")

    prompt_embeddings = model.encode(
        prompts,
        batch_size=64,
        show_progress_bar=True
    )

    # -------------------------
    # Create competency embeddings
    # -------------------------

    autonomy_embeddings = model.encode(
        AUTONOMY_TEXTS
    )

    diagnostic_embeddings = model.encode(
        DIAGNOSTIC_TEXTS
    )

    critical_embeddings = model.encode(
        CRITICAL_TEXTS
    )

    # -------------------------
    # Semantic similarity
    # -------------------------

    df["autonomy_similarity"] = calculate_similarity(
        prompt_embeddings,
        autonomy_embeddings
    )

    df["diagnostic_similarity"] = calculate_similarity(
        prompt_embeddings,
        diagnostic_embeddings
    )

    df["critical_similarity"] = calculate_similarity(
        prompt_embeddings,
        critical_embeddings
    )

    # -------------------------
    # Linguistic features
    # -------------------------

    linguistic_features = pd.DataFrame(
        [
            extract_linguistic_features(prompt)
            for prompt in prompts
        ]
    )

    df = pd.concat(
        [df, linguistic_features],
        axis=1
    )

    # -------------------------
    # Save
    # -------------------------

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n--- NLP features created ---")

    feature_columns = [
        "autonomy_similarity",
        "diagnostic_similarity",
        "critical_similarity",
        "word_count",
        "question_mark",
        "question_word_count",
        "explanation_word_count",
        "verification_word_count",
        "diagnostic_word_count",
        "delegation_word_count",
        "editing_word_count"
    ]

    for column in feature_columns:
        print(f"- {column}")

    print(f"\nSaved to:")
    print(OUTPUT_PATH)

    print("\n" + "=" * 60)
    print("NLP feature extraction complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()