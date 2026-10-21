"""
prepare_model.py
----------------
Reproducible script to build and save the TF-IDF vectorizer and internship
feature matrix from technical_internships_ready.csv.

ML Methodology (Faithful to the original college ML project):
- Text feature: combination of internship 'title' and 'skills_clean'.
- Feature extraction: scikit-learn TfidfVectorizer with:
    - lowercase=True
    - stop_words='english'
    - ngram_range=(1, 2) [unigrams and bigrams]
- Comparison metric: Cosine similarity in the TF-IDF vector space.
"""

import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "technical_internships_ready.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_artifacts")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
MATRIX_PATH = os.path.join(MODEL_DIR, "tfidf_matrix.joblib")
DATASET_PKL_PATH = os.path.join(MODEL_DIR, "internships_cleaned.joblib")


def load_and_preprocess_dataset(csv_path: str = DATA_PATH) -> pd.DataFrame:
    """
    Loads and preprocesses the technical internships dataset.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'. Please ensure technical_internships_ready.csv is located in the data/ directory."
        )

    df = pd.read_csv(csv_path)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()

    # Ensure required columns exist
    for col in ["title", "location", "skills", "skills_clean"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()
        else:
            df[col] = ""

    # Ensure salary columns are numeric
    for col in ["start_salary", "max_salary"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    # Combine title and cleaned skills into one text representation for vectorization
    df["internship_text"] = (
        df["title"] + " " + df["skills_clean"]
    ).str.strip()

    # Filter out empty records if any and reset index
    df = df[df["internship_text"] != ""].copy().reset_index(drop=True)

    return df


def build_and_save_model(csv_path: str = DATA_PATH, output_dir: str = MODEL_DIR) -> dict:
    """
    Fits the TF-IDF vectorizer on internship_text and saves fitted objects using joblib.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = load_and_preprocess_dataset(csv_path)

    # Scikit-learn TF-IDF Vectorizer with unigrams & bigrams and English stop-words
    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )

    tfidf_matrix = tfidf.fit_transform(df["internship_text"])

    # Persist artifacts
    joblib.dump(tfidf, os.path.join(output_dir, "tfidf_vectorizer.joblib"))
    joblib.dump(tfidf_matrix, os.path.join(output_dir, "tfidf_matrix.joblib"))
    joblib.dump(df, os.path.join(output_dir, "internships_cleaned.joblib"))

    info = {
        "record_count": len(df),
        "vocabulary_size": len(tfidf.vocabulary_),
        "matrix_shape": tfidf_matrix.shape,
        "output_dir": output_dir
    }
    return info


if __name__ == "__main__":
    print("=" * 60)
    print("Building AI Internship Recommendation System Model Artifacts...")
    print("=" * 60)
    result = build_and_save_model()
    print(f"Successfully processed {result['record_count']} internship listings.")
    print(f"TF-IDF vocabulary terms: {result['vocabulary_size']}")
    print(f"Feature matrix dimensions: {result['matrix_shape']}")
    print(f"Artifacts saved in: {result['output_dir']}")
    print("=" * 60)
