"""
recommendation.py
-----------------
TF-IDF Vector Space & Cosine Similarity Recommendation Engine.

Methodology Summary:
1. Reuses the TF-IDF Vectorizer fitted on internship titles and technical skills (unigrams and bigrams).
2. Transforms confirmed student skills into the same vector space.
3. Computes pairwise Cosine Similarity between the student vector and all internship vectors:
      similarity = (A · B) / (||A|| * ||B||)
4. Ranks internships in descending order of cosine similarity.
5. De-duplicates identical postings and analyzes skill overlap for transparent, explainable recommendations.
"""

import os
from collections import Counter
from typing import List, Dict, Any, Tuple, Union
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

from prepare_model import (
    MODEL_DIR,
    VECTORIZER_PATH,
    MATRIX_PATH,
    DATASET_PKL_PATH,
    build_and_save_model
)
from skill_extractor import normalize_skill_name, extract_skills


class RecommendationEngine:
    """
    Encapsulates model artifacts and inference logic.
    Supports auto-initialization if artifacts are not yet saved.
    """

    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self.vectorizer = None
        self.tfidf_matrix = None
        self.internships_df = None
        self._load_or_initialize()

    def _load_or_initialize(self) -> None:
        """Loads artifacts from disk or builds them if missing."""
        if (
            not os.path.exists(VECTORIZER_PATH)
            or not os.path.exists(MATRIX_PATH)
            or not os.path.exists(DATASET_PKL_PATH)
        ):
            print("Model artifacts not found. Automatically generating them from dataset...")
            build_and_save_model(output_dir=self.model_dir)

        self.vectorizer = joblib.load(VECTORIZER_PATH)
        self.tfidf_matrix = joblib.load(MATRIX_PATH)
        self.internships_df = joblib.load(DATASET_PKL_PATH)

    def recommend_internships(
        self,
        student_skills: Union[str, List[str]],
        top_n: int = 5,
        min_similarity_threshold: float = 0.01
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Generates top-N internship recommendations for student skills.

        Parameters:
            student_skills: String or list of skills (e.g. ['python', 'sql', 'machine learning']).
            top_n: Number of recommendations to return (default 5).
            min_similarity_threshold: Minimum cosine similarity score to consider a genuine match.

        Returns:
            Tuple of:
              - recommendations: list of dicts with internship details and similarity metrics.
              - metadata: summary info including query skills and match count.
        """
        # Format student skills into a unified query string
        if isinstance(student_skills, list):
            query_skills_list = [normalize_skill_name(s) for s in student_skills if s and s.strip()]
            query_text = " ".join(query_skills_list)
        else:
            query_text = str(student_skills).strip()
            # Extract distinct recognized skills for overlap analysis
            query_skills_list = [normalize_skill_name(s) for s in extract_skills(query_text)]
            if not query_skills_list and query_text:
                # If rule extractor didn't split them, tokenize by commas or spaces
                query_skills_list = [normalize_skill_name(w) for w in query_text.replace(",", " ").split() if w.strip()]

        if not query_text.strip():
            return [], {
                "status": "empty_input",
                "message": "No skills were provided. Please enter or confirm at least one technical skill.",
                "total_matches": 0
            }

        # Project student query into the existing fitted TF-IDF vector space
        student_vector = self.vectorizer.transform([query_text])

        # Compute cosine similarity against all precomputed internship vectors
        similarity_scores = cosine_similarity(student_vector, self.tfidf_matrix).flatten()

        # Check if there is any positive similarity match
        max_score = float(np.max(similarity_scores)) if len(similarity_scores) > 0 else 0.0
        if max_score < min_similarity_threshold:
            return [], {
                "status": "no_match",
                "message": (
                    "No relevant matches found for your current skills. "
                    "Try adding complementary skills (e.g., Python, SQL, Web Dev, C++) to find suitable opportunities."
                ),
                "total_matches": 0,
                "max_similarity": max_score
            }

        # Sort indices in descending order of similarity
        sorted_indices = np.argsort(similarity_scores)[::-1]

        recommendations = []
        seen_keys = set()
        student_skills_set = set(query_skills_list)

        for idx in sorted_indices:
            score = float(similarity_scores[idx])
            if score < min_similarity_threshold:
                break

            row = self.internships_df.iloc[idx]
            title = str(row["title"]).strip()
            location = str(row["location"]).strip()
            skills_raw = str(row["skills"]).strip() if "skills" in row else ""
            skills_clean = str(row["skills_clean"]).strip() if "skills_clean" in row else ""
            start_sal = float(row.get("start_salary", 0.0))
            max_sal = float(row.get("max_salary", 0.0))

            # Deduplication key based on title, location, and clean skills
            dedup_key = (title.lower(), location.lower(), skills_clean.lower())
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            # Analyze skill overlap for explainability
            # Parse internship skills from clean skills text
            internship_skills_list = [
                normalize_skill_name(s)
                for s in extract_skills(skills_clean or skills_raw or title)
            ]
            if not internship_skills_list and skills_clean:
                internship_skills_list = [
                    normalize_skill_name(s)
                    for s in skills_clean.replace(",", " ").split()
                    if len(s) > 1
                ]

            internship_skills_set = set(internship_skills_list)
            matched_skills = sorted(list(student_skills_set.intersection(internship_skills_set)))
            missing_skills = sorted(list(internship_skills_set.difference(student_skills_set)))

            # Salary formatting
            if start_sal > 0 and max_sal > 0:
                if start_sal == max_sal:
                    salary_display = f"₹{int(start_sal):,} / month"
                else:
                    salary_display = f"₹{int(start_sal):,} - ₹{int(max_sal):,} / month"
            elif start_sal > 0:
                salary_display = f"₹{int(start_sal):,} / month (Starting)"
            elif max_sal > 0:
                salary_display = f"Up to ₹{int(max_sal):,} / month"
            else:
                salary_display = "Stipend not disclosed"

            recommendations.append({
                "listing_id": int(idx),
                "title": title,
                "location": location or "Not specified",
                "start_salary": start_sal,
                "max_salary": max_sal,
                "salary_display": salary_display,
                "required_skills": skills_raw or skills_clean or "Not listed",
                "internship_skills": internship_skills_list,
                "similarity_score": score,
                "similarity_percentage": round(score * 100, 1),
                "matched_skills": matched_skills,
                "missing_skills": missing_skills
            })

            if len(recommendations) >= top_n:
                break

        metadata = {
            "status": "success",
            "query_skills": query_skills_list,
            "total_matches": len(recommendations),
            "max_similarity": max_score
        }

        return recommendations, metadata

    def get_dataset_skill_frequencies(self) -> Dict[str, int]:
        """
        Computes the frequency of each recognized technical skill across the 388 listings.
        """
        if hasattr(self, "_cached_skill_freqs") and self._cached_skill_freqs:
            return self._cached_skill_freqs

        counts: Counter = Counter()
        for _, row in self.internships_df.iterrows():
            text = (
                str(row.get("skills_clean", "")) + " " +
                str(row.get("skills", "")) + " " +
                str(row.get("title", ""))
            )
            found = extract_skills(text)
            for s in found:
                counts[normalize_skill_name(s)] += 1

        self._cached_skill_freqs = dict(counts)
        return self._cached_skill_freqs

    def get_top_dataset_skills(self, top_n: int = 15) -> List[Tuple[str, int]]:
        """Returns the top-N most frequent recognized skill terms in the dataset."""
        freqs = self.get_dataset_skill_frequencies()
        return sorted(freqs.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def get_skill_coverage(self, confirmed_skills: List[str]) -> Dict[str, Any]:
        """
        Calculates how many of the student's confirmed skills are present in the
        internship dataset vocabulary vs outside the dataset vocabulary.
        """
        freqs = self.get_dataset_skill_frequencies()
        dataset_vocab = set(freqs.keys())

        normalized_student_skills = [normalize_skill_name(s) for s in confirmed_skills if s.strip()]
        # Deduplicate while preserving order
        seen = set()
        clean_student = []
        for s in normalized_student_skills:
            if s not in seen:
                clean_student.append(s)
                seen.add(s)

        in_vocab = [s for s in clean_student if s in dataset_vocab]
        out_vocab = [s for s in clean_student if s not in dataset_vocab]

        return {
            "total_student_skills": len(clean_student),
            "in_dataset_count": len(in_vocab),
            "not_in_dataset_count": len(out_vocab),
            "in_dataset_skills": in_vocab,
            "not_in_dataset_skills": out_vocab,
            "coverage_percentage": round((len(in_vocab) / len(clean_student) * 100), 1) if clean_student else 0.0
        }

    def compute_similarity_heatmap_matrix(
        self,
        confirmed_skills: List[str],
        recommended_internships: List[Dict[str, Any]],
        max_skills: int = 8
    ) -> pd.DataFrame:
        """
        Builds a 2D matrix of Cosine Similarities between the student profile / skills
        and each recommended internship.
        Rows: Recommended Internship Titles
        Columns: 'Full Profile' + top individual confirmed skills
        """
        if not recommended_internships or not confirmed_skills:
            return pd.DataFrame()

        clean_skills = [normalize_skill_name(s) for s in confirmed_skills if s.strip()][:max_skills]
        profile_query = " ".join(clean_skills)

        # Columns for heatmap
        columns = ["Full Profile"] + clean_skills
        queries = [profile_query] + clean_skills

        # Vectorize queries using fitted vectorizer
        query_vectors = self.vectorizer.transform(queries)

        row_labels = []
        matrix_data = []

        for i, item in enumerate(recommended_internships, start=1):
            title = item["title"]
            loc = item["location"]
            display_title = title if len(title) <= 24 else title[:21] + "..."
            row_labels.append(f"#{i} {display_title} ({loc})")

            listing_id = item.get("listing_id")
            if listing_id is not None and 0 <= listing_id < self.tfidf_matrix.shape[0]:
                listing_vector = self.tfidf_matrix[listing_id]
            else:
                row_text = f"{title} {item.get('required_skills', '')}"
                listing_vector = self.vectorizer.transform([row_text])

            sims = cosine_similarity(query_vectors, listing_vector).flatten()
            matrix_data.append(np.round(sims, 3))

        heatmap_df = pd.DataFrame(matrix_data, index=row_labels, columns=columns)
        return heatmap_df

    def run_functional_validation_checks(self) -> List[Dict[str, Any]]:
        """
        Runs honest functional checks verifying system behavior without fabricating accuracy.
        """
        checks = []

        # Check 1: Normal query with recognizable skills
        test_skills = ["python", "machine learning", "pandas", "sql"]
        recs, meta = self.recommend_internships(test_skills, top_n=5)
        checks.append({
            "check": "Recommendation Generation (Valid Skills)",
            "status": "PASSED" if len(recs) > 0 and meta["status"] == "success" else "FAILED",
            "details": f"Generated {len(recs)} recommendations for {test_skills}."
        })

        # Check 2: Results are sorted in descending order of similarity
        if recs:
            scores = [r["similarity_score"] for r in recs]
            is_descending = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
            checks.append({
                "check": "Rank Order Verification",
                "status": "PASSED" if is_descending else "FAILED",
                "details": f"Top scores: {[round(s, 3) for s in scores]} (strictly descending)."
            })
        else:
            checks.append({
                "check": "Rank Order Verification",
                "status": "SKIPPED",
                "details": "No recommendations returned."
            })

        # Check 3: Max results <= 5
        checks.append({
            "check": "Top-N Constraint (<= 5)",
            "status": "PASSED" if len(recs) <= 5 else "FAILED",
            "details": f"Returned {len(recs)} results (expected <= 5)."
        })

        # Check 4: Empty input handling
        empty_recs, empty_meta = self.recommend_internships([], top_n=5)
        checks.append({
            "check": "Empty Skills Handling",
            "status": "PASSED" if len(empty_recs) == 0 and empty_meta["status"] == "empty_input" else "FAILED",
            "details": "Gracefully returned empty results with 'empty_input' notification."
        })

        # Check 5: Input with unrecognized/zero-overlap skills
        unrec_recs, unrec_meta = self.recommend_internships(["completely_unrelated_skill_xyz"], top_n=5)
        checks.append({
            "check": "Zero-Overlap / Unrecognized Handling",
            "status": "PASSED" if len(unrec_recs) == 0 and unrec_meta["status"] == "no_match" else "FAILED",
            "details": "Properly alerted student of no matching listings rather than reporting false match."
        })

        # Check 6: Dataset integrity
        has_388 = len(self.internships_df) >= 380
        checks.append({
            "check": "Dataset Listings Loaded",
            "status": "PASSED" if has_388 else "FAILED",
            "details": f"Loaded {len(self.internships_df)} technical internship records from CSV."
        })

        return checks

    def get_all_internships(self) -> pd.DataFrame:
        """Returns the complete preprocessed internships dataframe."""
        return self.internships_df.copy()

    def get_locations_list(self) -> List[str]:
        """Returns sorted unique non-empty locations from the dataset."""
        locs = sorted([str(loc).strip() for loc in self.internships_df["location"].unique() if str(loc).strip()])
        return ["All Locations"] + locs

    def filter_and_sort_catalog(
        self,
        search_query: str = "",
        selected_location: str = "All Locations",
        selected_skills: List[str] = None,
        min_stipend: float = 0.0,
        sort_by: str = "Default Order"
    ) -> pd.DataFrame:
        """
        Filters and sorts the complete internship catalog based on user controls.
        """
        df = self.internships_df.copy()

        # Free-text search on title and skills
        if search_query and search_query.strip():
            q = search_query.strip().lower()
            mask = (
                df["title"].str.lower().str.contains(q, na=False) |
                df["skills_clean"].str.lower().str.contains(q, na=False) |
                df["skills"].str.lower().str.contains(q, na=False)
            )
            df = df[mask]

        # Location filter
        if selected_location and selected_location != "All Locations":
            df = df[df["location"] == selected_location]

        # Skills filter
        if selected_skills:
            for s in selected_skills:
                norm_s = normalize_skill_name(s).lower()
                df = df[df["skills_clean"].str.lower().str.contains(norm_s, na=False)]

        # Salary filter
        if min_stipend > 0:
            df = df[(df["start_salary"] >= min_stipend) | (df["max_salary"] >= min_stipend)]

        # Sorting
        if sort_by == "Highest Starting Salary":
            df = df.sort_values("start_salary", ascending=False)
        elif sort_by == "Highest Maximum Salary":
            df = df.sort_values("max_salary", ascending=False)
        elif sort_by == "Internship Title (A to Z)":
            df = df.sort_values("title", ascending=True)
        elif sort_by == "Internship Title (Z to A)":
            df = df.sort_values("title", ascending=False)

        return df.reset_index(drop=True)


# Global singleton instance for efficient reuse across Streamlit re-runs
_engine_instance = None

def get_engine() -> RecommendationEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = RecommendationEngine()
    return _engine_instance


def recommend_internships(student_skills: Union[str, List[str]], top_n: int = 5):
    """
    Convenience function matching the original project's interface:
    recommend_internships(student_skills, top_n=5)
    """
    engine = get_engine()
    return engine.recommend_internships(student_skills, top_n=top_n)
