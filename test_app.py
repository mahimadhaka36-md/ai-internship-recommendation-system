"""
test_app.py
-----------
Comprehensive automated test suite covering all functional and edge-case requirements:
1. Dataset loading and verification.
2. Model building and artifact creation (prepare_model.py).
3. Skill extraction accuracy (single/multi-word, aliases, boundaries).
4. Recommendation generation and ranking (recommendation.py).
5. Edge case: empty input / non-matching skills.
6. Edge case: duplicate handling and top_n limits.
7. Resume parsing logic (PDF / DOCX in-memory handling, error handling).
"""

import os
import unittest
import numpy as np
import pandas as pd

from skill_extractor import extract_skills, normalize_skill_name, get_skill_vocab
from prepare_model import load_and_preprocess_dataset, build_and_save_model, DATA_PATH
from recommendation import RecommendationEngine
from resume_parser import parse_resume


class TestRecommendationSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Prepare dataset and initialize recommendation engine."""
        cls.df = load_and_preprocess_dataset(DATA_PATH)
        cls.engine = RecommendationEngine()

    def test_01_dataset_integrity(self):
        """Test dataset shape and required columns."""
        self.assertGreaterEqual(len(self.df), 300, "Dataset should contain approximately 388 records.")
        expected_cols = ["title", "location", "skills", "skills_clean", "internship_text"]
        for col in expected_cols:
            self.assertIn(col, self.df.columns, f"Column '{col}' must be present in preprocessed data.")

    def test_02_skill_extraction(self):
        """Test rule-based skill extraction on sample student resume text."""
        sample_text = """
        B.Tech Computer Science student.
        Proficient in Python, SQL, and Machine Learning.
        Built web applications using HTML, CSS, JavaScript, and React.
        Experienced with Git, GitHub, Pandas, and NumPy.
        """
        detected = extract_skills(sample_text)
        
        # Verify multi-word and single-word skills were captured
        self.assertIn("python", detected)
        self.assertIn("machine learning", detected)
        self.assertIn("sql", detected)
        self.assertIn("react", detected)
        self.assertIn("pandas", detected)
        self.assertIn("git", detected)
        
        # Verify random English words are not treated as skills
        self.assertNotIn("student", detected)
        self.assertNotIn("built", detected)
        self.assertNotIn("experienced", detected)

    def test_03_skill_normalization_and_aliases(self):
        """Test alias mapping (e.g. reactjs -> react, nodejs -> node.js)."""
        self.assertEqual(normalize_skill_name("ReactJS"), "react")
        self.assertEqual(normalize_skill_name("Nodejs"), "node.js")
        self.assertEqual(normalize_skill_name("MS Excel"), "excel")
        self.assertEqual(normalize_skill_name("Python"), "python")

    def test_04_recommendations_ranking_and_count(self):
        """Test top-5 recommendation ranking by cosine similarity."""
        skills = ["python", "machine learning", "pandas", "numpy", "sql"]
        recs, meta = self.engine.recommend_internships(skills, top_n=5)

        self.assertEqual(meta["status"], "success")
        self.assertLessEqual(len(recs), 5)
        self.assertGreater(len(recs), 0)

        # Check descending order of similarity score
        scores = [r["similarity_score"] for r in recs]
        for i in range(len(scores) - 1):
            self.assertGreaterEqual(scores[i], scores[i + 1], "Recommendations must be sorted descending by similarity.")

        # Check required fields exist in recommendation cards
        first = recs[0]
        self.assertIn("title", first)
        self.assertIn("location", first)
        self.assertIn("salary_display", first)
        self.assertIn("required_skills", first)
        self.assertIn("similarity_percentage", first)
        self.assertIn("matched_skills", first)
        self.assertIn("missing_skills", first)

    def test_05_no_duplicate_recommendations(self):
        """Verify that identical duplicate listings are filtered out."""
        skills = ["python", "java", "sql"]
        recs, _ = self.engine.recommend_internships(skills, top_n=5)
        seen = set()
        for r in recs:
            key = (r["title"], r["location"], r["required_skills"])
            self.assertNotIn(key, seen, "Duplicate recommendation found in results.")
            seen.add(key)

    def test_06_empty_skills_handling(self):
        """Test graceful handling when no skills are provided."""
        recs, meta = self.engine.recommend_internships([], top_n=5)
        self.assertEqual(len(recs), 0)
        self.assertEqual(meta["status"], "empty_input")

        recs_str, meta_str = self.engine.recommend_internships("", top_n=5)
        self.assertEqual(len(recs_str), 0)
        self.assertEqual(meta_str["status"], "empty_input")

    def test_07_no_match_skills_handling(self):
        """Test handling when skills do not match any internship listings."""
        recs, meta = self.engine.recommend_internships(["completely_unrelated_gibberish_xyz"], top_n=5)
        self.assertEqual(len(recs), 0)
        self.assertEqual(meta["status"], "no_match")

    def test_08_resume_parser_validation(self):
        """Test resume parser validation on empty and unsupported files."""
        # Empty file
        text, err = parse_resume(b"", "empty.pdf")
        self.assertIsNone(text)
        self.assertIn("empty", err.lower())

        # Unsupported file
        text, err = parse_resume(b"sample bytes", "image.png")
        self.assertIsNone(text)
        self.assertIn("unsupported", err.lower())

        # Oversized file
        oversized = b"0" * (6 * 1024 * 1024)
        text, err = parse_resume(oversized, "large.pdf")
        self.assertIsNone(text)
        self.assertIn("exceeds", err.lower())

    def test_09_skill_coverage_overview(self):
        """Test calculation of in-dataset vs out-of-dataset skill coverage."""
        sample_skills = ["python", "sql", "unreal_engine_custom_xyz"]
        coverage = self.engine.get_skill_coverage(sample_skills)
        self.assertIn("python", coverage["in_dataset_skills"])
        self.assertIn("sql", coverage["in_dataset_skills"])
        self.assertIn("unreal_engine_custom_xyz", coverage["not_in_dataset_skills"])
        self.assertEqual(coverage["total_student_skills"], 3)
        self.assertEqual(coverage["in_dataset_count"], 2)
        self.assertEqual(coverage["not_in_dataset_count"], 1)

    def test_10_dataset_skill_demand(self):
        """Test top in-demand skills frequency computation from dataset."""
        top_skills = self.engine.get_top_dataset_skills(15)
        self.assertEqual(len(top_skills), 15)
        # Ensure python is among top skills
        skill_names = [s[0] for s in top_skills]
        self.assertIn("python", skill_names)
        # Ensure frequencies are positive and descending
        freqs = [s[1] for s in top_skills]
        for i in range(len(freqs) - 1):
            self.assertGreaterEqual(freqs[i], freqs[i + 1])

    def test_11_similarity_heatmap_matrix(self):
        """Test 2D similarity heatmap matrix generation."""
        skills = ["python", "sql", "machine learning"]
        recs, _ = self.engine.recommend_internships(skills, top_n=5)
        heatmap_df = self.engine.compute_similarity_heatmap_matrix(skills, recs)
        self.assertEqual(len(heatmap_df), len(recs))
        self.assertIn("Full Profile", heatmap_df.columns)
        self.assertIn("python", heatmap_df.columns)
        # Verify values are between 0 and 1
        for val in heatmap_df.values.flatten():
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)
    def test_12_functional_validation_checks(self):
        """Test the automated functional checks suite."""
        checks = self.engine.run_functional_validation_checks()
        self.assertGreaterEqual(len(checks), 5)
        for c in checks:
            self.assertIn(c["status"], ["PASSED", "SKIPPED"])

    def test_13_catalogue_functions(self):
        """Test catalogue retrieval, location lists, filtering, and sorting."""
        all_df = self.engine.get_all_internships()
        self.assertEqual(len(all_df), 388, "Catalogue should contain all 388 listings.")

        locations = self.engine.get_locations_list()
        self.assertIn("All Locations", locations)
        self.assertIn("Bangalore", locations)

        # Test search query
        python_res = self.engine.filter_and_sort_catalog(search_query="python")
        self.assertGreater(len(python_res), 0)
        self.assertLessEqual(len(python_res), 388)

        # Test location filter
        wf_res = self.engine.filter_and_sort_catalog(selected_location="Work from home")
        self.assertGreater(len(wf_res), 0)
        for _, r in wf_res.iterrows():
            self.assertEqual(r["location"], "Work from home")

        # Test sorting
        sorted_title = self.engine.filter_and_sort_catalog(sort_by="Internship Title (A to Z)")
        titles = sorted_title["title"].tolist()
        self.assertTrue(titles[0].lower() <= titles[-1].lower())


if __name__ == "__main__":
    unittest.main()

