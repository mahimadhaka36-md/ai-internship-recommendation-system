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
from peer_match import PeerMatchEngine, STUDENT_DATA_PATH
from ai_provider import CovexcodeProvider
from interview_evaluator import InterviewAnswerEvaluator
from mock_interview import (
    QUESTIONS_PATH,
    ROLE_TO_DOMAIN,
    build_interview_summary,
    format_interview_report_markdown,
    load_question_bank,
    map_internship_to_interview_config,
    select_interview_questions,
    suggest_roles_for_student,
)
import charts
import matplotlib.pyplot as plt


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


class TestPeerMatchSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = RecommendationEngine()
        cls.peer_engine = PeerMatchEngine(STUDENT_DATA_PATH)

    def test_14_peer_dataset_integrity(self):
        """Verify synthetic_student_profiles_30.csv has 30 profiles and required fields."""
        df = self.peer_engine.get_all_students()
        self.assertEqual(len(df), 30)
        self.assertEqual(df["Career_Field"].nunique(), 10)
        self.assertTrue((df["cgpa_10"] > 0).all() and (df["cgpa_10"] <= 10.0).all())

    def test_15_knn_peer_match_ranking_and_career(self):
        """Verify KNN ranking order, K limit, and career field recommendation."""
        profile = {
            "student_name": "Test DS",
            "major": "Computer Science",
            "university_year": 4,
            "cgpa": 8.5,
            "cgpa_scale": 10.0,
            "technical_skills": ["python", "sql", "pandas", "numpy", "statistics"],
            "programming_skill": 8,
            "problem_solving": 7,
            "communication_skills": 7,
            "projects_completed": 4,
            "certifications": 2,
            "internships": 2,
        }
        for metric in ["euclidean", "cosine"]:
            res = self.peer_engine.find_similar_students(
                profile, k=5, metric=metric, dataset_skill_freqs=self.engine.get_dataset_skill_frequencies()
            )
            peers = res["peers"]
            self.assertEqual(len(peers), 5)
            dists = [p["distance"] for p in peers]
            sims = [p["similarity_score"] for p in peers]
            for i in range(len(dists) - 1):
                self.assertLessEqual(dists[i], dists[i + 1])
                self.assertGreaterEqual(sims[i], sims[i + 1])
            self.assertEqual(res["career_recommendations"][0]["career_field"], "Data Scientist")

    def test_16_knn_cgpa_scale_normalization_and_edge_cases(self):
        """Verify 4.0 vs 10.0 CGPA normalization, K clamping, unseen major, and empty skills."""
        base = {
            "student_name": "Scale Check",
            "major": "Computer Science",
            "university_year": 3,
            "cgpa": 4.0,
            "cgpa_scale": 4.0,
            "technical_skills": ["python", "sql"],
            "programming_skill": 7,
            "problem_solving": 7,
            "communication_skills": 7,
            "projects_completed": 3,
            "certifications": 1,
            "internships": 1,
        }
        res_4 = self.peer_engine.find_similar_students(base, k=5)
        res_10 = self.peer_engine.find_similar_students({**base, "cgpa": 10.0, "cgpa_scale": 10.0}, k=5)
        self.assertEqual(
            [p["student_id"] for p in res_4["peers"]],
            [p["student_id"] for p in res_10["peers"]],
        )

        edge = self.peer_engine.find_similar_students(
            {**base, "major": "Biomedical Engineering", "technical_skills": []},
            k=99,
            metric="cosine",
        )
        self.assertEqual(len(edge["peers"]), 30)
        self.assertTrue(edge["empty_skills_input"])

    def test_17_peer_skill_gaps_and_charts(self):
        """Verify skill gap detection and Matplotlib peer chart rendering."""
        profile = {
            "student_name": "Frontend Learner",
            "major": "Computer Science",
            "university_year": 2,
            "cgpa": 7.5,
            "cgpa_scale": 10.0,
            "technical_skills": ["html", "css", "javascript"],
            "programming_skill": 8,
            "problem_solving": 6,
            "communication_skills": 9,
            "projects_completed": 4,
            "certifications": 1,
            "internships": 0,
        }
        res = self.peer_engine.find_similar_students(
            profile, k=5, dataset_skill_freqs=self.engine.get_dataset_skill_frequencies()
        )
        gap_skills = [g["skill"] for g in res["skill_gaps"]]
        self.assertIn("react", gap_skills)

        fig1 = charts.plot_peer_competency_radar(res["user_profile"], res["peer_average"], res["peers"][0])
        fig2 = charts.plot_peer_experience_comparison(res["user_profile"], res["peer_average"], res["peers"][0])
        fig3 = charts.plot_career_support_chart(res["career_recommendations"])
        fig4 = charts.plot_knn_neighbors_chart(res["all_ranked"], res["k"], res["metric"])
        for fig in [fig1, fig2, fig3, fig4]:
            self.assertIsNotNone(fig)
            plt.close(fig)

    def test_18_knn_validation_checks(self):
        """Verify automated KNN functional validation checks all pass."""
        checks = self.peer_engine.run_knn_validation_checks()
        self.assertGreaterEqual(len(checks), 5)
        for c in checks:
            self.assertEqual(c["status"], "PASSED", f"Check failed: {c}")


class TestMockInterviewSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.bank = load_question_bank(QUESTIONS_PATH)

    def test_19_interview_question_bank_integrity(self):
        """Verify >= 150 unique questions across all 10 required roles and valid schemas."""
        self.assertGreaterEqual(len(self.bank), 150)
        ids = [q["question_id"] for q in self.bank]
        texts = [q["question_text"] for q in self.bank]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate question IDs found.")
        self.assertEqual(len(texts), len(set(texts)), "Duplicate question texts found.")

        roles_present = {q["role"] for q in self.bank}
        for required_role in ROLE_TO_DOMAIN:
            self.assertIn(required_role, roles_present)

        for q in self.bank:
            for field in [
                "question_id", "role", "domain", "difficulty", "question_type",
                "topic", "question_text", "reference_answer", "expected_concepts", "evaluation_rubric",
            ]:
                self.assertIn(field, q)
                self.assertTrue(q[field])

    def test_20_role_suggestions_and_non_repeating_selection(self):
        """Verify role suggestions, internship mapping, and non-repeating 5/10/15 selection."""
        suggestions = suggest_roles_for_student(
            confirmed_skills=["python", "pytorch", "deep learning", "nlp"],
            recommendations=[{"title": "Artificial Intelligence (AI)", "required_skills": "Python NLP"}],
        )
        self.assertEqual(suggestions[0]["role"], "AI Engineer")

        mapped = map_internship_to_interview_config("Cybersecurity Analyst Intern", "Linux Networking")
        self.assertEqual(mapped["role"], "Cybersecurity Analyst")

        for count in [5, 10, 15]:
            qs = select_interview_questions(
                self.bank,
                role="ML Engineer",
                domain="Data Science / AI",
                interview_type="Mixed",
                difficulty="Intermediate",
                num_questions=count,
                seed=42,
            )
            self.assertEqual(len(qs), count)
            self.assertEqual(len({q["question_id"] for q in qs}), count, "Repeated questions within session.")

        # Verify exclude_ids prevents repeating questions across sessions
        first_5 = select_interview_questions(
            self.bank, "ML Engineer", "Data Science / AI", "Technical", "Beginner", num_questions=5, seed=1
        )
        excluded = {q["question_id"] for q in first_5}
        next_5 = select_interview_questions(
            self.bank, "ML Engineer", "Data Science / AI", "Technical", "Beginner", num_questions=5, exclude_ids=excluded, seed=2
        )
        self.assertTrue(excluded.isdisjoint({q["question_id"] for q in next_5}))

    def test_21_evaluator_fallback_empty_skipped_and_quality(self):
        """Verify TF-IDF fallback when model unavailable, empty answers, skipped questions, and scoring quality."""
        sample_q = next(q for q in self.bank if q["question_id"] == "MLE-TECH-BEG-01")
        fallback_eval = InterviewAnswerEvaluator(force_fallback=True)
        status = fallback_eval.get_backend_status()
        self.assertFalse(status["using_transformer"])
        self.assertIn("Fallback", status["backend_label"])

        # Unavailable model name triggers fast automatic fallback without hanging
        unavail_eval = InterviewAnswerEvaluator(model_name="nonexistent-model-xyz-offline")
        unavail_status = unavail_eval.get_backend_status()
        self.assertFalse(unavail_status["using_transformer"])
        self.assertIn("Fallback", unavail_status["backend_label"])

        # Empty answer
        empty_res = fallback_eval.evaluate_answer(sample_q, "   ", skipped=False)
        self.assertEqual(empty_res["status"], "empty")
        self.assertEqual(empty_res["overall_score"], 0.0)
        self.assertEqual(empty_res["covered_concepts"], [])
        self.assertEqual(len(empty_res["missing_concepts"]), len(sample_q["expected_concepts"]))

        # Skipped question
        skip_res = fallback_eval.evaluate_answer(sample_q, "", skipped=True)
        self.assertEqual(skip_res["status"], "skipped")
        self.assertTrue(skip_res["skipped"])
        self.assertIsNone(skip_res["overall_score"])

        # Strong vs weak answer discrimination
        strong_ans = (
            "Overfitting occurs when a model memorizes training data noise and fails on validation or test sets, "
            "hurting generalization. We detect it by comparing training vs validation errors via cross-validation, "
            "and reduce it using L1/L2 regularization, simpler models, or more training data."
        )
        weak_ans = "I don't know maybe it is a bug."
        strong_res = fallback_eval.evaluate_answer(sample_q, strong_ans, skipped=False)
        weak_res = fallback_eval.evaluate_answer(sample_q, weak_ans, skipped=False)
        self.assertGreater(strong_res["overall_score"], weak_res["overall_score"] + 30.0)
        self.assertIn("overfitting", strong_res["covered_concepts"])
        self.assertIn("regularization", strong_res["covered_concepts"])

    def test_22_transformer_or_fallback_summary_and_charts(self):
        """Verify end-to-end evaluation, dashboard summary, markdown report, and interview charts."""
        qs = select_interview_questions(
            self.bank, "Data Scientist", "Data Science / AI", "Mixed", "Beginner", num_questions=5, seed=7
        )
        evaluator = InterviewAnswerEvaluator()
        responses = {
            0: {
                "answer_text": qs[0]["reference_answer"],
                "evaluation": evaluator.evaluate_answer(qs[0], qs[0]["reference_answer"], skipped=False),
            },
            1: {
                "answer_text": "",
                "evaluation": evaluator.evaluate_answer(qs[1], "", skipped=True),
            },
            2: {
                "answer_text": "We should check data and try something simple.",
                "evaluation": evaluator.evaluate_answer(qs[2], "We should check data and try something simple.", skipped=False),
            },
        }
        summary = build_interview_summary(
            {
                "role": "Data Scientist",
                "domain": "Data Science / AI",
                "interview_type": "Mixed",
                "difficulty": "Beginner",
                "timed_mode": False,
            },
            qs,
            responses,
        )
        self.assertEqual(summary["questions_attempted"], 2)
        self.assertEqual(summary["questions_skipped"], 1)
        self.assertGreater(summary["average_answer_score"], 0.0)
        self.assertIn("not hiring probabilities", summary["disclaimer"].lower())

        report = format_interview_report_markdown(summary)
        self.assertIn("# AI Mock Interview Performance Report", report)
        self.assertIn("Data Scientist", report)

        fig1 = charts.plot_interview_question_scores_chart(summary["question_rows"])
        fig2 = charts.plot_interview_rubric_dimensions_chart(summary["dimension_averages"])
        fig3 = charts.plot_interview_topic_performance_chart(summary["topic_averages"])
        for fig in [fig1, fig2, fig3]:
            self.assertIsNotNone(fig)
            plt.close(fig)


class TestCovexcodeProvider(unittest.TestCase):

    def test_23_provider_config_and_error_handling(self):
        """Verify CovexcodeProvider key masking, unconfigured safety, and HTTP error handling."""
        from unittest.mock import MagicMock, patch

        # Unconfigured provider
        unconf = CovexcodeProvider(api_key="")
        self.assertFalse(unconf.is_configured())
        tc = unconf.test_connection()
        self.assertFalse(tc["ok"])
        self.assertEqual(tc["error_kind"], "not_configured")

        # Key masking never reveals middle of key
        conf = CovexcodeProvider(api_key="bx-live-testsecret1234")
        status = conf.get_status()
        self.assertTrue(status["configured"])
        self.assertNotIn("testsecret", status["key_masked"])

        # Expired credits / 402 error handled gracefully
        mock_402 = MagicMock()
        mock_402.status_code = 402
        mock_402.json.return_value = {"error": {"message": "event allowance exhausted"}}
        mock_402.text = "event allowance exhausted"
        with patch("ai_provider.requests.post", return_value=mock_402):
            res_402 = conf.chat_completion([{"role": "user", "content": "Hi"}])
            self.assertFalse(res_402["ok"])
            self.assertEqual(res_402["error_kind"], "credits_exhausted")

        # Rate limit / 429 error handled gracefully
        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_429.json.return_value = {"error": {"message": "rate limit exceeded"}}
        mock_429.text = "rate limit exceeded"
        with patch("ai_provider.requests.post", return_value=mock_429):
            res_429 = conf.chat_completion([{"role": "user", "content": "Hi"}])
            self.assertFalse(res_429["ok"])
            self.assertEqual(res_429["error_kind"], "rate_limit")

        # Valid JSON parsing with code fences
        mock_200 = MagicMock()
        mock_200.status_code = 200
        mock_200.json.return_value = {
            "model": "deepseek-flash",
            "choices": [{"message": {"content": "```json\n{\"ok\": true}\n```"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        with patch("ai_provider.requests.post", return_value=mock_200):
            res_json = conf.chat_json([{"role": "user", "content": "Return json"}])
            self.assertTrue(res_json["ok"])
            self.assertEqual(res_json["data"], {"ok": True})
            self.assertEqual(res_json["usage"]["total_tokens"], 15)

    def test_24_covexcode_mock_interview_integration_and_fallback(self):
        """Verify Mock Interview integrates Covexcode AI and falls back seamlessly when credits expire."""
        from unittest.mock import MagicMock, patch
        from mock_interview import generate_optional_ai_questions

        bank = load_question_bank(QUESTIONS_PATH)
        sample_q = next(q for q in bank if q["question_id"] == "MLE-TECH-BEG-01")
        provider = CovexcodeProvider(api_key="bx-live-testsecret1234")

        # 1) Simulate successful Covexcode question generation
        mock_q_resp = MagicMock()
        mock_q_resp.status_code = 200
        mock_q_resp.json.return_value = {
            "model": "deepseek-flash",
            "choices": [{
                "message": {
                    "content": '[{"topic":"ML Pipelines","question_type":"Technical","question_text":"How do you prevent feature leakage in pipelines?","reference_answer":"Fit transformers only on training folds inside a Pipeline.","expected_concepts":["pipeline","leakage","fit","transform"],"evaluation_rubric":{"technical_correctness":"Accurate","concept_coverage":"Covers pipeline","completeness":"Complete","clarity":"Clear"}}]'
                }
            }],
            "usage": {"prompt_tokens": 80, "completion_tokens": 60, "total_tokens": 140},
        }
        with patch("ai_provider.requests.post", return_value=mock_q_resp):
            ai_qs, note, usage = generate_optional_ai_questions(
                "ML Engineer", "Data Science / AI", "Intermediate", "Technical", count=1, provider=provider
            )
            self.assertEqual(len(ai_qs), 1)
            self.assertEqual(usage["total_tokens"], 140)
            self.assertIn("Covexcode", note)

        # 2) Simulate expired credits -> graceful fallback to local 170-question bank & evaluator
        mock_expired = MagicMock()
        mock_expired.status_code = 402
        mock_expired.json.return_value = {"error": {"message": "Out of credit"}}
        mock_expired.text = "Out of credit"
        with patch("ai_provider.requests.post", return_value=mock_expired):
            ai_qs_exp, note_exp, _ = generate_optional_ai_questions(
                "ML Engineer", "Data Science / AI", "Intermediate", "Technical", count=1, provider=provider
            )
            self.assertEqual(ai_qs_exp, [])
            self.assertIn("170-question bank", note_exp)

            ev = InterviewAnswerEvaluator()
            ans = "Overfitting happens when a model memorizes training data instead of generalizing to validation sets; regularization and cross-validation reduce it."
            eval_res = ev.evaluate_answer(sample_q, ans, skipped=False, use_ai=True, ai_provider=provider)
            self.assertEqual(eval_res["status"], "evaluated")
            self.assertFalse(eval_res["ai_evaluation_used"])
            self.assertIsNotNone(eval_res["ai_fallback_note"])
            self.assertGreater(eval_res["overall_score"], 50.0)


if __name__ == "__main__":
    unittest.main()

