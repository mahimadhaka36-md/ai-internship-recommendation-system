"""
mock_interview.py
-----------------
AI Mock Interview Session Manager, Role Suggestion Engine, Question Selector,
Optional Generative AI Question Augmentation, and Performance Analytics.
"""

import json
import os
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from skill_extractor import normalize_skill_name
from ai_provider import CovexcodeProvider, get_ai_provider

QUESTIONS_PATH = os.path.join(
    os.path.dirname(__file__), "data", "interview_questions.json"
)

ROLE_TO_DOMAIN: Dict[str, str] = {
    "AI Engineer": "Data Science / AI",
    "ML Engineer": "Data Science / AI",
    "Data Scientist": "Data Science / AI",
    "Data Analyst": "Data Science / AI",
    "Python Developer": "Software Development",
    "Frontend Developer": "Web Development",
    "Backend Developer": "Software Development",
    "Full Stack Developer": "Web Development",
    "Software Engineer": "Software Development",
    "Cybersecurity Analyst": "Cybersecurity / Networking",
}

ROLE_SKILL_HINTS: Dict[str, List[str]] = {
    "AI Engineer": ["python", "pytorch", "tensorflow", "deep learning", "nlp", "artificial intelligence", "natural language processing"],
    "ML Engineer": ["python", "scikit-learn", "machine learning", "numpy", "pandas", "feature engineering", "model evaluation"],
    "Data Scientist": ["python", "sql", "pandas", "numpy", "statistics", "data science", "machine learning", "matplotlib"],
    "Data Analyst": ["sql", "excel", "power bi", "tableau", "data analytics", "data analysis", "python", "mysql"],
    "Python Developer": ["python", "django", "flask", "fastapi", "sql", "git", "rest api"],
    "Frontend Developer": ["html", "css", "javascript", "react", "typescript", "angular", "vue", "ui/ux"],
    "Backend Developer": ["java", "python", "node.js", "sql", "rest api", "postgresql", "mongodb", "docker", "spring boot"],
    "Full Stack Developer": ["javascript", "react", "node.js", "html", "css", "sql", "mongodb", "express.js", "django"],
    "Software Engineer": ["c++", "java", "python", "data structures", "algorithms", "oop", "git", "system design"],
    "Cybersecurity Analyst": ["linux", "networking", "cybersecurity", "python", "computer networks", "cloud computing"],
}

INTERVIEW_TYPES = ["Technical", "HR", "Behavioral", "Mixed"]
DIFFICULTY_LEVELS = ["Beginner", "Intermediate", "Advanced"]
QUESTION_COUNTS = [5, 10, 15]
SECONDS_PER_QUESTION_BY_DIFFICULTY = {
    "Beginner": 90,
    "Intermediate": 120,
    "Advanced": 150,
}


def load_question_bank(json_path: str = QUESTIONS_PATH) -> List[Dict[str, Any]]:
    """Load and validate the structured interview question bank from disk."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Interview question bank not found at '{json_path}'."
        )
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Interview question bank JSON must be a list of question objects.")
    return data


def suggest_roles_for_student(
    confirmed_skills: Optional[List[str]] = None,
    recommendations: Optional[List[Dict[str, Any]]] = None,
    top_n: int = 3,
) -> List[Dict[str, Any]]:
    """
    Suggest suitable interview roles automatically from confirmed skills
    and existing TF-IDF internship recommendations.
    """
    norm_skills = {
        normalize_skill_name(s) for s in (confirmed_skills or []) if str(s).strip()
    }
    rec_titles_text = " ".join(
        str(r.get("title", "")).lower() + " " + str(r.get("required_skills", "")).lower()
        for r in (recommendations or [])
    )

    scored: List[Dict[str, Any]] = []
    for role, hint_skills in ROLE_SKILL_HINTS.items():
        matched = sorted([s for s in hint_skills if s in norm_skills])
        skill_overlap_score = len(matched) * 2.0

        # Boost from internship recommendation titles/skills
        role_tokens = [tok for tok in role.lower().split() if len(tok) > 1]
        rec_boost = 0.0
        if role.lower() in rec_titles_text:
            rec_boost += 3.0
        elif any(tok in rec_titles_text for tok in role_tokens):
            rec_boost += 1.5
        for hs in hint_skills:
            if hs in rec_titles_text:
                rec_boost += 0.25

        total_score = skill_overlap_score + rec_boost
        if total_score > 0:
            reasons: List[str] = []
            if matched:
                reasons.append(f"Skills: {', '.join(matched[:4])}")
            if rec_boost >= 1.5:
                reasons.append("Aligns with recommended internships")
            scored.append({
                "role": role,
                "domain": ROLE_TO_DOMAIN[role],
                "score": round(total_score, 2),
                "matched_skills": matched,
                "reason": " • ".join(reasons) if reasons else "Matches profile context",
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    if not scored:
        return [
            {
                "role": "Software Engineer",
                "domain": ROLE_TO_DOMAIN["Software Engineer"],
                "score": 0.0,
                "matched_skills": [],
                "reason": "Popular general technical internship track",
            },
            {
                "role": "Python Developer",
                "domain": ROLE_TO_DOMAIN["Python Developer"],
                "score": 0.0,
                "matched_skills": [],
                "reason": "Versatile entry track across data & backend internships",
            },
            {
                "role": "Data Analyst",
                "domain": ROLE_TO_DOMAIN["Data Analyst"],
                "score": 0.0,
                "matched_skills": [],
                "reason": "High-demand analytics internship track",
            },
        ][:top_n]
    return scored[:top_n]


def map_internship_to_interview_config(
    internship_title: str,
    required_skills: str = "",
) -> Dict[str, str]:
    """
    Map an internship listing's title and required skills to the closest
    supported mock interview role and domain.
    """
    combined = f"{internship_title} {required_skills}".lower()

    keyword_role_rules = [
        (["cyber", "security", "network", "soc", "pentest"], "Cybersecurity Analyst"),
        (["ml engineer", "machine learning engineer", "scikit"], "ML Engineer"),
        (["ai engineer", "generative ai", "llm", "nlp", "deep learning", "pytorch", "artificial intelligence"], "AI Engineer"),
        (["data scien", "statistics", "predictive"], "Data Scientist"),
        (["data analy", "power bi", "tableau", "excel", "business intelligence"], "Data Analyst"),
        (["full stack", "mern", "mean", "fullstack"], "Full Stack Developer"),
        (["front end", "frontend", "react", "angular", "vue", "html", "css", "ui/ux"], "Frontend Developer"),
        (["back end", "backend", "django", "flask", "fastapi", "node.js", "nodejs", "spring", "rest api"], "Backend Developer"),
        (["python"], "Python Developer"),
    ]
    for keywords, role in keyword_role_rules:
        if any(kw in combined for kw in keywords):
            return {
                "role": role,
                "domain": ROLE_TO_DOMAIN[role],
                "difficulty": "Intermediate",
                "interview_type": "Mixed",
            }

    return {
        "role": "Software Engineer",
        "domain": ROLE_TO_DOMAIN["Software Engineer"],
        "difficulty": "Intermediate",
        "interview_type": "Mixed",
    }


def generate_optional_ai_questions(
    role: str,
    domain: str,
    difficulty: str,
    interview_type: str,
    count: int = 2,
    internship_context: str = "",
    skills: Optional[List[str]] = None,
    model: Optional[str] = None,
    provider: Optional[CovexcodeProvider] = None,
) -> Tuple[List[Dict[str, Any]], Optional[str], Dict[str, int]]:
    """
    Generate personalized contextual questions via Covexcode when configured.
    Gracefully returns ([], note, empty_usage) if unconfigured, offline, or credits expired,
    preserving the curated 170-question local bank as fallback.
    """
    empty_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    ai = provider or get_ai_provider()
    if not ai.is_configured() or count <= 0:
        return [], None, empty_usage

    res = ai.generate_interview_questions(
        role=role,
        domain=domain,
        difficulty=difficulty,
        interview_type=interview_type,
        count=count,
        skills=skills,
        internship_context=internship_context,
        model=model,
    )
    usage = res.get("usage") or empty_usage
    questions = res.get("questions") or []
    if res.get("ok") and questions:
        note = (
            f"Added {len(questions)} personalized question(s) via Covexcode "
            f"({res.get('model_used', ai.default_model)}; {usage.get('total_tokens', 0)} tokens)."
        )
        return questions, note, usage

    err_msg = res.get("error") or "Covexcode unavailable"
    return (
        [],
        f"Covexcode AI question generation skipped ({err_msg}); using curated 170-question bank.",
        usage,
    )


def select_interview_questions(
    question_bank: List[Dict[str, Any]],
    role: str,
    domain: str,
    interview_type: str,
    difficulty: str,
    num_questions: int = 5,
    exclude_ids: Optional[set] = None,
    seed: Optional[int] = None,
    extra_ai_questions: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Select `num_questions` non-repeating questions prioritized by role, domain,
    difficulty, and interview type. Guarantees no duplicate question IDs or texts.
    """
    rng = random.Random(seed)
    excluded = set(exclude_ids or set())
    selected: List[Dict[str, Any]] = []
    seen_ids = set(excluded)
    seen_texts = set()

    def _add_unique(candidates: List[Dict[str, Any]], limit: int) -> None:
        pool = list(candidates)
        rng.shuffle(pool)
        for q in pool:
            if len(selected) >= limit:
                break
            qid = q.get("question_id")
            qtxt = str(q.get("question_text", "")).strip().lower()
            if qid not in seen_ids and qtxt not in seen_texts:
                selected.append(q)
                seen_ids.add(qid)
                seen_texts.add(qtxt)

    # Include optional AI-generated contextual questions first if provided
    if extra_ai_questions:
        _add_unique(extra_ai_questions, num_questions)

    def _pick_for_type(target_type: str, target_count: int) -> None:
        type_bank = [q for q in question_bank if q.get("question_type") == target_type]
        if target_type == "Technical":
            tiers = [
                [q for q in type_bank if q.get("role") == role and q.get("difficulty") == difficulty],
                [q for q in type_bank if q.get("domain") == domain and q.get("difficulty") == difficulty],
                [q for q in type_bank if q.get("role") == role],
                [q for q in type_bank if q.get("domain") == domain],
                [q for q in type_bank if q.get("difficulty") == difficulty],
                type_bank,
            ]
        else:
            tiers = [
                [q for q in type_bank if q.get("role") == role and q.get("difficulty") == difficulty],
                [q for q in type_bank if q.get("difficulty") == difficulty],
                type_bank,
            ]
        for tier in tiers:
            if len(selected) >= target_count:
                break
            _add_unique(tier, target_count)

    if interview_type == "Mixed":
        # 60% Technical, 20% Behavioral, 20% HR
        n_tech = max(1, round(num_questions * 0.6))
        n_beh = max(1, (num_questions - n_tech) // 2)
        n_hr = max(1, num_questions - n_tech - n_beh)
        _pick_for_type("Technical", min(num_questions, len(selected) + n_tech))
        _pick_for_type("Behavioral", min(num_questions, len(selected) + n_beh))
        _pick_for_type("HR", min(num_questions, len(selected) + n_hr))
    else:
        _pick_for_type(interview_type, num_questions)

    # Final safety top-up from remaining bank if a filtered subset had < num_questions
    if len(selected) < num_questions:
        _add_unique(question_bank, num_questions)

    return selected[:num_questions]


def build_interview_summary(
    session_config: Dict[str, Any],
    questions: List[Dict[str, Any]],
    responses: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate completed interview session metrics, strongest/weakest topics,
    dimension averages, and grounded preparation recommendations.
    """
    question_rows: List[Dict[str, Any]] = []
    attempted_scores: List[float] = []
    skipped_count = 0
    topic_scores: Dict[str, List[float]] = defaultdict(list)
    dim_totals: Dict[str, List[float]] = defaultdict(list)
    missed_concept_counts: Counter = Counter()
    backends_used: Counter = Counter()

    for idx, q in enumerate(questions):
        resp = responses.get(idx)
        if resp is None:
            continue
        eval_res = resp["evaluation"]
        is_skipped = bool(eval_res.get("skipped", False))
        topic = q.get("topic", q.get("question_type", "General"))

        if is_skipped:
            skipped_count += 1
            for mc in eval_res.get("missing_concepts", []):
                missed_concept_counts[mc] += 1
            question_rows.append({
                "index": idx + 1,
                "question_id": q["question_id"],
                "topic": topic,
                "question_type": q["question_type"],
                "difficulty": q["difficulty"],
                "question_text": q["question_text"],
                "student_answer": "",
                "status": "Skipped",
                "score": None,
                "covered_concepts": [],
                "missing_concepts": eval_res.get("missing_concepts", []),
                "backend_used": eval_res.get("backend_used", "Skipped"),
            })
        else:
            score = float(eval_res["overall_score"])
            attempted_scores.append(score)
            topic_scores[topic].append(score)
            for dim_key, dim_val in eval_res.get("dimensions", {}).items():
                dim_totals[dim_key].append(float(dim_val))
            for mc in eval_res.get("missing_concepts", []):
                missed_concept_counts[mc] += 1
            backends_used[eval_res.get("backend_used", "Evaluator")] += 1

            question_rows.append({
                "index": idx + 1,
                "question_id": q["question_id"],
                "topic": topic,
                "question_type": q["question_type"],
                "difficulty": q["difficulty"],
                "question_text": q["question_text"],
                "student_answer": resp.get("answer_text", ""),
                "status": "Attempted",
                "score": score,
                "covered_concepts": eval_res.get("covered_concepts", []),
                "missing_concepts": eval_res.get("missing_concepts", []),
                "backend_used": eval_res.get("backend_used", "Evaluator"),
            })

    attempted_count = len(attempted_scores)
    avg_score = round(float(np.mean(attempted_scores)), 1) if attempted_scores else 0.0

    # Overall practice score penalizes unanswered/skipped questions lightly or reports attempted avg clearly
    completion_ratio = len(question_rows) / float(max(1, len(questions)))
    overall_practice_score = avg_score if attempted_count > 0 else 0.0

    topic_averages = [
        {"topic": t, "avg_score": round(float(np.mean(vals)), 1), "count": len(vals)}
        for t, vals in topic_scores.items()
    ]
    topic_averages.sort(key=lambda x: x["avg_score"], reverse=True)

    strongest_topics = topic_averages[:3]
    weakest_topics = list(reversed(topic_averages[-3:])) if len(topic_averages) > 1 else topic_averages

    dimension_averages = {
        dim: round(float(np.mean(vals)), 1) if vals else 0.0
        for dim, vals in dim_totals.items()
    }

    # Grounded preparation suggestions
    suggestions: List[str] = []
    if attempted_count == 0:
        suggestions.append(
            "No questions were answered in this session. Try answering at least 3-5 questions without skipping to receive topic diagnostics."
        )
    else:
        if weakest_topics and weakest_topics[0]["avg_score"] < 70.0:
            weak_names = ", ".join(
                f"{t['topic']} ({t['avg_score']}/100)"
                for t in weakest_topics if t["avg_score"] < 75.0
            )
            if weak_names:
                suggestions.append(f"Prioritize revision on lower-scoring topics: {weak_names}.")
        if missed_concept_counts:
            top_missed = [c for c, _ in missed_concept_counts.most_common(6)]
            suggestions.append(
                f"Review these frequently missed concepts before your next practice run: {', '.join(top_missed)}."
            )
        if dimension_averages.get("completeness", 100.0) < 60.0:
            suggestions.append(
                "Expand answers with clear definitions, cause-and-effect reasoning, and one concrete project example."
            )
        if skipped_count > 0:
            suggestions.append(
                f"Revisit the {skipped_count} skipped question(s) using the reference answers in the breakdown below."
            )
        if not suggestions:
            suggestions.append(
                f"Consistent performance across {attempted_count} questions. Try increasing difficulty or enabling timed mode next."
            )

    return {
        "session_config": session_config,
        "total_questions": len(questions),
        "questions_seen": len(question_rows),
        "questions_attempted": attempted_count,
        "questions_skipped": skipped_count,
        "completion_pct": round(completion_ratio * 100.0, 1),
        "overall_practice_score": overall_practice_score,
        "average_answer_score": avg_score,
        "strongest_topics": strongest_topics,
        "weakest_topics": weakest_topics,
        "topic_averages": topic_averages,
        "dimension_averages": dimension_averages,
        "most_missed_concepts": missed_concept_counts.most_common(8),
        "question_rows": question_rows,
        "preparation_suggestions": suggestions,
        "backends_used": dict(backends_used),
        "disclaimer": (
            "Practice scores reflect semantic and concept alignment against reference answers for self-study only. "
            "They are NOT hiring probabilities or validated predictors of real interview outcomes."
        ),
    }


def format_interview_report_markdown(summary: Dict[str, Any]) -> str:
    """Format a downloadable Markdown performance report for the completed mock interview."""
    cfg = summary["session_config"]
    lines = [
        "# AI Mock Interview Performance Report",
        "",
        f"- **Target Role:** {cfg.get('role', 'N/A')} ({cfg.get('domain', 'N/A')})",
        f"- **Interview Type:** {cfg.get('interview_type', 'N/A')}",
        f"- **Difficulty:** {cfg.get('difficulty', 'N/A')}",
        f"- **Mode:** {'Timed' if cfg.get('timed_mode') else 'Untimed'}",
        f"- **Overall Practice Score:** {summary['overall_practice_score']} / 100",
        f"- **Questions Attempted:** {summary['questions_attempted']} / {summary['total_questions']} (Skipped: {summary['questions_skipped']})",
        f"- **Average Answer Score:** {summary['average_answer_score']} / 100",
        "",
        "> **Academic Disclaimer:** " + summary["disclaimer"],
        "",
        "## Rubric Dimension Averages",
    ]
    for dim, val in summary["dimension_averages"].items():
        lines.append(f"- **{dim.replace('_', ' ').title()}:** {val} / 100")

    lines.append("")
    lines.append("## Strongest & Weakest Topics")
    if summary["strongest_topics"]:
        lines.append("- **Strongest:** " + ", ".join(f"{t['topic']} ({t['avg_score']})" for t in summary["strongest_topics"]))
    if summary["weakest_topics"]:
        lines.append("- **Weakest:** " + ", ".join(f"{t['topic']} ({t['avg_score']})" for t in summary["weakest_topics"]))

    lines.append("")
    lines.append("## Personalized Preparation Suggestions")
    for s in summary["preparation_suggestions"]:
        lines.append(f"- {s}")

    lines.append("")
    lines.append("## Question-Wise Performance")
    for row in summary["question_rows"]:
        score_str = f"{row['score']}/100" if row["score"] is not None else "Skipped"
        lines.extend([
            f"### Q{row['index']}: {row['topic']} ({row['question_type']} • {row['difficulty']}) — {score_str}",
            f"**Question:** {row['question_text']}",
            f"**Your Answer:** {row['student_answer'] or '_Skipped / No answer_'}",
            f"**Covered Concepts:** {', '.join(row['covered_concepts']) or 'None'}",
            f"**Missing Concepts:** {', '.join(row['missing_concepts']) or 'None'}",
            "",
        ])

    return "\n".join(lines)
