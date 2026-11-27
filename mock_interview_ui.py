"""
mock_interview_ui.py
--------------------
Streamlit UI for the AI Mock Interview module.

Provides:
1. Interview Setup (role, domain, type, difficulty, question count, timed/untimed, auto-suggested roles)
2. Active Interview Interface (question progress, optional countdown timer, answer input, Submit/Skip/Next/End)
3. Per-Answer ML Feedback (practice score, covered/missing concepts, rubric breakdown, reference & improved answer)
4. Final Interview Dashboard (overall score, strongest/weakest topics, charts, preparation suggestions, downloadable report)
"""

import time
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import streamlit as st

import charts
from ai_provider import SUPPORTED_COVEXCODE_MODELS, get_ai_provider
from interview_evaluator import InterviewAnswerEvaluator, RUBRIC_WEIGHTS
from mock_interview import (
    DIFFICULTY_LEVELS,
    INTERVIEW_TYPES,
    QUESTION_COUNTS,
    ROLE_TO_DOMAIN,
    SECONDS_PER_QUESTION_BY_DIFFICULTY,
    build_interview_summary,
    format_interview_report_markdown,
    generate_optional_ai_questions,
    load_question_bank,
    map_internship_to_interview_config,
    select_interview_questions,
    suggest_roles_for_student,
)


@st.cache_resource(show_spinner=False)
def get_cached_question_bank() -> List[Dict[str, Any]]:
    """Load and cache the curated interview question bank once."""
    return load_question_bank()


@st.cache_resource(show_spinner=False)
def get_cached_interview_evaluator() -> InterviewAnswerEvaluator:
    """Create and cache the evaluator instance (underlying transformer loads lazily on first evaluation)."""
    return InterviewAnswerEvaluator()


def init_interview_state() -> None:
    """Initialize session state keys for the AI Mock Interview module."""
    if "interview_stage" not in st.session_state:
        st.session_state.interview_stage = "setup"
    if "interview_config" not in st.session_state:
        provider_status = get_ai_provider().get_status()
        st.session_state.interview_config = {
            "role": "Software Engineer",
            "domain": "Software Development",
            "interview_type": "Mixed",
            "difficulty": "Intermediate",
            "num_questions": 5,
            "timed_mode": False,
            "internship_context": "",
            "use_ai": provider_status["configured"],
            "ai_model": provider_status["default_model"],
        }
    if "interview_questions" not in st.session_state:
        st.session_state.interview_questions = []
    if "interview_current_idx" not in st.session_state:
        st.session_state.interview_current_idx = 0
    if "interview_responses" not in st.session_state:
        st.session_state.interview_responses = {}
    if "interview_used_ids" not in st.session_state:
        st.session_state.interview_used_ids = set()
    if "interview_question_start_ts" not in st.session_state:
        st.session_state.interview_question_start_ts = time.time()
    if "interview_ai_note" not in st.session_state:
        st.session_state.interview_ai_note = None
    if "covexcode_tokens_used" not in st.session_state:
        st.session_state.covexcode_tokens_used = 0


def launch_interview_from_internship(
    internship_title: str,
    required_skills: str = "",
    navigate_to: Callable[[Any], None] = None,
) -> None:
    """
    Pre-configure the Mock Interview setup from a specific internship listing
    and navigate directly to the AI Mock Interview page.
    """
    init_interview_state()
    mapped = map_internship_to_interview_config(internship_title, required_skills)
    provider_status = get_ai_provider().get_status()
    st.session_state.interview_config = {
        "role": mapped["role"],
        "domain": mapped["domain"],
        "interview_type": mapped["interview_type"],
        "difficulty": mapped["difficulty"],
        "num_questions": 5,
        "timed_mode": False,
        "internship_context": internship_title,
        "use_ai": provider_status["configured"],
        "ai_model": provider_status["default_model"],
    }
    st.session_state.interview_stage = "setup"
    st.session_state.interview_questions = []
    st.session_state.interview_responses = {}
    st.session_state.interview_current_idx = 0
    if navigate_to is not None:
        navigate_to("mock_interview")


def reset_interview_session(clear_used_history: bool = False) -> None:
    """Reset the active interview run cleanly back to setup."""
    st.session_state.interview_stage = "setup"
    st.session_state.interview_questions = []
    st.session_state.interview_current_idx = 0
    st.session_state.interview_responses = {}
    st.session_state.interview_ai_note = None
    if clear_used_history:
        st.session_state.interview_used_ids = set()


def _start_new_interview(config: Dict[str, Any]) -> None:
    """Select non-repeating questions and transition to the active interview screen."""
    bank = get_cached_question_bank()
    extra_ai: List[Dict[str, Any]] = []
    ai_note = None
    if config.get("use_ai"):
        skills = list(st.session_state.get("confirmed_skills", []))
        if not skills and st.session_state.get("peer_profile"):
            skills = list(st.session_state.peer_profile.get("technical_skills", []))
        extra_ai, ai_note, q_usage = generate_optional_ai_questions(
            role=config["role"],
            domain=config["domain"],
            difficulty=config["difficulty"],
            interview_type=config["interview_type"],
            count=2,
            internship_context=config.get("internship_context", ""),
            skills=skills,
            model=config.get("ai_model"),
        )
        st.session_state.covexcode_tokens_used = int(
            st.session_state.get("covexcode_tokens_used", 0)
        ) + int(q_usage.get("total_tokens", 0))

    used_ids = set(st.session_state.get("interview_used_ids", set()))
    selected = select_interview_questions(
        question_bank=bank,
        role=config["role"],
        domain=config["domain"],
        interview_type=config["interview_type"],
        difficulty=config["difficulty"],
        num_questions=int(config["num_questions"]),
        exclude_ids=used_ids,
        extra_ai_questions=extra_ai,
    )

    # If the user exhausted the entire bank across many sessions, allow fresh rotation
    if len(selected) < int(config["num_questions"]):
        st.session_state.interview_used_ids = set()
        selected = select_interview_questions(
            question_bank=bank,
            role=config["role"],
            domain=config["domain"],
            interview_type=config["interview_type"],
            difficulty=config["difficulty"],
            num_questions=int(config["num_questions"]),
            exclude_ids=set(),
            extra_ai_questions=extra_ai,
        )

    for q in selected:
        st.session_state.interview_used_ids.add(q["question_id"])

    st.session_state.interview_config = config
    st.session_state.interview_questions = selected
    st.session_state.interview_current_idx = 0
    st.session_state.interview_responses = {}
    st.session_state.interview_question_start_ts = time.time()
    st.session_state.interview_ai_note = ai_note
    st.session_state.interview_stage = "active"


def render_mock_interview_page(navigate_to: Callable[[Any], None]) -> None:
    """Main router for the AI Mock Interview page."""
    init_interview_state()

    nav_c1, _ = st.columns([1.2, 3.8], gap="medium")
    with nav_c1:
        if st.button("← Back to Explore", key="btn_interview_back_explore", use_container_width=True):
            navigate_to("explore")

    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="page-header-card">
        <div class="page-title">🎤 AI Mock Interview</div>
        <div class="page-subtitle" style="margin-bottom: 0;">
            Practice role-specific Technical, HR, and Behavioral interviews with ML-powered semantic evaluation and actionable feedback.
        </div>
    </div>
    """, unsafe_allow_html=True)

    stage = st.session_state.interview_stage
    if stage == "setup":
        _render_setup_view()
    elif stage == "active":
        _render_active_interview_view()
    elif stage == "summary":
        _render_summary_view(navigate_to)
    else:
        reset_interview_session()
        _render_setup_view()


def _render_setup_view() -> None:
    """Render Section 1: Interview Setup with automatic role suggestions."""
    bank = get_cached_question_bank()
    cfg = dict(st.session_state.interview_config)

    # Automatic role suggestions from confirmed skills & internship recommendations
    confirmed_skills = list(st.session_state.get("confirmed_skills", []))
    if not confirmed_skills and st.session_state.get("peer_profile"):
        confirmed_skills = list(st.session_state.peer_profile.get("technical_skills", []))
    recommendations = st.session_state.get("recommendations") or []

    suggested_roles = suggest_roles_for_student(
        confirmed_skills=confirmed_skills,
        recommendations=recommendations,
        top_n=3,
    )

    if cfg.get("internship_context"):
        ctx_c1, ctx_c2 = st.columns([3.5, 1])
        with ctx_c1:
            st.info(
                f"🎯 **Launched from Internship:** *{cfg['internship_context']}* — "
                f"Pre-configured for **{cfg['role']}** ({cfg['domain']})."
            )
        with ctx_c2:
            if st.button("Clear Link", key="btn_clear_internship_ctx", use_container_width=True):
                cfg["internship_context"] = ""
                st.session_state.interview_config = cfg
                st.rerun()

    st.markdown('<div class="content-card-title">✨ Suggested Interview Roles for Your Profile</div>', unsafe_allow_html=True)
    if confirmed_skills or recommendations:
        st.caption("Automatically suggested from your confirmed skills and internship recommendations:")
    else:
        st.caption("Popular internship interview tracks (upload a resume or confirm skills for personalized suggestions):")

    sug_cols = st.columns(len(suggested_roles), gap="medium")
    for idx, item in enumerate(suggested_roles):
        with sug_cols[idx]:
            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-left: 4px solid #0d9488; border-radius: 10px; padding: 12px 14px; margin-bottom: 8px;">
                <div style="font-weight: 700; color: #0f172a; font-size: 0.98rem;">{item['role']}</div>
                <div style="font-size: 0.8rem; color: #0f766e; margin-bottom: 4px;">{item['domain']}</div>
                <div style="font-size: 0.8rem; color: #475569;">{item['reason']}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Select {item['role']}", key=f"btn_sug_role_{idx}", use_container_width=True):
                cfg["role"] = item["role"]
                cfg["domain"] = item["domain"]
                st.session_state.interview_config = cfg
                st.rerun()

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="content-card-title">⚙️ Interview Setup</div>', unsafe_allow_html=True)

    roles_list = list(ROLE_TO_DOMAIN.keys())
    domains_list = sorted(list(set(ROLE_TO_DOMAIN.values())))

    current_role = cfg.get("role", "Software Engineer")
    role_idx = roles_list.index(current_role) if current_role in roles_list else 0

    # Role & domain selection outside form so domain auto-syncs cleanly when role changes
    s_c1, s_c2 = st.columns(2, gap="large")
    with s_c1:
        selected_role = st.selectbox(
            "Target Internship Role",
            options=roles_list,
            index=role_idx,
            key="setup_role_select",
        )
    default_domain = ROLE_TO_DOMAIN.get(selected_role, "Software Development")
    if cfg.get("role") != selected_role:
        cfg["role"] = selected_role
        cfg["domain"] = default_domain
        st.session_state.interview_config = cfg
    current_domain = cfg.get("domain", default_domain)
    domain_idx = domains_list.index(current_domain) if current_domain in domains_list else 0

    with s_c2:
        selected_domain = st.selectbox(
            "Technical Domain",
            options=domains_list,
            index=domain_idx,
            key="setup_domain_select",
        )

    with st.form("mock_interview_setup_form", border=True):
        f_c1, f_c2, f_c3, f_c4 = st.columns(4, gap="medium")
        with f_c1:
            current_type = cfg.get("interview_type", "Mixed")
            type_idx = INTERVIEW_TYPES.index(current_type) if current_type in INTERVIEW_TYPES else 3
            interview_type = st.selectbox("Interview Type", options=INTERVIEW_TYPES, index=type_idx)
        with f_c2:
            current_diff = cfg.get("difficulty", "Intermediate")
            diff_idx = DIFFICULTY_LEVELS.index(current_diff) if current_diff in DIFFICULTY_LEVELS else 1
            difficulty = st.selectbox("Difficulty Level", options=DIFFICULTY_LEVELS, index=diff_idx)
        with f_c3:
            current_count = int(cfg.get("num_questions", 5))
            cnt_idx = QUESTION_COUNTS.index(current_count) if current_count in QUESTION_COUNTS else 0
            num_questions = st.selectbox("Number of Questions", options=QUESTION_COUNTS, index=cnt_idx)
        with f_c4:
            timed_label = st.selectbox(
                "Timing Mode",
                options=["Untimed Practice", "Timed Mode"],
                index=1 if cfg.get("timed_mode") else 0,
            )
            timed_mode = timed_label == "Timed Mode"

        per_q_sec = SECONDS_PER_QUESTION_BY_DIFFICULTY.get(difficulty, 120)
        provider_status = get_ai_provider().get_status()
        ai_c1, ai_c2 = st.columns([1.8, 1.0], gap="medium")
        with ai_c1:
            use_ai = st.checkbox(
                "✨ Enable Covexcode Generative AI (personalized questions, follow-ups & detailed feedback)",
                value=bool(cfg.get("use_ai", provider_status["configured"])),
                disabled=not provider_status["configured"],
            )
        with ai_c2:
            model_opts = list(SUPPORTED_COVEXCODE_MODELS)
            cur_model = cfg.get("ai_model", provider_status["default_model"])
            m_idx = model_opts.index(cur_model) if cur_model in model_opts else 0
            ai_model = st.selectbox(
                "Covexcode Model",
                options=model_opts,
                index=m_idx,
                disabled=not provider_status["configured"],
            )

        ai_status_line = (
            f"Covexcode AI ready (`{ai_model}` • Key: `{provider_status['key_masked']}` • Session tokens used: {st.session_state.get('covexcode_tokens_used', 0):,}). "
            if provider_status["configured"]
            else "Covexcode key not configured — running in full offline mode with 170 curated questions & local MiniLM/TF-IDF evaluator. "
        )
        st.caption(
            f"Question bank ready ({len(bank)} curated questions). "
            + (
                f"Timed mode allows {per_q_sec} seconds per {difficulty.lower()} question. "
                if timed_mode
                else "Untimed mode lets you reflect thoroughly at your own pace. "
            )
            + ai_status_line
        )

        _, start_col, _ = st.columns([1.2, 1.4, 1.2])
        with start_col:
            start_clicked = st.form_submit_button(
                "🚀 Start Mock Interview",
                type="primary",
                use_container_width=True,
            )

    if start_clicked:
        new_config = {
            "role": selected_role,
            "domain": selected_domain,
            "interview_type": interview_type,
            "difficulty": difficulty,
            "num_questions": int(num_questions),
            "timed_mode": timed_mode,
            "internship_context": cfg.get("internship_context", ""),
            "use_ai": bool(use_ai and provider_status["configured"]),
            "ai_model": ai_model,
        }
        _start_new_interview(new_config)
        st.rerun()


def _render_active_interview_view() -> None:
    """Render Section 4 & 5: Active question interface and immediate per-answer feedback."""
    questions = st.session_state.interview_questions
    if not questions:
        reset_interview_session()
        st.rerun()
        return

    cfg = st.session_state.interview_config
    idx = int(st.session_state.interview_current_idx)
    total = len(questions)
    if idx >= total:
        st.session_state.interview_stage = "summary"
        st.rerun()
        return

    q = questions[idx]
    responses = st.session_state.interview_responses
    already_evaluated = idx in responses

    if st.session_state.get("interview_ai_note"):
        st.info(f"ℹ️ {st.session_state.interview_ai_note}")

    # Top progress bar & controls
    attempted_cnt = sum(1 for r in responses.values() if not r["evaluation"].get("skipped"))
    skipped_cnt = sum(1 for r in responses.values() if r["evaluation"].get("skipped"))

    top_c1, top_c2, top_c3 = st.columns([2.2, 1.1, 0.9], gap="medium")
    with top_c1:
        st.markdown(
            f"**Question {idx + 1} of {total}** &nbsp;•&nbsp; `{cfg['role']}` &nbsp;•&nbsp; "
            f"`{q['question_type']}` &nbsp;•&nbsp; `{q['difficulty']}`"
        )
        st.progress((idx + (1 if already_evaluated else 0)) / float(total))
        st.caption(f"Attempted: {attempted_cnt} | Skipped: {skipped_cnt} | Remaining: {total - len(responses)}")

    with top_c2:
        if cfg.get("timed_mode"):
            limit_sec = SECONDS_PER_QUESTION_BY_DIFFICULTY.get(cfg.get("difficulty", "Intermediate"), 120)
            elapsed = max(0, int(time.time() - float(st.session_state.interview_question_start_ts)))
            remaining = max(0, limit_sec - elapsed)
            mins, secs = divmod(remaining, 60)
            timer_bg = "#fef2f2" if remaining <= 15 else "#eff6ff"
            timer_col = "#b91c1c" if remaining <= 15 else "#1d4ed8"
            timer_txt = f"{mins:02d}:{secs:02d} left" if remaining > 0 else "Time Elapsed"
            st.markdown(f"""
            <div style="background: {timer_bg}; color: {timer_col}; border: 1px solid #cbd5e1; border-radius: 8px; padding: 8px 12px; text-align: center; font-weight: 700; font-size: 0.95rem;">
                ⏱️ {timer_txt}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; border-radius: 8px; padding: 8px 12px; text-align: center; font-weight: 600; font-size: 0.9rem;">
                🕊️ Untimed Practice
            </div>
            """, unsafe_allow_html=True)

    with top_c3:
        if st.button("🛑 End Interview", key=f"btn_end_interview_{idx}", use_container_width=True):
            st.session_state.interview_stage = "summary"
            st.rerun()

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Current Question Card
    st.markdown(f"""
    <div class="rec-card">
        <div class="rec-title-row">
            <div style="font-size: 0.85rem; font-weight: 700; color: #1d4ed8; text-transform: uppercase; letter-spacing: 0.4px;">
                📌 Topic: {q.get('topic', 'General')}
            </div>
            <div class="rec-badge">{q['question_id']} • {q['question_type']}</div>
        </div>
        <div style="font-size: 1.15rem; font-weight: 700; color: #0f172a; line-height: 1.5; margin-top: 6px;">
            {q['question_text']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Answer Input Area
    existing_ans = responses[idx]["answer_text"] if already_evaluated else ""
    answer_text = st.text_area(
        label="Your Interview Answer",
        value=existing_ans,
        height=180,
        placeholder=(
            "Write your answer clearly. For technical questions, define the concept, explain how/why it works, and mention a concrete example. "
            "For behavioral questions, use Situation, Task, Action, and Result (STAR)."
        ),
        disabled=already_evaluated,
        key=f"interview_answer_box_{idx}",
    )

    # Action Buttons
    btn_c1, btn_c2, btn_c3 = st.columns([1.2, 1.0, 1.2], gap="medium")
    with btn_c1:
        submit_clicked = st.button(
            "✅ Submit Answer",
            type="primary",
            disabled=already_evaluated,
            use_container_width=True,
            key=f"btn_submit_q_{idx}",
        )
    with btn_c2:
        skip_clicked = st.button(
            "⏭️ Skip Question",
            disabled=already_evaluated,
            use_container_width=True,
            key=f"btn_skip_q_{idx}",
        )
    with btn_c3:
        next_label = "📊 Finish & View Dashboard ➔" if idx == total - 1 else "Next Question ➔"
        next_clicked = st.button(
            next_label,
            disabled=not already_evaluated,
            use_container_width=True,
            key=f"btn_next_q_{idx}",
        )

    if submit_clicked and not already_evaluated:
        evaluator = get_cached_interview_evaluator()
        use_ai = bool(cfg.get("use_ai", False))
        ai_model = cfg.get("ai_model")
        with st.spinner("Evaluating your response..."):
            evaluation = evaluator.evaluate_answer(
                q,
                answer_text,
                skipped=False,
                use_ai=use_ai,
                ai_model=ai_model,
            )
        st.session_state.covexcode_tokens_used = int(
            st.session_state.get("covexcode_tokens_used", 0)
        ) + int((evaluation.get("ai_usage") or {}).get("total_tokens", 0))

        followup_data = None
        if use_ai and answer_text.strip():
            ai = get_ai_provider()
            if ai.is_configured():
                fu_res = ai.generate_followup_question(
                    question=q,
                    student_answer=answer_text.strip(),
                    covered_concepts=evaluation.get("covered_concepts", []),
                    missing_concepts=evaluation.get("missing_concepts", []),
                    model=ai_model,
                )
                st.session_state.covexcode_tokens_used += int(
                    (fu_res.get("usage") or {}).get("total_tokens", 0)
                )
                if fu_res.get("ok") and fu_res.get("followup_question"):
                    followup_data = {
                        "question": fu_res["followup_question"],
                        "focus_area": fu_res.get("focus_area", ""),
                        "reference_hint": fu_res.get("reference_hint", ""),
                        "answer": "",
                        "feedback": None,
                    }

        elapsed_sec = round(time.time() - float(st.session_state.interview_question_start_ts), 1)
        st.session_state.interview_responses[idx] = {
            "answer_text": answer_text.strip(),
            "evaluation": evaluation,
            "time_spent_sec": elapsed_sec,
            "followup": followup_data,
        }
        st.rerun()

    if skip_clicked and not already_evaluated:
        evaluator = get_cached_interview_evaluator()
        evaluation = evaluator.evaluate_answer(q, "", skipped=True)
        elapsed_sec = round(time.time() - float(st.session_state.interview_question_start_ts), 1)
        st.session_state.interview_responses[idx] = {
            "answer_text": "",
            "evaluation": evaluation,
            "time_spent_sec": elapsed_sec,
            "followup": None,
        }
        st.rerun()

    if next_clicked and already_evaluated:
        if idx < total - 1:
            st.session_state.interview_current_idx = idx + 1
            st.session_state.interview_question_start_ts = time.time()
        else:
            st.session_state.interview_stage = "summary"
        st.rerun()

    # Section 5: Per-Answer Feedback Display
    if already_evaluated:
        _render_question_feedback(idx, responses[idx]["evaluation"], responses[idx].get("followup"), cfg)


def _render_question_feedback(
    q_idx: int,
    eval_res: Dict[str, Any],
    followup: Optional[Dict[str, Any]] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """Display transparent per-answer feedback immediately after submission or skip."""
    st.markdown("<hr style='margin: 18px 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)
    st.markdown('<div class="content-card-title">📝 Answer Evaluation & Feedback</div>', unsafe_allow_html=True)

    if eval_res.get("fallback_reason"):
        st.caption(f"ℹ️ Semantic Fallback Active: {eval_res['fallback_reason']}")
    if eval_res.get("ai_fallback_note"):
        st.caption(f"ℹ️ Covexcode AI Fallback: {eval_res['ai_fallback_note']}")

    if eval_res.get("skipped"):
        st.warning("⏭️ You skipped this question. Review the expected concepts and reference answer below.")
    else:
        score = float(eval_res["overall_score"])
        badge_bg = "#ecfdf5" if score >= 75 else ("#eff6ff" if score >= 50 else "#fffbeb")
        badge_col = "#047857" if score >= 75 else ("#1d4ed8" if score >= 50 else "#b45309")

        f_c1, f_c2 = st.columns([1.0, 2.2], gap="large")
        with f_c1:
            st.markdown(f"""
            <div style="background: {badge_bg}; border: 1px solid #cbd5e1; border-radius: 12px; padding: 18px; text-align: center;">
                <div style="font-size: 0.85rem; font-weight: 700; color: {badge_col}; text-transform: uppercase;">Practice Score</div>
                <div style="font-size: 2.3rem; font-weight: 800; color: #0f172a; line-height: 1.15;">{score:.1f}<span style="font-size: 1.05rem; color: #64748b;"> / 100</span></div>
                <div style="font-size: 0.78rem; color: #475569; margin-top: 6px;">{eval_res['backend_used']}</div>
            </div>
            """, unsafe_allow_html=True)

        with f_c2:
            dims = eval_res["dimensions"]
            d_cols = st.columns(5)
            d_cols[0].metric("Semantic (35%)", f"{dims['semantic_relevance']:.0f}")
            d_cols[1].metric("Concepts (30%)", f"{dims['concept_coverage']:.0f}")
            d_cols[2].metric("Correctness (15%)", f"{dims['technical_correctness']:.0f}")
            d_cols[3].metric("Complete (10%)", f"{dims['completeness']:.0f}")
            d_cols[4].metric("Clarity (10%)", f"{dims['answer_clarity']:.0f}")
            st.caption(eval_res["scoring_explanation"])

    # Covered vs Missing Concepts
    cov_html = "".join(
        [f'<span class="tag-matched">✓ {c}</span>' for c in eval_res.get("covered_concepts", [])]
    ) or '<span style="color: #94a3b8; font-style: italic;">None detected directly in submitted text</span>'

    miss_html = "".join(
        [f'<span class="tag-missing">+ {c}</span>' for c in eval_res.get("missing_concepts", [])]
    ) or '<span style="color: #047857; font-weight: 600;">All expected concepts covered!</span>'

    st.markdown(f"""
    <div class="overlap-box" style="margin: 12px 0;">
        <div style="margin-bottom: 6px;"><strong>Concepts Correctly Covered:</strong> {cov_html}</div>
        <div><strong>Missing Important Concepts:</strong> {miss_html}</div>
    </div>
    """, unsafe_allow_html=True)

    # Strengths (when Covexcode AI evaluation returned grounded highlights)
    if eval_res.get("ai_strengths"):
        st.markdown("**Observed Strengths:**")
        for s_item in eval_res["ai_strengths"]:
            st.markdown(f"- {s_item}")

    # Areas for improvement
    st.markdown("**Areas for Improvement:**")
    for area in eval_res.get("areas_for_improvement", []):
        st.markdown(f"- {area}")

    # Contextual Follow-up Question (Covexcode AI)
    if followup and followup.get("question"):
        st.markdown(f"""
        <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-left: 4px solid #2563eb; border-radius: 10px; padding: 14px 16px; margin: 12px 0;">
            <div style="font-size: 0.82rem; font-weight: 700; color: #1d4ed8; text-transform: uppercase;">
                💬 Contextual Follow-up Question {f"({followup['focus_area']})" if followup.get('focus_area') else ""}
            </div>
            <div style="font-size: 1.02rem; font-weight: 600; color: #0f172a; margin-top: 4px;">
                {followup['question']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        fu_done = followup.get("feedback") is not None
        fu_ans = st.text_area(
            "Optional Follow-up Response",
            value=followup.get("answer", ""),
            height=90,
            disabled=fu_done,
            placeholder="Answer the interviewer's follow-up question here (optional)...",
            key=f"fu_answer_{q_idx}",
        )
        if not fu_done:
            if st.button("Submit Follow-up", key=f"btn_submit_fu_{q_idx}"):
                if fu_ans.strip():
                    ai = get_ai_provider()
                    fu_eval = ai.career_assistant_reply(
                        user_question=(
                            f"Follow-up question: {followup['question']}\n"
                            f"Expected focus: {followup.get('reference_hint', '')}\n"
                            f"Student's follow-up answer: {fu_ans.strip()}\n"
                            "Give 2-3 concise, honest sentences of feedback grounded strictly in what the student wrote."
                        ),
                        model=(cfg or {}).get("ai_model"),
                    )
                    st.session_state.covexcode_tokens_used += int(
                        (fu_eval.get("usage") or {}).get("total_tokens", 0)
                    )
                    fb_text = (
                        fu_eval["content"]
                        if fu_eval.get("ok")
                        else f"Follow-up recorded (Covexcode feedback unavailable: {fu_eval.get('error')}). Reference focus: {followup.get('reference_hint', '')}"
                    )
                    st.session_state.interview_responses[q_idx]["followup"]["answer"] = fu_ans.strip()
                    st.session_state.interview_responses[q_idx]["followup"]["feedback"] = fb_text
                    st.rerun()
                else:
                    st.warning("Type a short follow-up answer before submitting.")
        else:
            st.success(f"**Follow-up Feedback:** {followup['feedback']}")

    # Reference Answer & Suggested Improved Answer
    ref_c1, ref_c2 = st.columns(2, gap="large")
    with ref_c1:
        with st.expander("📘 Reference Answer", expanded=True):
            st.markdown(eval_res.get("reference_answer", ""))
    with ref_c2:
        with st.expander("✨ Suggested Improved Answer", expanded=True):
            st.markdown(eval_res.get("suggested_improved_answer", ""))

    # Transparent Rubric Breakdown
    with st.expander("🔍 Scoring Method & Rubric Contribution Breakdown"):
        rows = []
        for dim_key, weight in RUBRIC_WEIGHTS.items():
            raw_dim = eval_res["dimensions"].get(dim_key, 0.0)
            contrib = eval_res["score_contributions"].get(dim_key, 0.0)
            rows.append({
                "Criterion": dim_key.replace("_", " ").title(),
                "Weight": f"{int(weight * 100)}%",
                "Dimension Score (/100)": raw_dim,
                "Weighted Points": contrib,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if eval_res.get("rubric"):
            st.markdown("**Question-Specific Rubric:**")
            for k, v in eval_res["rubric"].items():
                st.markdown(f"- **{k.replace('_', ' ').title()}:** {v}")


def _render_summary_view(navigate_to: Callable[[Any], None]) -> None:
    """Render Section 6: Final Interview Dashboard with charts and downloadable report."""
    questions = st.session_state.interview_questions
    responses = st.session_state.interview_responses
    cfg = st.session_state.interview_config

    summary = build_interview_summary(cfg, questions, responses)

    st.markdown('<div class="content-card-title">📊 Final Interview Performance Dashboard</div>', unsafe_allow_html=True)
    st.info(f"ℹ️ **Honest Practice Notice:** {summary['disclaimer']}")

    # Top KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Overall Practice Score", f"{summary['overall_practice_score']:.1f} / 100")
    k2.metric(
        "Questions Attempted",
        f"{summary['questions_attempted']} / {summary['total_questions']}",
        f"{summary['questions_skipped']} skipped" if summary["questions_skipped"] else "0 skipped",
    )
    k3.metric("Average Answer Score", f"{summary['average_answer_score']:.1f} / 100")
    backend_names = ", ".join(summary["backends_used"].keys()) if summary["backends_used"] else "No attempts"
    k4_label = "MiniLM + Covexcode" if "Covexcode" in backend_names else ("MiniLM" if "MiniLM" in backend_names else ("TF-IDF Fallback" if "Fallback" in backend_names else "N/A"))
    tokens_used = int(st.session_state.get("covexcode_tokens_used", 0))
    k4.metric("Evaluator Mode", k4_label, f"{tokens_used:,} Covexcode tokens" if tokens_used > 0 else "Local mode")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Strongest & Weakest Topics
    t_c1, t_c2 = st.columns(2, gap="large")
    with t_c1:
        st.markdown("""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-left: 4px solid #10b981; border-radius: 10px; padding: 14px 16px;">
            <div style="font-weight: 700; color: #047857; margin-bottom: 6px;">💪 Strongest Topics</div>
        """, unsafe_allow_html=True)
        if summary["strongest_topics"]:
            for t in summary["strongest_topics"]:
                st.markdown(f"- **{t['topic']}** — `{t['avg_score']}/100` ({t['count']} question(s))")
        else:
            st.caption("Attempt questions to identify strongest topics.")
        st.markdown("</div>", unsafe_allow_html=True)

    with t_c2:
        st.markdown("""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-left: 4px solid #f59e0b; border-radius: 10px; padding: 14px 16px;">
            <div style="font-weight: 700; color: #b45309; margin-bottom: 6px;">📚 Weakest / Focus Topics</div>
        """, unsafe_allow_html=True)
        if summary["weakest_topics"]:
            for t in summary["weakest_topics"]:
                st.markdown(f"- **{t['topic']}** — `{t['avg_score']}/100` ({t['count']} question(s))")
        else:
            st.caption("Attempt questions to identify focus areas.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Charts
    ch_c1, ch_c2 = st.columns(2, gap="large")
    with ch_c1:
        st.markdown('<div class="chart-title">Question-Wise Practice Scores</div>', unsafe_allow_html=True)
        fig_q = charts.plot_interview_question_scores_chart(summary["question_rows"])
        st.pyplot(fig_q, use_container_width=True)
    with ch_c2:
        st.markdown('<div class="chart-title">Rubric Dimension Averages</div>', unsafe_allow_html=True)
        fig_d = charts.plot_interview_rubric_dimensions_chart(summary["dimension_averages"])
        st.pyplot(fig_d, use_container_width=True)

    if len(summary["topic_averages"]) >= 2:
        st.markdown('<div class="chart-title">Topic-Level Performance Breakdown</div>', unsafe_allow_html=True)
        fig_t = charts.plot_interview_topic_performance_chart(summary["topic_averages"])
        st.pyplot(fig_t, use_container_width=True)

    # Personalized Preparation Suggestions
    st.markdown('<div class="content-card-title">🚀 Personalized Preparation Suggestions</div>', unsafe_allow_html=True)
    for suggestion in summary["preparation_suggestions"]:
        st.markdown(f"- {suggestion}")

    if summary["most_missed_concepts"]:
        missed_chips = "".join(
            [f'<span class="tag-missing">{concept} ({cnt}x)</span>' for concept, cnt in summary["most_missed_concepts"]]
        )
        st.markdown(f"<div style='margin: 8px 0 14px 0;'><strong>Most Frequently Missed Concepts:</strong> {missed_chips}</div>", unsafe_allow_html=True)

    # Question-wise performance table
    st.markdown('<div class="content-card-title">📋 Question-Wise Performance Breakdown</div>', unsafe_allow_html=True)
    if summary["question_rows"]:
        table_rows = [
            {
                "Q#": f"Q{r['index']}",
                "Topic": r["topic"],
                "Type": r["question_type"],
                "Difficulty": r["difficulty"],
                "Status": r["status"],
                "Score (/100)": f"{r['score']:.1f}" if r["score"] is not None else "Skipped",
                "Covered Concepts": ", ".join(r["covered_concepts"]) or "—",
                "Missing Concepts": ", ".join(r["missing_concepts"]) or "—",
            }
            for r in summary["question_rows"]
        ]
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        for r in summary["question_rows"]:
            score_label = f"{r['score']}/100" if r["score"] is not None else "Skipped"
            with st.expander(f"Q{r['index']}: {r['topic']} ({score_label})"):
                st.markdown(f"**Question:** {r['question_text']}")
                st.markdown(f"**Your Answer:** {r['student_answer'] or '_Skipped_'}")
                st.markdown(f"**Evaluation Backend:** `{r['backend_used']}`")

    # Downloadable Report & Navigation Controls
    report_md = format_interview_report_markdown(summary)
    st.markdown("<hr style='margin: 20px 0 14px 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)
    act_c1, act_c2, act_c3 = st.columns([1.2, 1.2, 1.1], gap="medium")
    with act_c1:
        st.download_button(
            label="📥 Download Interview Report (.md)",
            data=report_md,
            file_name=f"mock_interview_report_{cfg.get('role', 'role').lower().replace(' ', '_')}.md",
            mime="text/markdown",
            type="primary",
            use_container_width=True,
        )
    with act_c2:
        if st.button("🔄 Start New Mock Interview", key="btn_restart_mock_interview", use_container_width=True):
            reset_interview_session(clear_used_history=False)
            st.rerun()
    with act_c3:
        if st.button("🔍 Back to Explore Internships", key="btn_summary_to_explore", use_container_width=True):
            navigate_to("explore")
