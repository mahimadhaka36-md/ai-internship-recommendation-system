"""
peer_match_ui.py
----------------
Streamlit UI for AI Peer Match — Powered by K-Nearest Neighbors (KNN).

Includes all 8 structured sections matching the existing Academic Dashboard theme:
1. My Academic Profile
2. Find Similar Students
3. Recommended Peer Profiles
4. Student Profile Comparison
5. Career Recommendations
6. Recommended Internships
7. Personalized Skill Improvement
8. KNN Model Insights
"""

from typing import Any, Callable, Dict, List
import pandas as pd
import streamlit as st

import charts
from peer_match import canonicalize_peer_skill
from skill_extractor import get_skill_vocab


SAMPLE_PRESETS: Dict[str, Dict[str, Any]] = {
    "Custom / Current Input": {},
    "Sample: Aspiring Data Scientist": {
        "major": "Computer Science",
        "university_year": 3,
        "cgpa": 8.4,
        "cgpa_scale": 10.0,
        "technical_skills": ["python", "sql", "pandas", "numpy"],
        "programming_skill": 7,
        "problem_solving": 7,
        "communication_skills": 7,
        "projects_completed": 3,
        "certifications": 1,
        "internships": 0,
    },
    "Sample: Full Stack Web Builder": {
        "major": "Computer Science",
        "university_year": 2,
        "cgpa": 8.1,
        "cgpa_scale": 10.0,
        "technical_skills": ["javascript", "react", "html", "css", "sql"],
        "programming_skill": 8,
        "problem_solving": 7,
        "communication_skills": 8,
        "projects_completed": 4,
        "certifications": 1,
        "internships": 0,
    },
    "Sample: Cybersecurity & Systems": {
        "major": "Computer Science",
        "university_year": 3,
        "cgpa": 8.6,
        "cgpa_scale": 10.0,
        "technical_skills": ["linux", "networking", "python", "cybersecurity"],
        "programming_skill": 7,
        "problem_solving": 8,
        "communication_skills": 6,
        "projects_completed": 3,
        "certifications": 2,
        "internships": 0,
    },
}


def init_peer_match_state() -> None:
    """Initialize session state keys for the AI Peer Match page."""
    if "peer_profile" not in st.session_state:
        default_skills = list(st.session_state.get("confirmed_skills", []))
        if not default_skills:
            default_skills = ["python", "sql", "pandas", "machine learning"]
        st.session_state.peer_profile = {
            "student_name": st.session_state.get("user_name", "Student User"),
            "major": "Computer Science",
            "university_year": 3,
            "cgpa": 8.3,
            "cgpa_scale": 10.0,
            "technical_skills": default_skills,
            "programming_skill": 7,
            "problem_solving": 7,
            "communication_skills": 7,
            "projects_completed": 3,
            "certifications": 1,
            "internships": 0,
        }
    if "peer_k" not in st.session_state:
        st.session_state.peer_k = 5
    if "peer_metric" not in st.session_state:
        st.session_state.peer_metric = "euclidean"
    if "peer_weight_preset" not in st.session_state:
        st.session_state.peer_weight_preset = "Balanced (Skills + Academics)"
    if "peer_match_results" not in st.session_state:
        st.session_state.peer_match_results = None


def _build_skill_options(peer_engine, current_skills: List[str]) -> List[str]:
    """Combine dataset skills, global skill vocab, and current selections cleanly."""
    combined = set(peer_engine.get_available_skills())
    combined.update(get_skill_vocab())
    combined.update(canonicalize_peer_skill(s) for s in current_skills if s.strip())
    return sorted(combined)


def render_peer_match_page(engine, peer_engine, navigate_to: Callable[[Any], None]) -> None:
    """Render the complete AI Peer Match page with all 8 sections."""
    init_peer_match_state()

    # Top-left quick return navigation
    nav_c1, _ = st.columns([1.2, 3.8], gap="medium")
    with nav_c1:
        if st.button("← Back to Explore", key="btn_peer_back_explore", use_container_width=True):
            navigate_to("explore")

    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

    # Page Header
    st.markdown("""
    <div class="page-header-card">
        <div class="page-title">👥 Explore Profiles</div>
        <div class="page-subtitle" style="margin-bottom: 0;">Discover Similar Student Profiles, Explore Career Paths, and Improve Your Skills with AI.</div>
    </div>
    """, unsafe_allow_html=True)

    # =========================================================================
    # SECTION 1: My Academic Profile
    # =========================================================================
    st.markdown('<div class="content-card-title">🎓 My Academic Profile</div>', unsafe_allow_html=True)

    # Quick-fill toolbar outside form so buttons trigger cleanly
    qf_col1, qf_col2, qf_col3 = st.columns([1.5, 1.3, 1.2], gap="medium")
    with qf_col1:
        selected_preset = st.selectbox(
            "Load Sample Student Preset (Optional)",
            options=list(SAMPLE_PRESETS.keys()),
            index=0,
            key="peer_preset_select",
        )
    with qf_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("✨ Apply Sample Preset", key="btn_apply_peer_preset", use_container_width=True):
            preset = SAMPLE_PRESETS.get(selected_preset, {})
            if preset:
                updated = dict(st.session_state.peer_profile)
                updated.update(preset)
                st.session_state.peer_profile = updated
                st.session_state.peer_match_results = None
                st.rerun()
    with qf_col3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        has_resume_skills = bool(st.session_state.get("confirmed_skills"))
        if st.button(
            "📥 Import Resume Skills",
            key="btn_import_resume_skills",
            use_container_width=True,
            disabled=not has_resume_skills,
            help="Copy skills extracted & confirmed from Steps 1–2 into your profile",
        ):
            updated = dict(st.session_state.peer_profile)
            updated["technical_skills"] = [
                canonicalize_peer_skill(s) for s in st.session_state.confirmed_skills
            ]
            st.session_state.peer_profile = updated
            st.session_state.peer_match_results = None
            st.rerun()

    profile = st.session_state.peer_profile
    major_options = [
        "Computer Science",
        "Information Technology",
        "Data Science & AI",
        "Software Engineering",
        "Electronics & Communication",
        "Other",
    ]
    current_major = profile.get("major", "Computer Science")
    major_idx = major_options.index(current_major) if current_major in major_options else 0

    with st.form("peer_profile_form", border=True):
        r1_c1, r1_c2, r1_c3, r1_c4 = st.columns([1.4, 1.3, 0.9, 1.1], gap="medium")
        with r1_c1:
            student_name = st.text_input(
                "Student Name",
                value=str(profile.get("student_name", "Student User")),
                placeholder="Enter your name",
            )
        with r1_c2:
            major_choice = st.selectbox("Academic Major", options=major_options, index=major_idx)
            custom_major = st.text_input(
                "Specify Major (if Other)",
                value="" if current_major in major_options else current_major,
                placeholder="e.g. Computer Engineering",
            )
        with r1_c3:
            university_year = st.selectbox(
                "University Year",
                options=[1, 2, 3, 4],
                index=max(0, min(3, int(profile.get("university_year", 3)) - 1)),
                format_func=lambda y: f"Year {y}",
            )
        with r1_c4:
            cgpa_scale = st.selectbox(
                "CGPA Scale",
                options=[10.0, 4.0],
                index=0 if float(profile.get("cgpa_scale", 10.0)) == 10.0 else 1,
                format_func=lambda s: f"{int(s)}.0 Scale",
            )
            max_cgpa = float(cgpa_scale)
            default_cgpa = min(float(profile.get("cgpa", 8.0)), max_cgpa)
            cgpa = st.number_input(
                "CGPA",
                min_value=0.0,
                max_value=max_cgpa,
                value=default_cgpa,
                step=0.05,
                format="%.2f",
            )

        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)

        # Skills input
        sk_c1, sk_c2 = st.columns([2.2, 1.1], gap="medium")
        normalized_current = [canonicalize_peer_skill(s) for s in profile.get("technical_skills", [])]
        skill_options = _build_skill_options(peer_engine, normalized_current)
        with sk_c1:
            selected_skills = st.multiselect(
                "Technical Skills",
                options=skill_options,
                default=[s for s in normalized_current if s in skill_options],
                placeholder="Select your technical skills (e.g., python, sql, react, pytorch)...",
            )
        with sk_c2:
            extra_skills_text = st.text_input(
                "Add Additional Skills (comma-separated)",
                value="",
                placeholder="e.g. docker, fastapi, tableau",
            )

        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)

        # Proficiency sliders & experience counts
        p_c1, p_c2, p_c3 = st.columns(3, gap="medium")
        with p_c1:
            programming_skill = st.slider(
                "Programming Skill (1–10)",
                min_value=1,
                max_value=10,
                value=int(profile.get("programming_skill", 7)),
            )
        with p_c2:
            problem_solving = st.slider(
                "Problem Solving (1–10)",
                min_value=1,
                max_value=10,
                value=int(profile.get("problem_solving", 7)),
            )
        with p_c3:
            communication_skills = st.slider(
                "Communication Skills (1–10)",
                min_value=1,
                max_value=10,
                value=int(profile.get("communication_skills", 7)),
            )

        e_c1, e_c2, e_c3 = st.columns(3, gap="medium")
        with e_c1:
            projects_completed = st.number_input(
                "Projects Completed",
                min_value=0,
                max_value=30,
                value=int(profile.get("projects_completed", 3)),
                step=1,
            )
        with e_c2:
            certifications = st.number_input(
                "Certifications Earned",
                min_value=0,
                max_value=30,
                value=int(profile.get("certifications", 1)),
                step=1,
            )
        with e_c3:
            internships = st.number_input(
                "Previous Internships",
                min_value=0,
                max_value=15,
                value=int(profile.get("internships", 0)),
                step=1,
            )

        st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
        _, btn_col, _ = st.columns([1.2, 1.4, 1.2])
        with btn_col:
            submit_knn = st.form_submit_button(
                "🤝 Find Similar Students",
                type="primary",
                use_container_width=True,
            )

    k_val = 5
    metric_choice = "euclidean"
    weight_preset = "Balanced (Skills + Academics)"

    # Persist form changes & run KNN when submitted or on first visit
    if submit_knn or st.session_state.peer_match_results is None:
        final_major = custom_major.strip() if major_choice == "Other" and custom_major.strip() else major_choice
        combined_skills = list(selected_skills)
        if extra_skills_text.strip():
            for raw_s in extra_skills_text.split(","):
                norm_s = canonicalize_peer_skill(raw_s)
                if norm_s and norm_s not in combined_skills:
                    combined_skills.append(norm_s)

        updated_profile = {
            "student_name": student_name.strip() or "Student User",
            "major": final_major,
            "university_year": int(university_year),
            "cgpa": float(cgpa),
            "cgpa_scale": float(cgpa_scale),
            "technical_skills": combined_skills,
            "programming_skill": int(programming_skill),
            "problem_solving": int(problem_solving),
            "communication_skills": int(communication_skills),
            "projects_completed": int(projects_completed),
            "certifications": int(certifications),
            "internships": int(internships),
        }
        st.session_state.peer_profile = updated_profile
        st.session_state.peer_k = int(k_val)
        st.session_state.peer_metric = metric_choice
        st.session_state.peer_weight_preset = weight_preset

        dataset_freqs = engine.get_dataset_skill_frequencies()
        st.session_state.peer_match_results = peer_engine.find_similar_students(
            student_profile=updated_profile,
            k=int(k_val),
            metric=metric_choice,
            weight_preset=weight_preset,
            dataset_skill_freqs=dataset_freqs,
        )

    results = st.session_state.peer_match_results
    if not results:
        return

    user_norm = results["user_profile"]
    peers = results["peers"]
    peer_avg = results["peer_average"]
    career_recs = results["career_recommendations"]
    skill_gaps = results["skill_gaps"]
    improvements = results["profile_improvements"]

    if results["empty_skills_input"]:
        st.warning(
            "💡 No technical skills are currently selected — KNN matched strictly on academic and proficiency metrics. "
            "Add technical skills above for higher-precision peer and career alignment."
        )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # SECTION 2: Recommended Peer Profiles
    # =========================================================================
    st.markdown(f'<div class="content-card-title">👥 Recommended Peer Profiles (Top {results["k"]} Nearest Neighbors)</div>', unsafe_allow_html=True)
    st.caption(
        "All peer cards below come from `synthetic_student_profiles_30.csv` (fictional synthetic demo profiles for academic ML matching)."
    )

    for p in peers:
        matched_html = "".join(
            [f'<span class="tag-matched">✓ {s}</span>' for s in p["matched_skills"]]
        ) or '<span style="color: #94a3b8; font-style: italic;">No overlapping technical skills</span>'

        add_html = "".join(
            [f'<span class="tag-skill-chip">+ {s}</span>' for s in p["additional_skills"]]
        ) or '<span style="color: #047857; font-weight: 500;">All skills already in your profile</span>'

        intern_badge = (
            f'<span style="background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; border-radius: 12px; padding: 2px 10px; font-size: 0.78rem; font-weight: 600;">💼 {p["internship_role"]}</span>'
            if p["has_internship"]
            else '<span style="background: #f1f5f9; color: #64748b; border: 1px solid #cbd5e1; border-radius: 12px; padding: 2px 10px; font-size: 0.78rem; font-weight: 500;">No Prior Internship</span>'
        )

        st.markdown(f"""
        <div class="rec-card">
            <div class="rec-title-row">
                <div>
                    <div class="rec-title">#{p['rank']}. {p['student_name']} <span style="font-size: 0.86rem; font-weight: 500; color: #64748b;">({p['student_id']})</span></div>
                    <div style="font-size: 0.88rem; color: #475569; margin-top: 2px;">
                        🎯 <strong>{p['career_field']}</strong> • {p['technical_domain']} &nbsp;{intern_badge}
                    </div>
                </div>
                <div class="rec-badge">KNN Match: {p['similarity_pct']}% &nbsp;|&nbsp; d = {p['distance']:.3f}</div>
            </div>
            <div class="rec-meta-row">
                <div>🎓 <strong>Major:</strong> {p['major']} (Year {p['university_year']})</div>
                <div>📊 <strong>CGPA:</strong> {p['cgpa']} / {int(p['cgpa_scale'])}</div>
                <div>💻 <strong>Prog:</strong> {p['programming_skill']}/10</div>
                <div>🧩 <strong>Problem Solving:</strong> {p['problem_solving']}/10</div>
                <div>🗣️ <strong>Comm:</strong> {p['communication_skills']}/10</div>
                <div>🛠️ <strong>Projects:</strong> {p['projects_completed']}</div>
                <div>📜 <strong>Certs:</strong> {p['certifications']}</div>
                <div>🏢 <strong>Internships:</strong> {p['internships']}</div>
            </div>
            <div class="overlap-box">
                <div style="margin-bottom: 4px;">
                    <strong>Shared Technical Skills:</strong> {matched_html}
                </div>
                <div style="margin-bottom: 6px;">
                    <strong>Peer's Additional Skills:</strong> {add_html}
                </div>
                <div style="font-size: 0.78rem; color: #64748b;">
                    ℹ️ <em>{p['data_source']} • LinkedIn: {p['linkedin_status']} • Consent: {p['profile_consent']}</em>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # SECTION 3: Student Profile Comparison
    # =========================================================================
    st.markdown('<div class="content-card-title">📊 Student Profile Comparison</div>', unsafe_allow_html=True)
    st.caption(f"Side-by-side comparison of your profile ({user_norm['student_name']}) against your Top-{results['k']} KNN peers.")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Normalized CGPA (/10)", f"{user_norm['cgpa_10']:.2f}", f"{user_norm['cgpa_10'] - peer_avg['cgpa_10']:+.2f} vs Peer Avg")
    m2.metric("Programming (/10)", f"{int(user_norm['programming_skill'])}", f"{user_norm['programming_skill'] - peer_avg['programming_skill']:+.1f} vs Peer Avg")
    m3.metric("Problem Solving (/10)", f"{int(user_norm['problem_solving'])}", f"{user_norm['problem_solving'] - peer_avg['problem_solving']:+.1f} vs Peer Avg")
    m4.metric("Projects Completed", f"{int(user_norm['projects_completed'])}", f"{user_norm['projects_completed'] - peer_avg['projects_completed']:+.1f} vs Peer Avg")
    m5.metric("Certifications", f"{int(user_norm['certifications'])}", f"{user_norm['certifications'] - peer_avg['certifications']:+.1f} vs Peer Avg")

    comp_c1, comp_c2 = st.columns([1, 1.1], gap="large")
    with comp_c1:
        st.markdown('<div class="chart-title">Core Competencies (0–10 Scale)</div>', unsafe_allow_html=True)
        fig_radar = charts.plot_peer_competency_radar(user_norm, peer_avg, peers[0])
        st.pyplot(fig_radar, use_container_width=True)
    with comp_c2:
        st.markdown('<div class="chart-title">Practical Experience Comparison</div>', unsafe_allow_html=True)
        fig_exp = charts.plot_peer_experience_comparison(user_norm, peer_avg, peers[0])
        st.pyplot(fig_exp, use_container_width=True)

    with st.expander("📋 View Full Comparison Table (You vs. Matched Peers)"):
        comp_rows = [{
            "Profile": f"You ({user_norm['student_name']})",
            "Career Field": "Target (Recommended below)",
            "Year": user_norm["university_year"],
            "CGPA (/10)": user_norm["cgpa_10"],
            "Programming": int(user_norm["programming_skill"]),
            "Problem Solving": int(user_norm["problem_solving"]),
            "Communication": int(user_norm["communication_skills"]),
            "Projects": int(user_norm["projects_completed"]),
            "Certs": int(user_norm["certifications"]),
            "Internships": int(user_norm["internships"]),
            "Match %": 100.0,
        }]
        for p in peers:
            comp_rows.append({
                "Profile": f"#{p['rank']} {p['student_name']}",
                "Career Field": p["career_field"],
                "Year": p["university_year"],
                "CGPA (/10)": p["cgpa_10"],
                "Programming": p["programming_skill"],
                "Problem Solving": p["problem_solving"],
                "Communication": p["communication_skills"],
                "Projects": p["projects_completed"],
                "Certs": p["certifications"],
                "Internships": p["internships"],
                "Match %": p["similarity_pct"],
            })
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # SECTION 4: Career Recommendations
    # =========================================================================
    st.markdown('<div class="content-card-title">🧭 Career Recommendations</div>', unsafe_allow_html=True)
    st.caption("Derived via similarity-weighted voting across your K nearest student neighbors.")

    top_career = career_recs[0]
    car_c1, car_c2 = st.columns([1.05, 1.35], gap="large")
    with car_c1:
        st.markdown(f"""
        <div class="rec-card" style="border-left-color: #0d9488; height: 100%;">
            <div style="font-size: 0.82rem; font-weight: 700; color: #0f766e; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">
                Top KNN Career Alignment
            </div>
            <div style="font-size: 1.45rem; font-weight: 800; color: #0f172a; margin-bottom: 6px;">
                {top_career['career_field']}
            </div>
            <div style="font-size: 0.94rem; color: #334155; margin-bottom: 10px;">
                <strong>Technical Domain:</strong> {top_career['technical_domain']}<br>
                <strong>Target Internship Role:</strong> {top_career['recommended_internship_role']}
            </div>
            <div class="overlap-box">
                <strong>Neighbor Consensus:</strong> {top_career['peer_count']} of {top_career['total_k']} nearest peers<br>
                <strong>Weighted KNN Support:</strong> {top_career['weighted_share_pct']}%<br>
                <strong>Avg Peer Similarity:</strong> {top_career['avg_peer_similarity_pct']}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    with car_c2:
        fig_career = charts.plot_career_support_chart(career_recs)
        st.pyplot(fig_career, use_container_width=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # SECTION 5: Recommended Internships (TF-IDF + Cosine Similarity Bridge)
    # =========================================================================
    st.markdown('<div class="content-card-title">💼 Recommended Internships</div>', unsafe_allow_html=True)
    st.caption("Matches your profile against all 388 listings in `technical_internships_ready.csv` using the existing TF-IDF & Cosine Similarity engine.")

    query_mode = st.radio(
        "Internship Query Mode",
        options=[
            "Use My Current Skills",
            "Simulate: My Skills + Top Recommended Peer Skills",
        ],
        horizontal=True,
        key="peer_internship_query_mode",
    )

    base_skills = list(user_norm["technical_skills"])
    if query_mode.startswith("Simulate") and results["top_suggested_skills"]:
        rec_query_skills = sorted(list(set(base_skills + results["top_suggested_skills"])))
        st.info(f"Simulating matches with added peer skills: **{', '.join(results['top_suggested_skills'])}**")
    else:
        rec_query_skills = base_skills if base_skills else [top_career["career_field"], top_career["technical_domain"]]

    internship_recs, internship_meta = engine.recommend_internships(
        student_skills=rec_query_skills,
        top_n=5,
    )

    if internship_recs:
        for idx, item in enumerate(internship_recs, start=1):
            matched_html = "".join(
                [f'<span class="tag-matched">✓ {m}</span>' for m in item["matched_skills"]]
            ) or '<span style="color: #94a3b8; font-style: italic;">Matched via role title / TF-IDF keywords</span>'
            missing_html = "".join(
                [f'<span class="tag-missing">+ {m}</span>' for m in item["missing_skills"][:6]]
            ) or '<span style="color: #047857; font-weight: 500;">All listed skills covered</span>'

            st.markdown(f"""
            <div class="rec-card">
                <div class="rec-title-row">
                    <div class="rec-title">#{idx}. {item['title']}</div>
                    <div class="rec-badge">Cosine Similarity: {item['similarity_score']:.3f} ({item['similarity_percentage']}%)</div>
                </div>
                <div class="rec-meta-row">
                    <div>📍 <strong>Location:</strong> {item['location']}</div>
                    <div>💰 <strong>Stipend:</strong> {item['salary_display']}</div>
                </div>
                <div style="font-size: 0.88rem; margin-bottom: 8px;">
                    <strong>Required Skills in Listing:</strong> {item['required_skills']}
                </div>
                <div class="overlap-box">
                    <div style="margin-bottom: 4px;"><strong>Your Matching Skills:</strong> {matched_html}</div>
                    <div><strong>Skills to Learn:</strong> {missing_html}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if st.session_state.get("is_authenticated", False):
            if st.button("🎯 Open Full Resume & ML Analysis Dashboard (Page 3) →", key="btn_peer_to_p3"):
                st.session_state.confirmed_skills = list(user_norm["technical_skills"])
                st.session_state["_sync_skills_widget"] = True
                st.session_state.recommendations = None
                st.session_state.rec_metadata = None
                navigate_to(3)
    else:
        st.warning(internship_meta.get("message", "No matching internships found for the current query skills."))

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # SECTION 6: Personalized Skill Improvement
    # =========================================================================
    st.markdown('<div class="content-card-title">🚀 Personalized Skill Improvement</div>', unsafe_allow_html=True)

    imp_c1, imp_c2 = st.columns([1.05, 1.15], gap="large")
    with imp_c1:
        st.markdown("**High-Impact Technical Skills from Similar Peers**")
        if skill_gaps:
            for g in skill_gaps[:6]:
                demand_txt = (
                    f"Appears in <strong>{g['internship_dataset_listings']}</strong> internship listings"
                    if g["internship_dataset_listings"] > 0
                    else "Valued in peer cohort"
                )
                st.markdown(f"""
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-left: 4px solid #0d9488; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong style="color: #0f172a; font-size: 0.98rem;">+ {g['skill']}</strong>
                        <span style="font-size: 0.8rem; color: #0f766e; background: #ecfdf5; border: 1px solid #a7f3d0; padding: 2px 8px; border-radius: 12px; font-weight: 600;">
                            {g['peer_count']}/{g['total_k']} peers ({g['peer_prevalence_pct']}%)
                        </span>
                    </div>
                    <div style="font-size: 0.83rem; color: #475569; margin-top: 4px;">
                        {demand_txt}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            top_gap_skills = [g["skill"] for g in skill_gaps[:3]]
            if st.button(
                f"➕ Add Top {len(top_gap_skills)} Suggested Skill(s) & Re-run KNN",
                key="btn_add_suggested_skills",
                use_container_width=True,
            ):
                updated = dict(st.session_state.peer_profile)
                merged = list(updated.get("technical_skills", []))
                for s in top_gap_skills:
                    if s not in merged:
                        merged.append(s)
                updated["technical_skills"] = merged
                st.session_state.peer_profile = updated
                st.session_state.peer_match_results = None
                st.rerun()
        else:
            st.success("✅ Your technical skills already cover all skills held by your top-K nearest peers!")

    with imp_c2:
        st.markdown("**Academic & Experience Growth Benchmarks**")
        for item in improvements:
            if item["status"] == "Opportunity to Grow":
                badge_style = "background: #fffbeb; color: #b45309; border: 1px solid #fde68a;"
                border_col = "#f59e0b"
            elif item["status"] == "Near Benchmark":
                badge_style = "background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe;"
                border_col = "#3b82f6"
            else:
                badge_style = "background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0;"
                border_col = "#10b981"

            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-left: 4px solid {border_col}; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 3px;">
                    <strong style="color: #0f172a; font-size: 0.94rem;">{item['dimension']}</strong>
                    <span style="{badge_style} padding: 2px 8px; border-radius: 12px; font-size: 0.76rem; font-weight: 600; white-space: nowrap;">
                        {item['status']}
                    </span>
                </div>
                <div style="font-size: 0.83rem; color: #475569; line-height: 1.4;">
                    {item['recommendation']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # SECTION 7: KNN Model Insights
    # =========================================================================
    st.markdown('<div class="content-card-title">🔬 KNN Model Insights</div>', unsafe_allow_html=True)

    summary = peer_engine.get_dataset_summary()
    w_skills, w_num, w_major = results["weights"]

    ins_c1, ins_c2 = st.columns([1.25, 1], gap="large")
    with ins_c1:
        st.markdown('<div class="chart-title">Neighbor Ranking & K-Cutoff Profile</div>', unsafe_allow_html=True)
        st.caption("Top 15 synthetic students ranked by KNN similarity; blue bars indicate your selected K nearest neighbors.")
        fig_knn = charts.plot_knn_neighbors_chart(
            results["all_ranked"], k=results["k"], metric=results["metric"]
        )
        st.pyplot(fig_knn, use_container_width=True)

    with ins_c2:
        st.markdown('<div class="chart-title">Pipeline Architecture & Feature Weights</div>', unsafe_allow_html=True)
        st.markdown(f"""
        - **Dataset:** `synthetic_student_profiles_30.csv` ({summary['total_students']} profiles, {summary['career_fields_count']} career fields, {summary['unique_skills_count']} unique technical skills)
        - **Active Preset:** `{results['weight_preset']}`
          - Technical Skills TF-IDF block: **{int(w_skills * 100)}%**
          - Standardized Numeric Profile block (`Year`, `CGPA/10`, `Prog`, `ProbSolving`, `Comm`, `Projects`, `Certs`, `Internships`): **{int(w_num * 100)}%**
          - Academic Major One-Hot block: **{int(w_major * 100)}%**
        - **Distance Metric:** `{results['metric'].title()}` (`K = {results['k']}`)
        - **Similarity Mapping:** $\\text{{sim}} = \\frac{{1}}{{1 + d}}$ (Euclidean) or $1 - d_{{\\text{{cos}}}}$ (Cosine)
        """)

    with st.expander("🧪 Honest ML Evaluation & Automated KNN Validation Checks"):
        st.markdown("""
        - **Dataset Scope & Limitations:** `synthetic_student_profiles_30.csv` contains **30 fictional demo student profiles** (`Synthetic demo`, `Placeholder - not a real profile`). It does not measure real-world hiring or career placement outcomes.
        - **Why KNN Similarity Is Not Hiring Probability:** Similarity reflects geometric proximity in the weighted feature space between your profile and synthetic profiles, not guaranteed career success.
        - **Deterministic Functional Checks:** Below are live verification checks plus an internal Leave-One-Out (LOO) consistency check across the 30 synthetic profiles:
        """)
        knn_checks = peer_engine.run_knn_validation_checks()
        st.dataframe(pd.DataFrame(knn_checks), use_container_width=True, hide_index=True)

    with st.expander("📂 Browse All 30 Synthetic Student Profiles (Dataset Reference)"):
        raw_df = peer_engine.get_all_students()
        display_cols = [
            "Student_ID",
            "Student_Name",
            "Major",
            "University_Year",
            "CGPA",
            "CGPA_Scale",
            "Technical_Skills",
            "Programming_Skill",
            "Problem_Solving",
            "Communication_Skills",
            "Projects_Completed",
            "Certifications",
            "Internships",
            "Career_Field",
            "Technical_Domain",
            "Internship_Role",
            "Data_Source",
        ]
        st.dataframe(raw_df[display_cols], use_container_width=True, hide_index=True)
