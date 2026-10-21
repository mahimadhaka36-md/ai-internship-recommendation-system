"""
app.py
------
AI Internship Recommendation System — Three Pages + ML Visualization Dashboard
Academic second-year ML project using:
- TF-IDF Vectorization & Cosine Similarity
- In-memory PDF / DOCX parsing (PyMuPDF, python-docx)
- Explainable rule-based skill extraction and skill coverage overview
- Interactive ML analysis dashboard with 4 explainable visualizations
- Honest ML evaluation and functional validation checks
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any

from resume_parser import parse_resume
from skill_extractor import extract_skills, normalize_skill_name, get_skill_vocab
from recommendation import get_engine
import importlib
import charts
import auth
import catalog_ui
importlib.reload(auth)
importlib.reload(catalog_ui)

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Internship Recommendation System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Authentication Gate & Session State Initialization
# -----------------------------------------------------------------------------
auth.init_auth_state()

if "current_page" not in st.session_state:
    st.session_state.current_page = "explore"

if not st.session_state.get("is_authenticated", False) and not st.session_state.get("is_guest", False):
    auth.render_auth_page()
    st.stop()

# If an unauthenticated guest directly targets recommendation steps, redirect to auth
if not st.session_state.get("is_authenticated", False) and st.session_state.get("current_page") in [1, 2, 3]:
    st.session_state.is_guest = False
    auth.render_auth_page()
    st.stop()

# -----------------------------------------------------------------------------
# Custom Styling: Modern Minimal Academic Dashboard (Authenticated)
# Light background, dark navy text, restrained blue/teal accents, rounded cards
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Base typography and background */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Navigation Progress Steps */
    .step-nav-container {
        display: flex;
        justify-content: space-between;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px 24px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
    .step-item {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 0.95rem;
        font-weight: 500;
        color: #64748b;
    }
    .step-item.active {
        color: #1d4ed8;
        font-weight: 700;
    }
    .step-circle {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
        font-weight: 600;
        background: #f1f5f9;
        color: #64748b;
        border: 1px solid #cbd5e1;
    }
    .step-circle.active {
        background: #2563eb;
        color: #ffffff;
        border-color: #2563eb;
    }
    .step-circle.completed {
        background: #10b981;
        color: #ffffff;
        border-color: #10b981;
    }

    /* Page Header */
    .page-header-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 22px 26px;
        margin-bottom: 22px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
    .page-title {
        color: #0f172a;
        font-size: 1.7rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .page-subtitle {
        color: #475569;
        font-size: 1.02rem;
        margin-bottom: 8px;
    }
    .page-desc {
        color: #64748b;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    /* Section Titles */
    .content-card-title {
        color: #0f172a;
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Recommendation Cards */
    .rec-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #2563eb;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 16px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .rec-card:hover {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.06);
    }
    .rec-title-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 8px;
    }
    .rec-title {
        font-size: 1.18rem;
        font-weight: 700;
        color: #0f172a;
    }
    .rec-badge {
        background-color: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 4px 12px;
        white-space: nowrap;
    }
    .rec-meta-row {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        font-size: 0.88rem;
        color: #475569;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px dashed #e2e8f0;
    }
    .overlap-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 0.86rem;
        color: #334155;
    }
    .tag-matched {
        display: inline-block;
        background-color: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
        border-radius: 4px;
        padding: 2px 7px;
        font-size: 0.8rem;
        margin: 2px 4px 2px 0;
        font-weight: 500;
    }
    .tag-missing {
        display: inline-block;
        background-color: #f1f5f9;
        color: #64748b;
        border: 1px solid #cbd5e1;
        border-radius: 4px;
        padding: 2px 7px;
        font-size: 0.8rem;
        margin: 2px 4px 2px 0;
    }
    .tag-skill-chip {
        display: inline-block;
        background-color: #eff6ff;
        color: #1e40af;
        border: 1px solid #bfdbfe;
        border-radius: 6px;
        padding: 3px 8px;
        font-size: 0.82rem;
        margin: 2px 4px 2px 0;
    }
    .tag-unrec-chip {
        display: inline-block;
        background-color: #f8fafc;
        color: #64748b;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 3px 8px;
        font-size: 0.82rem;
        margin: 2px 4px 2px 0;
    }

    /* Privacy and Note Badges */
    .privacy-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.82rem;
        color: #047857;
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        padding: 5px 12px;
        border-radius: 6px;
        margin-top: 10px;
    }

    /* ML Dashboard Section */
    .dashboard-header {
        border-top: 2px solid #e2e8f0;
        padding-top: 24px;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    .dashboard-title {
        font-size: 1.45rem;
        font-weight: 700;
        color: #0f172a;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .dashboard-subtitle {
        font-size: 0.95rem;
        color: #475569;
        margin-top: 4px;
    }

    .chart-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 4px;
    }
    .chart-desc {
        font-size: 0.85rem;
        color: #64748b;
        margin-bottom: 12px;
        line-height: 1.45;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Session State Initialization & Navigation
# -----------------------------------------------------------------------------
if "current_page" not in st.session_state:
    st.session_state.current_page = "explore"

if "confirmed_skills" not in st.session_state:
    st.session_state.confirmed_skills = []

if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

if "rec_metadata" not in st.session_state:
    st.session_state.rec_metadata = None

if "resume_filename" not in st.session_state:
    st.session_state.resume_filename = None

if "parse_error" not in st.session_state:
    st.session_state.parse_error = None

if "extracted_resume_text" not in st.session_state:
    st.session_state.extracted_resume_text = None

engine = get_engine()


def navigate_to(page_key: Any):
    st.session_state.current_page = page_key
    st.rerun()


def reset_all_state():
    st.session_state.current_page = "explore"
    st.session_state.catalog_page = 1
    st.session_state.confirmed_skills = []
    st.session_state.recommendations = None
    st.session_state.rec_metadata = None
    st.session_state.resume_filename = None
    st.session_state.parse_error = None
    st.session_state.extracted_resume_text = None
    st.rerun()


# -----------------------------------------------------------------------------
# Sidebar: Project Info, Direct Navigation & Reset
# -----------------------------------------------------------------------------
with st.sidebar:
    if st.session_state.get("is_guest", False) and not st.session_state.is_authenticated:
        st.markdown("👤 **Visitor Profile:**  \n**Guest Explorer**  \n`Browsing Listings`")
        if st.button("🔑 Log In / Sign Up", use_container_width=True, key="btn_guest_login_sidebar"):
            st.session_state.is_guest = False
            st.session_state.is_authenticated = False
            st.rerun()
    else:
        user_display = st.session_state.get("user_name", "Student User")
        email_display = st.session_state.get("user_email", "student@university.edu")
        st.markdown(f"👤 **Student Profile:**  \n**{user_display}**  \n`{email_display}`")
        
        if st.button("🚪 Log Out", use_container_width=True, key="btn_logout_sidebar"):
            st.session_state.is_authenticated = False
            st.session_state.is_guest = False
            st.session_state.user_email = ""
            st.rerun()

    st.divider()

    st.markdown("### 🎓 Academic ML Project")
    st.markdown("""
    **Topic:** AI Internship Recommendation System  
    **Level:** Second-Year College ML Project  
    **Methodology:** Unsupervised Content-Based Filtering  
    **Feature Matrix:** TF-IDF (Unigrams & Bigrams)  
    **Ranking Metric:** Cosine Similarity  
    **Dataset:** 388 Listings (`technical_internships_ready.csv`)  
    """)

    st.divider()

    st.markdown("### 🧭 Navigation")
    nav_options = [
        ("explore", "🔍 Explore Internships"),
        (1, "1. 📄 Resume Upload"),
        (2, "2. 🛠️ Review & Confirm Skills"),
        (3, "3. 🎯 Recommendations & ML Analysis")
    ]
    page_keys = [opt[0] for opt in nav_options]
    nav_labels = [opt[1] for opt in nav_options]

    if st.session_state.current_page in page_keys:
        current_idx = page_keys.index(st.session_state.current_page)
    else:
        current_idx = 0

    selected_nav = st.radio(
        label="Select Page",
        options=nav_labels,
        index=current_idx,
        label_visibility="collapsed"
    )

    selected_page_key = page_keys[nav_labels.index(selected_nav)]
    if selected_page_key != st.session_state.current_page:
        navigate_to(selected_page_key)

    st.divider()

    st.markdown("### 🛡️ Privacy Guarantee")
    st.caption("Resumes are processed purely in-memory. No uploaded resume files or texts are stored permanently on disk or transmitted to third-party APIs.")

    st.divider()

    if st.button("🔄 Reset Project / Start Over", use_container_width=True):
        reset_all_state()


# -----------------------------------------------------------------------------
# Top Breadcrumb Progress Navigation (Shown during recommendation workflow)
# -----------------------------------------------------------------------------
if st.session_state.current_page in [1, 2, 3]:
    # Previous page mapping: top-left back button navigates to the previous screen
    prev_nav_map = {
        1: ("explore", "← Back to Explore", "Return to the public internship catalogue"),
        3: (2, "Back", "Return to Step 2: Review Skills")
    }
    prev_info = prev_nav_map.get(st.session_state.current_page)

    top_nav_col1, top_nav_col2 = st.columns([1.1, 3.9], gap="medium")
    with top_nav_col1:
        if prev_info:
            prev_target, prev_label, prev_tip = prev_info
            if st.button(prev_label, use_container_width=True, key=f"btn_top_back_page_{st.session_state.current_page}", help=prev_tip):
                navigate_to(prev_target)

    with top_nav_col2:
        step_1_class = "completed" if st.session_state.current_page > 1 else ("active" if st.session_state.current_page == 1 else "")
        step_2_class = "completed" if st.session_state.current_page > 2 else ("active" if st.session_state.current_page == 2 else "")
        step_3_class = "active" if st.session_state.current_page == 3 else ""

        step_1_text_class = "active" if st.session_state.current_page == 1 else ""
        step_2_text_class = "active" if st.session_state.current_page == 2 else ""
        step_3_text_class = "active" if st.session_state.current_page == 3 else ""

        st.markdown(f"""
        <div class="step-nav-container" style="margin-bottom: 0;">
            <div class="step-item {step_1_text_class}">
                <div class="step-circle {step_1_class}">1</div>
                <span>Upload Resume</span>
            </div>
            <div style="color: #cbd5e1; font-weight: bold; align-self: center;">──▶</div>
            <div class="step-item {step_2_text_class}">
                <div class="step-circle {step_2_class}">2</div>
                <span>Review & Confirm Skills</span>
            </div>
            <div style="color: #cbd5e1; font-weight: bold; align-self: center;">──▶</div>
            <div class="step-item {step_3_text_class}">
                <div class="step-circle {step_3_class}">3</div>
                <span>Recommendations & ML Analysis</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)


# =============================================================================
# DEFAULT PAGE — Explore Internships Catalogue
# =============================================================================
if st.session_state.current_page == "explore":
    catalog_ui.render_explore_page(engine, navigate_to)

# =============================================================================
# PAGE 1 — Resume Upload
# =============================================================================
elif st.session_state.current_page == 1:

    st.markdown("""
    <div class="page-header-card">
        <div class="page-title">📄 Step 1: Upload Your Resume</div>
        <div class="page-subtitle">Upload your resume document in PDF or DOCX format.</div>
        <div class="page-desc">
            The system will extract your resume text in-memory and parse it using our explainable technical skills dictionary.
            Scanned image-only PDFs without a text layer are detected, prompting you to provide a text-based document or enter skills manually.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_upload, col_guidance = st.columns([1.3, 1], gap="large")

    with col_upload:
        st.markdown('<div class="content-card-title">Upload Resume File</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            label="Upload resume document",
            type=["pdf", "docx"],
            help="Upload a standard PDF or DOCX resume (max 5 MB). Processed in-memory without saving to disk."
        )

        st.markdown("""
        <div class="privacy-badge">
            🔒 In-memory processing • No permanent storage • Confidential
        </div>
        """, unsafe_allow_html=True)

        # Process upload if newly provided
        if uploaded_file is not None:
            if st.session_state.resume_filename != uploaded_file.name:
                st.session_state.resume_filename = uploaded_file.name
                st.session_state.parse_error = None
                st.session_state.recommendations = None
                st.session_state.rec_metadata = None

                with st.spinner("Extracting text from resume in memory..."):
                    file_bytes = uploaded_file.read()
                    extracted_text, error = parse_resume(file_bytes, uploaded_file.name)

                    if error:
                        st.session_state.parse_error = error
                        st.session_state.extracted_resume_text = None
                        st.session_state.confirmed_skills = []
                    else:
                        st.session_state.parse_error = None
                        st.session_state.extracted_resume_text = extracted_text
                        detected = extract_skills(extracted_text)
                        st.session_state.confirmed_skills = detected

        # Display status feedback
        if st.session_state.parse_error:
            st.error(f"⚠️ {st.session_state.parse_error}")

        elif st.session_state.resume_filename and st.session_state.extracted_resume_text:
            char_count = len(st.session_state.extracted_resume_text)
            skill_count = len(st.session_state.confirmed_skills)
            st.success(
                f"✅ **{st.session_state.resume_filename}** parsed successfully ({char_count:,} characters extracted in memory). "
                f"Identified {skill_count} technical skill(s)."
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Navigation buttons
        nav_col1, nav_col2 = st.columns([1.2, 1])
        with nav_col1:
            if st.button("Continue to Skills Review ➔", type="primary", use_container_width=True):
                navigate_to(2)
        with nav_col2:
            if st.button("Skip Upload & Enter Skills Manually ➔", use_container_width=True):
                navigate_to(2)

    with col_guidance:
        st.markdown('<div class="content-card-title">📋 Document Guidelines</div>', unsafe_allow_html=True)
        st.markdown("""
        - **Accepted Formats:** Text-based PDF (`.pdf`) or Word (`.docx`).
        - **Size Limit:** Up to 5 MB.
        - **Scanned Resumes:** Documents created purely by taking photographs or flat image scans do not contain text layers. Optical Character Recognition (OCR) is not supported in this initial version.
        - **Privacy Assurance:** Resumes are processed purely through in-memory streams (`io.BytesIO`) and are never written to permanent disk storage.
        """)


# =============================================================================
# PAGE 2 — Review and Confirm Skills
# =============================================================================
elif st.session_state.current_page == 2:
    st.markdown("""
    <div class="page-header-card">
        <div class="page-title">🛠️ Step 2: Review and Confirm Skills</div>
        <div class="page-subtitle">Verify, edit, or supplement the technical skills extracted from your resume.</div>
        <div class="page-desc">
            The recommendation model relies on your confirmed technical skills to perform vector-space matching.
            Review the skills detected below, add any missing skills or tools, and examine the <strong>Skill Coverage Overview</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_edit, col_vis = st.columns([1.2, 1], gap="large")

    with col_edit:
        st.markdown('<div class="content-card-title">Confirmed Skill Profile</div>', unsafe_allow_html=True)

        all_vocab = get_skill_vocab()

        if not st.session_state.confirmed_skills:
            st.info("💡 No skills have been recognized yet. Use the selector below or quick-add buttons to enter your technical skills.")

        # Multiselect control for confirmed skills
        selected_skills = st.multiselect(
            label="Confirmed Technical Skills",
            options=sorted(list(set(all_vocab + st.session_state.confirmed_skills))),
            default=st.session_state.confirmed_skills,
            help="Select skills from vocabulary or press backspace to remove.",
            placeholder="Search or add technical skills..."
        )

        # Sync when edited
        if selected_skills != st.session_state.confirmed_skills:
            st.session_state.confirmed_skills = selected_skills
            # Reset stale recommendations so they recompute cleanly
            st.session_state.recommendations = None
            st.session_state.rec_metadata = None

        # Custom skill input
        custom_col1, custom_col2 = st.columns([3, 1])
        with custom_col1:
            custom_input = st.text_input(
                "Add Custom Skill",
                placeholder="e.g. PyTorch, Docker, TypeScript, AWS",
                label_visibility="collapsed"
            )
        with custom_col2:
            if st.button("➕ Add", use_container_width=True):
                if custom_input.strip():
                    norm_skill = normalize_skill_name(custom_input.strip())
                    if norm_skill not in st.session_state.confirmed_skills:
                        st.session_state.confirmed_skills.append(norm_skill)
                        st.session_state.recommendations = None
                        st.session_state.rec_metadata = None
                        st.rerun()

        # Quick-add pills
        with st.expander("💡 Quick-add common skills"):
            quick_skills = [
                "python", "java", "c++", "sql", "machine learning", "data analysis",
                "html", "css", "javascript", "react", "pandas", "numpy",
                "data structures", "git", "power bi", "django"
            ]
            q_cols = st.columns(4)
            for i, qs in enumerate(quick_skills):
                col = q_cols[i % 4]
                if col.button(f"+ {qs}", key=f"quick_{qs}", use_container_width=True):
                    if qs not in st.session_state.confirmed_skills:
                        st.session_state.confirmed_skills.append(qs)
                        st.session_state.recommendations = None
                        st.session_state.rec_metadata = None
                        st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # Navigation row
        find_btn = st.button(
            "Confirm Skills & Find Internships ➔",
            type="primary",
            use_container_width=True,
            disabled=(len(st.session_state.confirmed_skills) == 0)
        )
        if find_btn:
            # Recompute recommendations with latest confirmed skills
            with st.spinner("Computing TF-IDF cosine similarity against 388 listings..."):
                recs, meta = engine.recommend_internships(
                    student_skills=st.session_state.confirmed_skills,
                    top_n=5
                )
                st.session_state.recommendations = recs
                st.session_state.rec_metadata = meta
            navigate_to(3)

    # -------------------------------------------------------------------------
    # Page 2 Visualization: Skill Coverage Overview
    # -------------------------------------------------------------------------
    with col_vis:
        st.markdown('<div class="content-card-title">📊 Skill Coverage Overview</div>', unsafe_allow_html=True)
        st.caption("How your confirmed skills relate to the technical internship dataset vocabulary:")

        if st.session_state.confirmed_skills:
            coverage = engine.get_skill_coverage(st.session_state.confirmed_skills)
            fig_cov = charts.plot_skill_coverage_chart(coverage)
            st.pyplot(fig_cov, use_container_width=True)

            in_list = coverage["in_dataset_skills"]
            out_list = coverage["not_in_dataset_skills"]

            if in_list:
                st.markdown("**Recognized in Dataset Listings:**")
                in_html = "".join([f'<span class="tag-skill-chip">✓ {s}</span>' for s in in_list])
                st.markdown(in_html, unsafe_allow_html=True)

            if out_list:
                st.markdown("<br>**Outside Dataset Vocabulary:**", unsafe_allow_html=True)
                out_html = "".join([f'<span class="tag-unrec-chip">• {s}</span>' for s in out_list])
                st.markdown(out_html, unsafe_allow_html=True)

            st.caption(
                "ℹ️ *Note: Absence from the dataset vocabulary indicates that the 388 listings do not explicitly require that keyword; "
                "it does not mean the skill is unimportant.*"
            )
        else:
            st.markdown("""
            <div style="text-align: center; padding: 30px 10px; color: #94a3b8;">
                Add one or more skills on the left to see your dataset coverage overview.
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# PAGE 3 — Internship Recommendations and ML Analysis
# =============================================================================
elif st.session_state.current_page == 3:
    st.markdown("""
    <div class="page-header-card">
        <div class="page-title">🎯 Your Internship Recommendations</div>
        <div class="page-subtitle">Ranked matches based on content-based TF-IDF vector similarity.</div>
        <div class="page-desc">
            The system transformed your confirmed skills into a TF-IDF feature vector (unigrams and bigrams, English stop words removed) 
            and calculated the Cosine Similarity against all 388 technical internship records.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Top search status and actions bar
    top_col1, top_col2 = st.columns([2.5, 1], gap="medium")
    with top_col1:
        st.markdown("**Confirmed Skills Used for Search:**")
        if st.session_state.confirmed_skills:
            skills_html = "".join([f'<span class="tag-skill-chip">{s}</span>' for s in st.session_state.confirmed_skills])
            st.markdown(skills_html, unsafe_allow_html=True)
        else:
            st.warning("No skills provided. Please return to Page 2 to select your skills.")

    with top_col2:
        act_col1, act_col2 = st.columns(2)
        with act_col1:
            if st.button("✏️ Edit Skills", use_container_width=True):
                navigate_to(2)
        with act_col2:
            if st.button("🔄 Reset", use_container_width=True):
                reset_all_state()

    st.markdown("<br>", unsafe_allow_html=True)

    # Automatically compute recommendations if not already in session state
    if st.session_state.recommendations is None and st.session_state.confirmed_skills:
        with st.spinner("Calculating cosine similarity scores..."):
            recs, meta = engine.recommend_internships(
                student_skills=st.session_state.confirmed_skills,
                top_n=5
            )
            st.session_state.recommendations = recs
            st.session_state.rec_metadata = meta

    recs = st.session_state.recommendations
    meta = st.session_state.rec_metadata

    # -------------------------------------------------------------------------
    # Recommendation Results Cards
    # -------------------------------------------------------------------------
    if not st.session_state.confirmed_skills or (recs is not None and len(recs) == 0):
        msg = meta.get("message", "No matching internships found for the selected skills.") if meta else "No skills entered."
        st.warning(f"🔍 {msg}")
        if st.button("⬅ Return to Page 2 to Add Complementary Skills"):
            navigate_to(2)

    elif recs:
        st.success(f"Generated {len(recs)} ranked internship recommendations based on your {len(st.session_state.confirmed_skills)} confirmed skills.")

        for idx, item in enumerate(recs, start=1):
            title = item["title"]
            loc = item["location"]
            salary = item["salary_display"]
            score = item["similarity_score"]
            pct = item["similarity_percentage"]
            req_skills = item["required_skills"]
            matched = item["matched_skills"]
            missing = item["missing_skills"]

            matched_html = "".join([
                f'<span class="tag-matched">✓ {m}</span>' for m in matched
            ]) if matched else '<span style="color: #94a3b8; font-style: italic;">No exact vocabulary overlap</span>'

            missing_html = "".join([
                f'<span class="tag-missing">+ {m}</span>' for m in missing[:6]
            ]) if missing else '<span style="color: #047857; font-weight: 500;">None! All listed skills matched</span>'

            st.markdown(f"""
            <div class="rec-card">
                <div class="rec-title-row">
                    <div class="rec-title">#{idx}. {title}</div>
                    <div class="rec-badge">Cosine Similarity: {score:.3f} ({pct}%)</div>
                </div>
                <div class="rec-meta-row">
                    <div>📍 <strong>Location:</strong> {loc}</div>
                    <div>💰 <strong>Stipend:</strong> {salary}</div>
                </div>
                <div style="font-size: 0.88rem; margin-bottom: 8px;">
                    <strong>Required Skills in Listing:</strong> {req_skills}
                </div>
                <div class="overlap-box">
                    <div style="margin-bottom: 4px;">
                        <strong>Your Matching Skills:</strong> {matched_html}
                    </div>
                    <div>
                        <strong>Other Skills in Listing:</strong> {missing_html}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # ML VISUALIZATION DASHBOARD
        # ---------------------------------------------------------------------
        st.markdown("""
        <div class="dashboard-header">
            <div class="dashboard-title">📊 Understand Your Matches — ML Analysis</div>
            <div class="dashboard-subtitle">Visual interpretability of the recommendation scoring pipeline and dataset feature distributions.</div>
        </div>
        """, unsafe_allow_html=True)

        chart_col1, chart_col2 = st.columns(2, gap="large")

        # GRAPH A: Top 5 Internship Match Scores
        with chart_col1:
            st.markdown("""
            <div class="chart-title">Graph A: Top 5 Internship Match Scores</div>
            <div class="chart-desc">
                Horizontal ranking of cosine similarity scores (0.0 to 1.0). Higher cosine similarity indicates greater textual alignment 
                between your confirmed skills and the internship description. It is not a hiring probability.
            </div>
            """, unsafe_allow_html=True)
            fig_a = charts.plot_top_match_scores_chart(recs)
            st.pyplot(fig_a, use_container_width=True)

        # GRAPH B: Skill Match Breakdown
        with chart_col2:
            st.markdown("""
            <div class="chart-title">Graph B: Skill Match Breakdown</div>
            <div class="chart-desc">
                Compares your confirmed skills (constant profile reference) against the count of overlapping skills found in each internship.
                Overlap count is not a suitability percentage.
            </div>
            """, unsafe_allow_html=True)
            fig_b = charts.plot_skill_match_breakdown_chart(recs, len(st.session_state.confirmed_skills))
            st.pyplot(fig_b, use_container_width=True)

        # GRAPH C: Most In-Demand Skills in the Dataset
        st.markdown("""
        <div class="chart-title">Graph C: Most In-Demand Skills in the Dataset</div>
        <div class="chart-desc">
            Frequency distribution of technical skills across all 388 listings in <code>technical_internships_ready.csv</code>. 
            This chart shows how frequently recognized skills appear in the internship dataset. It describes the available listings and does not prove that these skills guarantee employment.
        </div>
        """, unsafe_allow_html=True)
        top_skills = engine.get_top_dataset_skills(15)
        fig_c = charts.plot_dataset_skill_demand_chart(top_skills)
        st.pyplot(fig_c, use_container_width=True)

        # GRAPH D: Advanced Similarity Heatmap (Expandable)
        with st.expander("🔬 Advanced Model Insights — Internship Similarity Heatmap (Click to Expand)"):
            st.markdown("""
            <div class="chart-desc">
                <strong>Graph D: Feature-Level Cosine Similarity Heatmap</strong><br>
                Shows the actual cosine similarity values between your full confirmed profile as well as individual skill components against each recommended internship's feature vector.
                <em>Note: Because these values are based on the same TF-IDF calculation used for ranking, the heatmap visualizes model output across query components; it is not independent evidence of recommendation accuracy.</em>
            </div>
            """, unsafe_allow_html=True)
            heatmap_df = engine.compute_similarity_heatmap_matrix(st.session_state.confirmed_skills, recs)
            fig_d = charts.plot_similarity_heatmap_chart(heatmap_df)
            st.pyplot(fig_d, use_container_width=True)

    # -------------------------------------------------------------------------
    # How the Recommendation Model Works
    # -------------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📘 How the Recommendation Model Works (Workflow & Components)"):
        st.markdown("""
        The recommendation pipeline operates through four modular steps:

        1. **Resume Text Extraction:**  
           The uploaded PDF or DOCX file is processed in memory using `PyMuPDF` or `python-docx`. Plain text is extracted from all document pages or paragraphs.

        2. **Rule-Based Skill Extraction:**  
           A curated technical skill dictionary scans the text using regex word boundary matching (`\\b<skill>\\b`). Long multiword phrases (e.g. *machine learning*, *data structures*) are matched first to prevent substring collisions.  
           *Note for examiners: This rule-based extractor is deterministic and explainable; it is NOT a trained NLP model.*

        3. **TF-IDF Feature Extraction:**  
           Internship listings (`title + skills_clean`) and student confirmed skills are vectorized using Scikit-Learn's `TfidfVectorizer` (unigrams + bigrams, English stop words removed). Each listing is transformed into a high-dimensional numerical vector.

        4. **Cosine Similarity Ranking:**  
           Pairwise cosine similarity measures the angle between the student skill vector $\\vec{A}$ and each internship listing vector $\\vec{B}$:
           $$\\text{Cosine Similarity}(\\vec{A}, \\vec{B}) = \\frac{\\vec{A} \\cdot \\vec{B}}{\\|\\vec{A}\\| \\|\\vec{B}\\|}$$
           Internships are sorted in descending order of similarity, filtered for duplicate listings, and the top five matches are displayed.
        """)

    # -------------------------------------------------------------------------
    # Honest ML Evaluation & Functional Validation
    # -------------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🧪 Model Evaluation & Limitations (Honest Academic Report)"):
        st.markdown("""
        ### Evaluation Methodology & Limitations
        The dataset (`technical_internships_ready.csv`) contains internship listings without ground-truth student-internship suitability labels or hiring outcomes.
        
        Therefore:
        - **Supervised metrics (Accuracy, Precision, Recall, F1-Score, ROC Curves)** are **not applicable** to this unsupervised content-based recommendation model. Fabricating such metrics would be academically dishonest.
        - High cosine similarity indicates **textual alignment**, not a hiring guarantee or real-world selection success.
        - True recommendation quality requires human evaluation or real-world student interaction logs.

        ### Automated Functional Validation Checks
        To verify operational correctness, the system runs automated functional checks on ranking order, constraint bounds, and edge cases:
        """)

        checks = engine.run_functional_validation_checks()
        check_df = pd.DataFrame(checks)
        st.dataframe(check_df, use_container_width=True, hide_index=True)

    st.markdown("<hr style='margin: 30px 0 18px 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)
    p3_bcol1, p3_bcol2 = st.columns([1, 1.4])
    with p3_bcol1:
        if st.button("⬅ Back to Skills Review", use_container_width=True, key="btn_p3_bottom_back"):
            navigate_to(2)
    with p3_bcol2:
        if st.button("🔍 Back to Explore Internships", use_container_width=True, key="btn_p3_bottom_explore"):
            navigate_to("explore")
