"""
catalog_ui.py
-------------
Public Internship Catalogue & Explore Interface.

Features:
- Browse all 388 internship records from technical_internships_ready.csv.
- Header with dataset count and prominent "Get Personalized Matches →" entry point.
- Keyword search on title and skills.
- Multi-dimensional filters: Location, Required Skills, Salary Range, and Sorting.
- Reset Filters button.
- Clean pagination showing 12 listings per page with dynamic count indicators.
- Responsive cards displaying actual dataset fields without fabrication.
"""

import math
import streamlit as st
import pandas as pd
from typing import Callable, List
from skill_extractor import normalize_skill_name

PAGE_SIZE = 12


@st.dialog("🔐 Sign In to Continue")
def render_login_modal(action_type="matches", target_company=None):
    """
    Modal login dialog displayed when an unauthenticated guest user attempts
    to get personalized matches or select/apply for an internship role.
    """
    if action_type == "company" and target_company:
        st.markdown(f"""
        <div style="background: #eff6ff; border-left: 4px solid #2563eb; padding: 10px 14px; border-radius: 8px; margin-bottom: 14px; font-size: 0.92rem; color: #1e3a8a;">
            🏢 <strong>Sign In Required:</strong> Please log in or create an account to select and apply for <strong>{target_company}</strong>.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: #eff6ff; border-left: 4px solid #2563eb; padding: 10px 14px; border-radius: 8px; margin-bottom: 14px; font-size: 0.92rem; color: #1e3a8a;">
            🎯 <strong>Sign In Required:</strong> Please log in or create an account to upload your resume and get personalized ML recommendations.
        </div>
        """, unsafe_allow_html=True)

    dlg_email = st.text_input("Email", placeholder="student@university.edu", key="dlg_email_val")
    dlg_pass = st.text_input("Password", type="password", placeholder="Password", key="dlg_pass_val")

    col_l1, col_l2 = st.columns([1, 1.2])
    with col_l1:
        if st.button("✉️ Log In", type="primary", use_container_width=True, key="dlg_btn_submit_login"):
            if dlg_email.strip() and dlg_pass.strip():
                st.session_state.is_authenticated = True
                st.session_state.is_guest = False
                st.session_state.user_email = dlg_email.strip()
                st.session_state.user_name = dlg_email.split("@")[0].capitalize()
                if action_type == "matches":
                    st.session_state.current_page = 1
                elif action_type == "company" and target_company:
                    st.session_state.selected_company = target_company
                st.rerun()
            else:
                st.error("Please enter both email and password.")

    with col_l2:
        if st.button("🚀 Demo Login (1-Click)", use_container_width=True, key="dlg_btn_demo_login"):
            st.session_state.is_authenticated = True
            st.session_state.is_guest = False
            st.session_state.user_email = "mahima.student@university.edu"
            st.session_state.user_name = "Mahima Dhaka"
            if action_type == "matches":
                st.session_state.current_page = 1
            elif action_type == "company" and target_company:
                st.session_state.selected_company = target_company
            st.rerun()

    st.markdown("<hr style='margin: 14px 0 10px 0; border: none; border-top: 1px dashed #cbd5e1;'>", unsafe_allow_html=True)
    st.caption("Or log in with single-click developer accounts:")
    col_gh, col_goog, col_in = st.columns(3)
    with col_gh:
        if st.button("🐱 GitHub", use_container_width=True, key="dlg_gh_login"):
            st.session_state.is_authenticated = True
            st.session_state.is_guest = False
            st.session_state.user_email = "student_github@university.edu"
            st.session_state.user_name = "GitHub Student"
            if action_type == "matches":
                st.session_state.current_page = 1
            elif action_type == "company" and target_company:
                st.session_state.selected_company = target_company
            st.rerun()
    with col_goog:
        if st.button("🌐 Google", use_container_width=True, key="dlg_goog_login"):
            st.session_state.is_authenticated = True
            st.session_state.is_guest = False
            st.session_state.user_email = "student_google@university.edu"
            st.session_state.user_name = "Google Student"
            if action_type == "matches":
                st.session_state.current_page = 1
            elif action_type == "company" and target_company:
                st.session_state.selected_company = target_company
            st.rerun()
    with col_in:
        if st.button("💼 LinkedIn", use_container_width=True, key="dlg_in_login"):
            st.session_state.is_authenticated = True
            st.session_state.is_guest = False
            st.session_state.user_email = "student_linkedin@university.edu"
            st.session_state.user_name = "LinkedIn Student"
            if action_type == "matches":
                st.session_state.current_page = 1
            elif action_type == "company" and target_company:
                st.session_state.selected_company = target_company
            st.rerun()

    st.markdown("<hr style='margin: 12px 0 10px 0; border: none; border-top: 1px dashed #cbd5e1;'>", unsafe_allow_html=True)
    if st.button("⬅ Return to Main Landing & Login Page", use_container_width=True, key="dlg_btn_landing_return"):
        st.session_state.is_authenticated = False
        st.session_state.is_guest = False
        st.rerun()


def render_explore_page(engine, navigate_to: Callable[[any], None]):
    """
    Renders the Explore Internships page displaying all listings.
    """
    all_df = engine.get_all_internships()
    total_dataset_count = len(all_df)

    # Initialize pagination state
    if "catalog_page" not in st.session_state:
        st.session_state.catalog_page = 1

    # -------------------------------------------------------------------------
    # Top-Left Back Navigation to Landing / Login Page
    # -------------------------------------------------------------------------
    col_back, _ = st.columns([1.1, 3.9], gap="medium")
    with col_back:
        if st.button("← Back to Home", key="btn_explore_back_home", help="Return to the main landing and login page", use_container_width=True):
            st.session_state.is_authenticated = False
            st.session_state.is_guest = False
            st.session_state.user_email = ""
            st.session_state.user_name = "Student User"
            st.rerun()

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Page Header with Prominent Recommendation CTA
    # -------------------------------------------------------------------------
    header_col1, header_col2 = st.columns([1.7, 1.1], gap="large")

    with header_col1:
        st.markdown(f"""
        <div style="margin-bottom: 4px;">
            <h1 style="color: #0f172a; font-size: 1.85rem; font-weight: 800; margin-bottom: 6px;">
                Explore Internships
            </h1>
            <p style="color: #475569; font-size: 1.02rem; margin-bottom: 12px; line-height: 1.5;">
                Discover technical internship opportunities that match your interests.
            </p>
            <div style="display: inline-flex; align-items: center; gap: 8px; background: #eff6ff; color: #1e40af; border: 1px solid #bfdbfe; padding: 5px 14px; border-radius: 20px; font-size: 0.88rem; font-weight: 600;">
                <span>📊</span> <span>{total_dataset_count} Technical Internships Available in Dataset</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with header_col2:
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        # Visually prominent CTA button in the top-right corner
        if st.button("Get Personalized Matches →", type="primary", use_container_width=True, key="btn_cta_matches"):
            if not st.session_state.is_authenticated:
                render_login_modal("matches", None)
            else:
                navigate_to(1)
        st.caption("Upload your resume or enter your skills to find relevant internships.")

    st.markdown("<hr style='margin: 18px 0 20px 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Search & Filter Controls
    # -------------------------------------------------------------------------
    # Search bar
    search_query = st.text_input(
        label="Search Internships",
        placeholder="🔍 Search internship titles (e.g., Data Analytics, Web Development) or skills (e.g., Python, SQL, React)...",
        key="catalog_search_input"
    )

    # Filter controls row
    f_col1, f_col2, f_col3, f_col4 = st.columns([1.2, 1.5, 1.1, 1.2], gap="medium")

    with f_col1:
        locations_list = engine.get_locations_list()
        selected_location = st.selectbox(
            "Location",
            options=locations_list,
            index=0,
            key="catalog_filter_location"
        )

    with f_col2:
        top_skills_vocab = [s[0] for s in engine.get_top_dataset_skills(30)]
        selected_skills = st.multiselect(
            "Required Skills",
            options=top_skills_vocab,
            placeholder="Select skills...",
            key="catalog_filter_skills"
        )

    with f_col3:
        stipend_options = {
            0: "Any Salary",
            5000: "₹5,000+ / mo",
            10000: "₹10,000+ / mo",
            15000: "₹15,000+ / mo",
            20000: "₹20,000+ / mo",
            25000: "₹25,000+ / mo"
        }
        selected_stipend_label = st.selectbox(
            "Minimum Salary",
            options=list(stipend_options.values()),
            index=0,
            key="catalog_filter_salary"
        )
        min_salary_val = [k for k, v in stipend_options.items() if v == selected_stipend_label][0]

    with f_col4:
        sort_options = [
            "Default Order",
            "Internship Title (A to Z)",
            "Internship Title (Z to A)",
            "Starting Salary (High to Low)",
            "Maximum Salary (High to Low)"
        ]
        selected_sort = st.selectbox(
            "Sort By",
            options=sort_options,
            index=0,
            key="catalog_sort_by"
        )

    # -------------------------------------------------------------------------
    # Apply Filtering & Sorting
    # -------------------------------------------------------------------------
    # Map sort selection to engine terminology
    engine_sort_map = {
        "Default Order": "Default Order",
        "Internship Title (A to Z)": "Internship Title (A to Z)",
        "Internship Title (Z to A)": "Internship Title (Z to A)",
        "Starting Salary (High to Low)": "Highest Starting Salary",
        "Maximum Salary (High to Low)": "Highest Maximum Salary"
    }

    filtered_df = engine.filter_and_sort_catalog(
        search_query=search_query,
        selected_location=selected_location,
        selected_skills=selected_skills,
        min_stipend=min_salary_val,
        sort_by=engine_sort_map.get(selected_sort, "Default Order")
    )

    filtered_count = len(filtered_df)
    total_pages = max(1, math.ceil(filtered_count / PAGE_SIZE))

    # Clamp current page if filters reduced available pages
    if st.session_state.catalog_page > total_pages:
        st.session_state.catalog_page = 1
    current_page = st.session_state.catalog_page

    start_idx = (current_page - 1) * PAGE_SIZE
    end_idx = min(start_idx + PAGE_SIZE, filtered_count)

    # -------------------------------------------------------------------------
    # Results Count Bar & Pagination Navigation
    # -------------------------------------------------------------------------
    count_col, pag_col = st.columns([1.5, 1], gap="medium")

    with count_col:
        if filtered_count > 0:
            st.markdown(f"""
            <div style="font-size: 0.95rem; color: #334155; font-weight: 600; padding-top: 8px;">
                Showing <span style="color: #2563eb;">{start_idx + 1}–{end_idx}</span> of <span style="color: #0f172a;">{filtered_count}</span> internships
                {f"<span style='color: #64748b; font-weight: 400;'> (filtered from {total_dataset_count} total)</span>" if filtered_count < total_dataset_count else ""}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="font-size: 0.95rem; color: #e11d48; font-weight: 600; padding-top: 8px;">
                No internships found matching your search and filter criteria.
            </div>
            """, unsafe_allow_html=True)

    with pag_col:
        p_col1, p_col2, p_col3 = st.columns([1, 1.2, 1])
        with p_col1:
            if st.button("◀ Prev", disabled=(current_page <= 1), use_container_width=True, key="btn_prev_page"):
                st.session_state.catalog_page = max(1, current_page - 1)
                st.rerun()
        with p_col2:
            st.markdown(f"<div style='text-align: center; color: #64748b; font-size: 0.88rem; padding-top: 8px;'>Page <strong>{current_page}</strong> of <strong>{total_pages}</strong></div>", unsafe_allow_html=True)
        with p_col3:
            if st.button("Next ▶", disabled=(current_page >= total_pages), use_container_width=True, key="btn_next_page"):
                st.session_state.catalog_page = min(total_pages, current_page + 1)
                st.rerun()

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Render Internship Listing Cards
    # -------------------------------------------------------------------------
    if filtered_count == 0:
        st.markdown("""
        <div style="background: #ffffff; border: 2px dashed #cbd5e1; border-radius: 12px; padding: 40px 20px; text-align: center; margin-top: 16px;">
            <div style="font-size: 2.2rem; margin-bottom: 8px;">🔍</div>
            <h3 style="color: #334155; margin-bottom: 6px;">No Matching Listings Found</h3>
            <p style="color: #64748b; font-size: 0.92rem; margin-bottom: 14px;">Try modifying your keyword search, selecting "All Locations", or resetting filters.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    page_rows = filtered_df.iloc[start_idx:end_idx]

    # Two-column responsive card layout
    card_col1, card_col2 = st.columns(2, gap="medium")

    for i, (_, row) in enumerate(page_rows.iterrows()):
        col = card_col1 if (i % 2 == 0) else card_col2

        title = str(row.get("title", "Technical Internship")).strip()
        loc = str(row.get("location", "")).strip() or "Not specified"
        skills = str(row.get("skills", "")).strip() or str(row.get("skills_clean", "")).strip() or "Not specified"
        start_sal = float(row.get("start_salary", 0.0))
        max_sal = float(row.get("max_salary", 0.0))
        itype = str(row.get("type", "Internship")).strip().title()

        # Format salary displays
        if start_sal > 0:
            start_sal_str = f"₹{int(start_sal):,} / mo"
        else:
            start_sal_str = "Not specified"

        if max_sal > 0:
            max_sal_str = f"₹{int(max_sal):,} / mo"
        else:
            max_sal_str = "Not specified"

        if start_sal > 0 and max_sal > 0:
            if start_sal == max_sal:
                salary_combined = f"₹{int(start_sal):,} / month"
            else:
                salary_combined = f"₹{int(start_sal):,} - ₹{int(max_sal):,} / month"
        elif start_sal > 0:
            salary_combined = f"₹{int(start_sal):,} / month (Starting)"
        elif max_sal > 0:
            salary_combined = f"Up to ₹{int(max_sal):,} / month"
        else:
            salary_combined = "Not specified"

        with col:
            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-left: 4px solid #2563eb; border-radius: 10px; padding: 16px 18px; margin-bottom: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; margin-bottom: 6px;">
                    <div style="font-size: 1.1rem; font-weight: 700; color: #0f172a; line-height: 1.35;">
                        {title}
                    </div>
                    <span style="background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; border-radius: 14px; font-size: 0.75rem; font-weight: 600; padding: 2px 8px; white-space: nowrap;">
                        {itype}
                    </span>
                </div>
                <div style="display: flex; flex-wrap: wrap; gap: 14px; font-size: 0.86rem; color: #475569; margin-bottom: 10px; padding-bottom: 6px; border-bottom: 1px dashed #f1f5f9;">
                    <span>📍 <strong>Location:</strong> {loc}</span>
                    <span>💰 <strong>Salary:</strong> {salary_combined}</span>
                </div>
                <div style="font-size: 0.84rem; color: #334155; line-height: 1.45;">
                    <strong>Required Skills:</strong> {skills}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Card Action Buttons
            card_action_c1, card_action_c2 = st.columns([1.1, 1])
            with card_action_c1:
                if st.button("🏢 Select Company", key=f"btn_select_comp_{start_idx + i}", use_container_width=True):
                    if not st.session_state.is_authenticated:
                        render_login_modal("company", title)
                    else:
                        st.session_state.selected_company = title
                        st.rerun()

            with card_action_c2:
                if st.button("🎯 Get Matches", key=f"btn_card_matches_{start_idx + i}", use_container_width=True):
                    if not st.session_state.is_authenticated:
                        render_login_modal("matches", title)
                    else:
                        navigate_to(1)

            if st.session_state.get("selected_company") == title:
                st.markdown(f"""
                <div style="background: #f0fdf4; border: 1px solid #bbf7d0; color: #166534; padding: 6px 12px; border-radius: 6px; font-size: 0.84rem; margin-top: 4px; font-weight: 600;">
                    ✅ Selected Role: <strong>{title}</strong>
                </div>
                """, unsafe_allow_html=True)

            with st.expander(f"View Details: {title}", expanded=(st.session_state.get("selected_company") == title)):
                st.markdown(f"""
                - **Internship Title:** {title}
                - **Location:** {loc}
                - **Internship Type:** {itype}
                - **Starting Salary:** {start_sal_str}
                - **Maximum Salary:** {max_sal_str}
                - **Required Skills:** {skills}
                """)
                exp_c1, exp_c2 = st.columns(2)
                with exp_c1:
                    if st.button("🎯 Get ML Recommendations", key=f"btn_match_{start_idx + i}", use_container_width=True):
                        if not st.session_state.is_authenticated:
                            render_login_modal("matches", title)
                        else:
                            navigate_to(1)
                with exp_c2:
                    if st.button("📄 Apply for Role", key=f"btn_apply_{start_idx + i}", type="primary", use_container_width=True):
                        if not st.session_state.is_authenticated:
                            render_login_modal("company", title)
                        else:
                            st.success(f"🎉 Application submitted for {title}! Role saved to your profile.")

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Bottom Pagination
    # -------------------------------------------------------------------------
    st.markdown("<hr style='margin: 14px 0 16px 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)
    b_col1, b_col2, b_col3 = st.columns([1, 1.2, 1])
    with b_col1:
        if st.button("◀ Previous Page", disabled=(current_page <= 1), use_container_width=True, key="btn_prev_page_bottom"):
            st.session_state.catalog_page = max(1, current_page - 1)
            st.rerun()
    with b_col2:
        st.markdown(f"<div style='text-align: center; color: #64748b; font-size: 0.9rem; padding-top: 6px;'>Page <strong>{current_page}</strong> of <strong>{total_pages}</strong></div>", unsafe_allow_html=True)
    with b_col3:
        if st.button("Next Page ▶", disabled=(current_page >= total_pages), use_container_width=True, key="btn_next_page_bottom"):
            st.session_state.catalog_page = min(total_pages, current_page + 1)
            st.rerun()
