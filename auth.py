"""
auth.py
-------
Student Authentication & Landing Module (Two-Section Vertical Layout).
- Section 1: Welcome Hero with academic branding, headline, quote, description, and scroll cue.
- Section 2: Full Devpost-styled dark teal Login & Sign-Up authentication section.

Preserves all original authentication handlers, session management, social logins,
email/password form, signup toggle, and one-click demo login.
"""

import streamlit as st


def init_auth_state():
    """Initializes session state keys for authentication."""
    if "is_authenticated" not in st.session_state:
        st.session_state.is_authenticated = False

    if "is_guest" not in st.session_state:
        st.session_state.is_guest = False

    if "auth_mode" not in st.session_state:
        st.session_state.auth_mode = "login"

    if "user_email" not in st.session_state:
        st.session_state.user_email = ""

    if "user_name" not in st.session_state:
        st.session_state.user_name = "Student User"

    if "show_forgot_dialog" not in st.session_state:
        st.session_state.show_forgot_dialog = False


def render_auth_page():
    """
    Renders the two-section, vertically scrollable landing page:
    Section 1: Welcome Hero (Headline, Quote, Description, Explore CTA & Scroll Cue)
    Section 2: Authentication (Social logins, Email/Password, Signup, Demo Login)
    """
    init_auth_state()

    # Inject modern Google Fonts, layout styling, smooth scrolling, and dark teal aesthetics
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,400;0,500;0,600;0,700;0,800;1,400;1,500&family=Inter:wght@400;500;600;700&display=swap');

        /* Enable smooth scrolling across the entire page */
        html, body, [data-testid="stAppViewContainer"], .main {
            scroll-behavior: smooth !important;
            font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif !important;
        }

        /* Dark teal background matching project's academic Devpost palette */
        .stApp {
            background-color: #023644 !important;
            background-image: 
                radial-gradient(circle at 50% 15%, rgba(30, 111, 168, 0.22) 0%, rgba(2, 54, 68, 0) 65%),
                radial-gradient(circle at 80% 85%, rgba(245, 166, 35, 0.08) 0%, rgba(2, 54, 68, 0) 50%) !important;
            color: #ffffff !important;
        }

        /* Hide default sidebar on landing/login page for distraction-free view */
        [data-testid="stSidebar"] {
            display: none;
        }

        /* ------------------------------------------------------------- */
        /* SECTION 1: HERO CONTAINER (75-90% Initial Viewport Height)   */
        /* ------------------------------------------------------------- */
        .hero-container {
            min-height: 80vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            padding: 48px 20px 24px 20px;
            max-width: 860px;
            margin: 0 auto;
        }

        /* Subtle understated academic project pill badge */
        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(30, 111, 168, 0.22);
            border: 1px solid rgba(56, 189, 248, 0.32);
            padding: 6px 16px;
            border-radius: 9999px;
            color: #7dd3fc;
            font-size: 0.86rem;
            font-weight: 600;
            letter-spacing: 0.3px;
            margin-bottom: 22px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
        }

        /* Headline */
        .hero-headline {
            color: #ffffff;
            font-size: 2.85rem;
            font-weight: 800;
            line-height: 1.2;
            letter-spacing: -0.8px;
            margin-bottom: 20px;
            text-shadow: 0 2px 12px rgba(0, 0, 0, 0.25);
        }

        /* Quote */
        .hero-quote {
            color: #94b8c7;
            font-size: 1.22rem;
            font-weight: 500;
            line-height: 1.5;
            font-style: italic;
            max-width: 660px;
            margin: 0 auto 24px auto;
            border-left: 3px solid #f5a623;
            padding-left: 14px;
            text-align: left;
        }

        /* Description */
        .hero-description {
            color: #cbd5e1;
            font-size: 1.04rem;
            line-height: 1.68;
            max-width: 740px;
            margin: 0 auto 32px auto;
        }

        /* Scroll cue link */
        .scroll-cue-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-top: 18px;
            margin-bottom: 30px;
        }
        .scroll-cue-link {
            display: inline-flex;
            flex-direction: column;
            align-items: center;
            gap: 5px;
            text-decoration: none !important;
            color: #7dd3fc !important;
            font-size: 0.90rem;
            font-weight: 700;
            letter-spacing: 0.6px;
            text-transform: uppercase;
            padding: 10px 22px;
            border-radius: 9999px;
            background: rgba(30, 111, 168, 0.22);
            border: 1px solid rgba(56, 189, 248, 0.35);
            transition: all 0.25s ease-in-out;
            cursor: pointer;
        }
        .scroll-cue-link:hover {
            background: rgba(30, 111, 168, 0.45);
            border-color: #38bdf8;
            color: #ffffff !important;
            transform: translateY(2px);
        }
        .scroll-cue-arrow {
            font-size: 1.25rem;
            line-height: 1;
            animation: bounce 2s infinite;
        }
        @keyframes bounce {
            0%, 20%, 50%, 80%, 100% {
                transform: translateY(0);
            }
            40% {
                transform: translateY(5px);
            }
            60% {
                transform: translateY(3px);
            }
        }

        /* Section divider */
        .section-separator {
            width: 100%;
            max-width: 820px;
            margin: 20px auto 40px auto;
            border-top: 1px solid rgba(33, 91, 109, 0.55);
        }

        /* ------------------------------------------------------------- */
        /* SECTION 2: AUTHENTICATION CONTAINER                           */
        /* ------------------------------------------------------------- */
        #auth-section {
            scroll-margin-top: 36px;
        }
        .auth-container {
            max-width: 820px;
            margin: 10px auto 30px auto;
            text-align: center;
        }
        .auth-title {
            color: #ffffff;
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 8px;
            letter-spacing: -0.5px;
        }
        .auth-subtitle {
            color: #94b8c7;
            font-size: 1.02rem;
            margin-bottom: 32px;
        }

        /* Left side buttons */
        .social-subtext {
            color: #8daab6;
            font-size: 0.82rem;
            font-style: italic;
            margin-top: 14px;
            text-align: center;
        }

        /* OR Divider */
        .or-divider {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #8daab6;
            font-size: 0.88rem;
            font-weight: 600;
        }
        .or-line {
            width: 1px;
            height: 100px;
            border-left: 1px dashed #215b6d;
            margin: 10px 0;
        }

        /* Form labels */
        .auth-label {
            color: #ffffff;
            font-size: 0.95rem;
            font-weight: 600;
            text-align: left;
            margin-bottom: 4px;
        }

        /* Inputs */
        .stTextInput input {
            background-color: #ffffff !important;
            color: #0f172a !important;
            border-radius: 6px !important;
            border: 1px solid #cbd5e1 !important;
            padding: 10px 14px !important;
        }

        /* Forgot password link */
        .forgot-link-btn {
            background: none;
            border: none;
            color: #f5a623 !important;
            font-size: 0.85rem;
            padding: 0;
            cursor: pointer;
            text-decoration: underline;
            margin-top: 4px;
            margin-bottom: 12px;
            display: inline-block;
        }

        /* Primary yellow submit button */
        button[kind="primary"] {
            background-color: #f5a623 !important;
            border-color: #f5a623 !important;
            color: #000000 !important;
            font-weight: 700 !important;
            border-radius: 6px !important;
            font-size: 0.95rem !important;
            padding: 10px 18px !important;
            transition: all 0.2s ease !important;
        }
        button[kind="primary"]:hover {
            background-color: #e09612 !important;
            border-color: #e09612 !important;
            color: #000000 !important;
            transform: translateY(-1px);
        }

        /* Secondary social login buttons */
        button[kind="secondary"] {
            background-color: #1e6fa8 !important;
            border-color: #1a5e8f !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            border-radius: 6px !important;
            font-size: 0.93rem !important;
            padding: 9px 16px !important;
            text-align: left !important;
            transition: all 0.2s ease !important;
        }
        button[kind="secondary"]:hover {
            background-color: #2680be !important;
            border-color: #2680be !important;
            color: #ffffff !important;
            transform: translateY(-1px);
        }

        /* Mobile adjustments */
        @media (max-width: 768px) {
            .hero-headline {
                font-size: 1.95rem;
            }
            .hero-quote {
                font-size: 1.05rem;
                padding-left: 10px;
            }
            .hero-description {
                font-size: 0.94rem;
            }
            .auth-title {
                font-size: 1.75rem;
            }
            .or-line {
                height: 30px;
            }
        }
    </style>
    """, unsafe_allow_html=True)

    # =========================================================================
    # SECTION 1: WELCOME HERO (Top of Page — Initial Viewport)
    # =========================================================================
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">
            <span>🎓</span>
            <span>Academic ML Project</span>
            <span>•</span>
            <span>Content-Based Recommendation Engine</span>
        </div>
        <h1 class="hero-headline">Welcome to AI Internship Recommendation System</h1>
        <div class="hero-quote">
            “Your skills are your starting point. Let the right opportunity be your next step.”
        </div>
        <p class="hero-description">
            Discover technical internship opportunities, explore the skills employers are looking for, and find opportunities that align with your technical strengths. Browse internship listings freely or get personalized recommendations based on the skills extracted from your resume.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Optional Exploration CTA & Scroll Cue Buttons
    hero_col_left, hero_col_center, hero_col_right = st.columns([1, 1.4, 1])
    with hero_col_center:
        # Direct navigation to the public internship catalogue as a guest explorer
        if st.button("🔍 Explore Internship Opportunities", key="btn_hero_explore", use_container_width=True):
            st.session_state.is_authenticated = False
            st.session_state.is_guest = True
            st.session_state.user_name = "Guest Visitor"
            st.session_state.user_email = ""
            st.session_state.current_page = "explore"
            st.session_state.show_login_modal = None
            st.rerun()

    # Clear scroll cue navigating directly to Section 2 below
    st.markdown("""
    <div class="scroll-cue-container">
        <a href="#auth-section" target="_self" class="scroll-cue-link">
            <span>Get Started</span>
            <span class="scroll-cue-arrow">↓</span>
        </a>
    </div>
    <div class="section-separator"></div>
    """, unsafe_allow_html=True)

    # =========================================================================
    # SECTION 2: EXISTING LOGIN AND SIGNUP (Reached by scrolling down)
    # =========================================================================
    is_login = (st.session_state.auth_mode == "login")
    title_text = "Get Started"
    subtitle_text = "Sign in to explore personalized internship recommendations." if is_login else "Sign up to explore personalized internship recommendations."

    st.markdown(f"""
    <div id="auth-section"></div>
    <div class="auth-container">
        <div class="auth-title">{title_text}</div>
        <div class="auth-subtitle">{subtitle_text}</div>
    </div>
    """, unsafe_allow_html=True)

    # Centered grid container preserving original column proportions
    _, center_col, _ = st.columns([0.15, 0.7, 0.15])

    with center_col:
        col_left, col_mid, col_right = st.columns([1.1, 0.2, 1.2], gap="large")

        # LEFT COLUMN: Social / Developer Single-Click Logins
        with col_left:
            st.markdown('<div style="height: 12px;"></div>', unsafe_allow_html=True)

            # GitHub Login
            if st.button("🐱 Log in with GitHub", key="btn_gh", use_container_width=True):
                st.session_state.is_authenticated = True
                st.session_state.is_guest = False
                st.session_state.user_email = "student_github@university.edu"
                st.session_state.user_name = "GitHub Student"
                st.rerun()

            st.markdown('<div style="height: 6px;"></div>', unsafe_allow_html=True)

            # Facebook Login
            if st.button("📘 Log in with Facebook", key="btn_fb", use_container_width=True):
                st.session_state.is_authenticated = True
                st.session_state.is_guest = False
                st.session_state.user_email = "student_fb@university.edu"
                st.session_state.user_name = "Facebook Student"
                st.rerun()

            st.markdown('<div style="height: 6px;"></div>', unsafe_allow_html=True)

            # Google Login
            if st.button("🌐 Log in with Google", key="btn_google", use_container_width=True):
                st.session_state.is_authenticated = True
                st.session_state.is_guest = False
                st.session_state.user_email = "student_google@university.edu"
                st.session_state.user_name = "Google Student"
                st.rerun()

            st.markdown('<div style="height: 6px;"></div>', unsafe_allow_html=True)

            # LinkedIn Login
            if st.button("💼 Log in with LinkedIn", key="btn_linkedin", use_container_width=True):
                st.session_state.is_authenticated = True
                st.session_state.is_guest = False
                st.session_state.user_email = "student_linkedin@university.edu"
                st.session_state.user_name = "LinkedIn Student"
                st.rerun()

            st.markdown("""
            <div class="social-subtext">
                We'll never post without your permission.
            </div>
            """, unsafe_allow_html=True)

        # MIDDLE COLUMN: OR Divider
        with col_mid:
            st.markdown("""
            <div class="or-divider">
                <div class="or-line"></div>
                <div>OR</div>
                <div class="or-line"></div>
            </div>
            """, unsafe_allow_html=True)

        # RIGHT COLUMN: Email & Password Form
        with col_right:
            if not is_login:
                st.markdown('<div class="auth-label">Full Name</div>', unsafe_allow_html=True)
                st.text_input("Name", placeholder="e.g. Mahima Dhaka", label_visibility="collapsed", key="reg_name")

            st.markdown('<div class="auth-label">Email</div>', unsafe_allow_html=True)
            email_input = st.text_input("Email", placeholder="Email", label_visibility="collapsed", key="auth_email_val")

            st.markdown('<div class="auth-label">Password</div>', unsafe_allow_html=True)
            password_input = st.text_input(
                "Password",
                type="password",
                placeholder="Password",
                label_visibility="collapsed",
                key="auth_pass_val"
            )

            if is_login:
                with st.expander("Forgot your password?"):
                    st.caption("Demo Mode: In this academic ML project, password reset emails are simulated. You can use any test credentials or the one-click demo student login below.")

            # Yellow Submit Button
            button_label = "✉️ Log in with email" if is_login else "✉️ Sign up with email"
            if st.button(button_label, type="primary", use_container_width=True, key="btn_submit_auth"):
                if email_input.strip() and password_input.strip():
                    st.session_state.is_authenticated = True
                    st.session_state.is_guest = False
                    st.session_state.user_email = email_input.strip()
                    if not is_login and "reg_name" in st.session_state and st.session_state.reg_name:
                        st.session_state.user_name = st.session_state.reg_name.strip()
                    else:
                        st.session_state.user_name = email_input.split("@")[0].capitalize()
                    st.rerun()
                else:
                    st.error("Please enter both email and password.")

        # Bottom Toggle between Login and Signup
        st.markdown("<br>", unsafe_allow_html=True)
        toggle_col1, toggle_col2, toggle_col3 = st.columns([1, 2, 1])
        with toggle_col2:
            if is_login:
                st.markdown('<div style="text-align: center; color: #cbd5e1; font-size: 0.95rem; margin-top: 14px;">New to the platform?</div>', unsafe_allow_html=True)
                if st.button("👉 Sign up for an account", use_container_width=True, key="btn_goto_signup"):
                    st.session_state.auth_mode = "signup"
                    st.rerun()
            else:
                st.markdown('<div style="text-align: center; color: #cbd5e1; font-size: 0.95rem; margin-top: 14px;">Already have an account?</div>', unsafe_allow_html=True)
                if st.button("👉 Log in to your account", use_container_width=True, key="btn_goto_login"):
                    st.session_state.auth_mode = "login"
                    st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)
            # Instant Viva Demo Login button
            if st.button("🚀 Quick Demo Student Login (One-Click)", use_container_width=True, key="btn_demo_viva"):
                st.session_state.is_authenticated = True
                st.session_state.is_guest = False
                st.session_state.user_email = "mahima.student@university.edu"
                st.session_state.user_name = "Mahima Dhaka"
                st.rerun()
