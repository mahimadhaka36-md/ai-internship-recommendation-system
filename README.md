# AI Internship Recommendation System

A clean, explainable, and interactive Machine Learning web application built with **Python** and **Streamlit**. The platform lets students browse technical internships, upload resumes (PDF or DOCX) for personalized **TF-IDF + Cosine Similarity** internship matching, use **Explore Profiles** (powered by **K-Nearest Neighbors / KNN**) to discover similar student profiles and career paths, and practice in the **AI Mock Interview** module with semantic ML answer evaluation.

---

## 🎯 Features & Application Flow

1. **Welcome & Authentication Gate (`auth.py`):**
   - Two-section hero landing page with guest catalog access, sign-up/login flow, and one-click demo student login.
2. **Explore Internships Catalog (`catalog_ui.py`):**
   - Search, filter (location, required skills, minimum stipend), sort, and paginate across all 388 listings in `data/technical_internships_ready.csv`.
3. **3-Step Personalized Internship Pipeline (`app.py`, `resume_parser.py`, `skill_extractor.py`, `recommendation.py`):**
   - **Step 1 — Resume Upload:** In-memory PDF (`PyMuPDF`) and DOCX (`python-docx`) text extraction with scanned-PDF detection and zero disk persistence.
   - **Step 2 — Review & Confirm Skills:** Interactive skill review with regex word-boundary extraction, alias normalization, custom skill input, quick-add pills, and a **Skill Coverage Overview** chart.
   - **Step 3 — Recommendations & ML Analysis:** Top-5 ranked internship cards with Cosine Similarity scores, matched vs. missing skills, 4 explainable Matplotlib visualizations (`charts.py`), and honest functional validation checks.
4. **Explore Profiles — KNN Peer & Career Matching (`peer_match.py`, `peer_match_ui.py`):**
   - **My Academic Profile:** Structured form for major, university year, normalized CGPA (`10.0` or `4.0` scale), technical skills, proficiency ratings (`1–10`), projects, certifications, and internships (plus sample presets and resume skill import).
   - **Recommended Peer Profiles:** Top-$K$ nearest student profiles from `data/synthetic_student_profiles_30.csv` using standardized numeric features, One-Hot encoded major, and exact-token TF-IDF technical skill vectors.
   - **Student Profile Comparison:** Metric deltas, competency radar chart (`0–10` scale), and practical experience comparison bar chart.
   - **Career Recommendations:** Distance-weighted KNN neighbor consensus recommending target career field, technical domain, and internship role.
   - **Recommended Internships:** Bridges KNN profile skills directly into the 388-listing TF-IDF + Cosine Similarity engine.
   - **Personalized Skill Improvement & KNN Model Insights:** Missing-skill prioritization (peer prevalence + internship listing demand), benchmark comparisons, neighbor $K$-cutoff chart, and automated KNN validation checks.
5. **AI Mock Interview (`mock_interview.py`, `interview_evaluator.py`, `mock_interview_ui.py`):**
   - **Interview Setup:** Choose target role (10 supported roles), technical domain, interview type (`Technical`, `HR`, `Behavioral`, `Mixed`), difficulty (`Beginner`, `Intermediate`, `Advanced`), question count (`5`, `10`, `15`), and timed or untimed mode, with automatic role suggestions from confirmed skills and internship recommendations (plus direct launch from recommended internships).
   - **Structured Question Bank:** 170 non-repeating curated questions in `data/interview_questions.json` with reference answers, expected concepts, and question-specific rubrics.
   - **Optional Covexcode Generative AI (`ai_provider.py`):** When `COVEXCODE_API_KEY` is configured via environment variables or `.streamlit/secrets.toml`, Covexcode (`https://llm.covexcode.com/v1`, `deepseek-v4-flash` / `deepseek-v4-pro`) provides personalized interview questions, contextual follow-up questions, and detailed answer feedback—with automatic fallback to the 170-question bank and local evaluator if credits expire or offline.
   - **ML Answer Evaluation:** Lazy-loaded, resource-cached `SentenceTransformer("all-MiniLM-L6-v2")` semantic comparison paired with deterministic concept coverage, technical correctness alignment, completeness, and clarity—with automatic TF-IDF + concept-matching fallback when the transformer model is unavailable.
   - **Final Dashboard & Report:** Question-wise, rubric-dimension, and topic-level Matplotlib visualizations, strongest/weakest topics, preparation suggestions, and downloadable Markdown performance report.

---

## 📊 ML Methodology

### 1. Internship Recommendation Engine (TF-IDF + Cosine Similarity)
- **Unsupervised Content-Based Filtering:** Internship `title + skills_clean` records and confirmed student skills are projected into a unigram + bigram `TfidfVectorizer` space (`stop_words="english"`).
- **Ranking Metric:**
$$\text{Cosine Similarity}(\vec{A}, \vec{B}) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|}$$

### 2. Peer & Career Matching Engine (K-Nearest Neighbors)
- **Feature Preprocessing:**
  - Technical skills encoded with `TfidfVectorizer` over canonicalized skill tokens.
  - Academic & proficiency attributes (`University_Year`, normalized `CGPA/10`, `Programming_Skill`, `Problem_Solving`, `Communication_Skills`, `Projects_Completed`, `Certifications`, `Internships`) scaled with `StandardScaler`.
  - Academic `Major` encoded with `OneHotEncoder(handle_unknown="ignore")`.
- **Similarity Retrieval:** `sklearn.neighbors.NearestNeighbors` retrieves top-$K$ peers and aggregates career paths via similarity-weighted voting ($\frac{1}{1 + d}$ for Euclidean distance).

### 3. Mock Interview Semantic Evaluation (`all-MiniLM-L6-v2` + TF-IDF Fallback)
- **Transparent Rubric (0–100):** Semantic Relevance (`35%`), Expected Concept Coverage (`30%`), Technical Correctness alignment (`15%`), Completeness (`10%`), and Answer Clarity (`10%`).
- **Honest Framing:** Semantic similarity measures alignment with reference phrasing and does not by itself verify factual correctness; practice scores are for self-study and are not hiring probabilities.

---

## 📁 Project Structure

```text
ai-internship-recommendation-system/
├── app.py                              # Main Streamlit app & multi-page routing
├── auth.py                             # Landing hero & authentication UI
├── catalog_ui.py                       # Explore Internships catalog UI
├── peer_match.py                       # KNN student profile & career matching engine
├── peer_match_ui.py                    # Explore Profiles UI & insights
├── mock_interview.py                   # Mock interview setup, role suggestions & reporting
├── interview_evaluator.py              # Lazy all-MiniLM-L6-v2 evaluator + TF-IDF fallback
├── ai_provider.py                      # Centralized Covexcode OpenAI-compatible AI provider
├── mock_interview_ui.py                # AI Mock Interview setup, active loop & dashboard UI
├── recommendation.py                   # TF-IDF & Cosine Similarity recommendation engine
├── skill_extractor.py                  # Rule-based skill extraction & alias normalization
├── resume_parser.py                    # In-memory PDF & DOCX text extraction
├── charts.py                           # Matplotlib visualizations for internships, peers & interviews
├── prepare_model.py                    # Reproducible TF-IDF artifact builder
├── test_app.py                         # Automated test suite (22 unit & integration tests)
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
├── data/
│   ├── README.md
│   ├── technical_internships_ready.csv     # 388 internship listings
│   ├── synthetic_student_profiles_30.csv   # 30 synthetic demo student profiles
│   └── interview_questions.json            # 170 structured interview questions
└── model_artifacts/
    ├── README.md
    ├── tfidf_vectorizer.joblib
    ├── tfidf_matrix.joblib
    └── internships_cleaned.joblib
```

---

## 🚀 Setup & Local Execution

### Prerequisites
- Python 3.10+

### 1. Clone the Repository
```bash
git clone https://github.com/mahimadhaka36-md/ai-internship-recommendation-system.git
cd ai-internship-recommendation-system
```

### 2. Create & Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Rebuild TF-IDF Model Artifacts
Pre-built artifacts are included in `model_artifacts/`, and are also generated automatically if missing:
```bash
python prepare_model.py
```

### 5. Launch the Streamlit App
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## ☁️ Streamlit Community Cloud Deployment

1. Push this repository to GitHub (`main` branch).
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
3. Select repository `mahimadhaka36-md/ai-internship-recommendation-system`, branch `main`, and main file path `app.py`.
4. Click **Deploy** — dependencies are installed automatically from `requirements.txt`, and all data/model paths resolve relative to the repository root.

---

## 🧪 Automated Testing

Run the automated test suite covering internship recommendations, parsers, catalogs, KNN peer matching, and visualizations:
```bash
python -m unittest test_app.py -v
```

---

## 🎓 Honest Academic Evaluation & Limitations

- Neither `technical_internships_ready.csv` nor `synthetic_student_profiles_30.csv` contains real-world hiring or placement outcomes.
- Accordingly, supervised hiring accuracy or ROC curves are not fabricated; validation relies on deterministic functional checks (ranking monotonicity, edge-case handling, scale normalization, and internal Leave-One-Out consistency on the 30 synthetic profiles).
