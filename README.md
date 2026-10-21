# AI Internship Recommendation System

A clean, explainable, and professional web application for an academic second-year Machine Learning project. The system allows students to upload their resumes (PDF or DOCX), automatically extracts and lets them customize their technical skills, and provides personalized, ranked internship recommendations using **TF-IDF Vectorization** and **Cosine Similarity**.

---

## 🎯 Architecture & Three-Page Flow

The application follows an intuitive, sequential three-page workflow:

```
┌──────────────────────┐       ┌──────────────────────┐       ┌──────────────────────┐       ┌──────────────────────┐
│  LOGIN / SIGNUP GATE │  ──▶  │ PAGE 1: Resume Upload│  ──▶  │ PAGE 2: Confirm Skills│  ──▶  │ PAGE 3: ML Dashboard │
│  • Devpost aesthetic │       │ • PDF / DOCX in-mem  │       │ • Multiselect review │       │ • Top 5 match cards  │
│  • GitHub/Google/In  │       │ • Size & type checks │       │ • Custom skill entry │       │ • Graphs A, B, C, D  │
│  • Email & Password  │       │ • Scanned PDF alert  │       │ • Skill Coverage Plot│       │ • Honest Evaluation  │
└──────────────────────┘       └──────────────────────┘       └──────────────────────┘       └──────────────────────┘
```

---

## 📊 ML Visualization Dashboard (Page 3)

1. **Graph A — Top 5 Internship Match Scores:**
   - Horizontal bar chart of actual Cosine Similarity scores ($0.0$ to $1.0$).
   - Shows numerical score and percentage match beside each bar.
   - Explains that cosine similarity represents textual alignment, not hiring probability.

2. **Graph B — Skill Match Breakdown:**
   - Horizontal grouped bar chart comparing confirmed student skills (profile-level reference) against the count of overlapping skills required by each internship.

3. **Graph C — Most In-Demand Skills in the Dataset:**
   - Frequency distribution of the top 10–15 recognized skills across all 388 listings in `technical_internships_ready.csv` (e.g. Python: 162, JavaScript: 123, HTML: 91, CSS: 86, MySQL: 75, Machine Learning: 59).
   - Handles multi-word skills like "machine learning" and "data science" as complete terms.

4. **Graph D — Internship Similarity Heatmap (Expandable):**
   - 2D matrix of cosine similarity values between the overall student profile + individual query skills against each recommended listing's feature vector.

---

## 📁 Project Structure

```text
ml project/
├── app.py                      # Main Streamlit web application (Three Pages)
├── recommendation.py           # TF-IDF & Cosine Similarity recommendation pipeline
├── skill_extractor.py          # Rule-based skill extraction & canonical normalization
├── resume_parser.py            # In-memory PDF & DOCX text extraction
├── charts.py                   # Matplotlib visualization functions for ML Dashboard
├── prepare_model.py            # Reproducible model artifact generation script
├── test_app.py                 # Automated test suite (12 unit & integration tests)
├── requirements.txt            # Python dependencies
├── README.md                   # Setup and execution guide
├── data/
│   └── technical_internships_ready.csv  # 388 internship records
└── model_artifacts/            # Saved joblib artifacts (reproducible)
    ├── tfidf_vectorizer.joblib
    ├── tfidf_matrix.joblib
    └── internships_cleaned.joblib
```

---

## 🚀 Setup & Execution Instructions

### Prerequisites
- Python 3.10 to 3.14
- Recommended: Virtual environment (`venv` or `conda`)

### 1. Navigate to Project Directory
```bash
cd "/Users/mahimadhaka/ml project"
```

### 2. Activate Virtual Environment
```bash
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Generate Model Artifacts
Artifacts are automatically built on the first run, but can also be created explicitly:
```bash
python prepare_model.py
```

### 5. Launch Web Application
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 🧪 Automated Testing

Run the automated test suite covering dataset integrity, parsers, ranking, and new visualization helpers:
```bash
python test_app.py
```

---

## 🎓 College Viva Guide: Honest Evaluation & Methodology

### Q1: Is this a supervised or unsupervised model?
**Answer:** It is an **unsupervised content-based recommendation system**. We do not have pre-labeled student-internship suitability classes or hiring outcomes; instead, both student skills and internship descriptions are projected into a continuous TF-IDF vector space, and ranked using Cosine Similarity.

### Q2: Why don't you report Precision, Recall, or an ROC Curve?
**Answer:** Because the dataset does not contain ground-truth labeled outcomes (e.g. whether a student was accepted or rejected for each listing). Fabricating an accuracy or F1-score for unsupervised ranking is academically dishonest. We instead validate the system through deterministic functional checks (rank ordering, constraint limits, and edge case handling).

### Q3: What is the mathematical formulation of Cosine Similarity?
**Answer:**
$$\text{Cosine Similarity}(\vec{A}, \vec{B}) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$
It evaluates the angle between vectors rather than magnitude, ensuring long descriptions do not unfairly dominate shorter, focused profiles.
