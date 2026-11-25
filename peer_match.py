"""
peer_match.py
-------------
AI Peer Match — K-Nearest Neighbors (KNN) Student Profile Recommendation Engine.

Methodology Summary:
1. Loads the 30 synthetic student profiles from data/synthetic_student_profiles_30.csv.
2. Normalizes academic & proficiency features:
   - CGPA normalized to a unified 10-point scale: (CGPA / CGPA_Scale) * 10
   - University Year, Programming, Problem Solving, Communication, Projects, Certifications, Internships
   - Standardized using scikit-learn StandardScaler and block-scaled by feature group size
3. Encodes Technical Skills as exact normalized skill tokens via TfidfVectorizer
   (preserving multi-word and special-character skills like 'c++', 'node.js', 'scikit-learn', 'machine learning').
4. Encodes Academic Major via OneHotEncoder(handle_unknown='ignore').
5. Uses scikit-learn NearestNeighbors (Euclidean or Cosine distance) to retrieve the K most similar peers.
6. Derives distance-weighted Career Recommendations, Skill Gap Analysis, and Profile Improvement Insights.
"""

import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from skill_extractor import normalize_skill_name

STUDENT_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "data", "synthetic_student_profiles_30.csv"
)

NUMERIC_FEATURES = [
    "university_year",
    "cgpa_10",
    "programming_skill",
    "problem_solving",
    "communication_skills",
    "projects_completed",
    "certifications",
    "internships",
]

# Feature-block weights: (skills_weight, numeric_weight, major_weight)
WEIGHT_PRESETS: Dict[str, Tuple[float, float, float]] = {
    "Balanced (Skills + Academics)": (0.60, 0.35, 0.05),
    "Skills-Focused": (0.75, 0.20, 0.05),
    "Academics & Experience-Focused": (0.35, 0.60, 0.05),
}

# Small extra aliases specific to student CSV labels vs internship vocabulary
_PEER_SKILL_ALIASES = {
    "ml": "machine learning",
    "dl": "deep learning",
    "js": "javascript",
    "ts": "typescript",
    "postgres": "postgresql",
    "rest apis": "rest api",
}


def canonicalize_peer_skill(skill: str) -> str:
    """Normalize a technical skill string consistently across student and internship datasets."""
    norm = normalize_skill_name(str(skill))
    return _PEER_SKILL_ALIASES.get(norm, norm)


def parse_skills_input(skills_input: Union[str, List[str]]) -> List[str]:
    """Parse comma-separated string or list of skills into deduplicated canonical skill tokens."""
    if isinstance(skills_input, list):
        raw_items = skills_input
    elif isinstance(skills_input, str):
        raw_items = skills_input.split(",")
    else:
        raw_items = []

    seen = set()
    cleaned: List[str] = []
    for item in raw_items:
        token = canonicalize_peer_skill(item)
        if token and token not in seen:
            seen.add(token)
            cleaned.append(token)
    return cleaned


def _identity_tokenizer(tokens: List[str]) -> List[str]:
    """Pass pre-tokenized canonical skill lists directly into TfidfVectorizer."""
    return tokens


class PeerMatchEngine:
    """
    Encapsulates the KNN student profile recommendation pipeline using
    synthetic_student_profiles_30.csv.
    """

    def __init__(self, csv_path: str = STUDENT_DATA_PATH):
        self.csv_path = csv_path
        self.students_df: pd.DataFrame = pd.DataFrame()
        self.skill_vectorizer: Optional[TfidfVectorizer] = None
        self.numeric_scaler: Optional[StandardScaler] = None
        self.major_encoder: Optional[OneHotEncoder] = None
        self.student_skills_matrix: Optional[np.ndarray] = None
        self.student_numeric_matrix: Optional[np.ndarray] = None
        self.student_major_matrix: Optional[np.ndarray] = None
        self.all_dataset_skills: List[str] = []
        self.all_majors: List[str] = []
        self._load_and_fit()

    def _load_and_fit(self) -> None:
        """Load synthetic student CSV and fit preprocessing transformers."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"Student dataset not found at '{self.csv_path}'. "
                "Please ensure synthetic_student_profiles_30.csv is in the data/ directory."
            )

        df = pd.read_csv(self.csv_path)
        df.columns = [c.strip() for c in df.columns]

        # Ensure string columns have safe defaults
        str_cols = [
            "Student_ID",
            "Student_Name",
            "Major",
            "Technical_Skills",
            "Career_Field",
            "Technical_Domain",
            "Internship_Role",
            "LinkedIn_Profile",
            "LinkedIn_Status",
            "Data_Source",
            "Profile_Consent",
        ]
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str).str.strip()
            else:
                df[col] = ""

        # Numeric parsing and unified 10-point CGPA normalization
        df["University_Year"] = pd.to_numeric(df["University_Year"], errors="coerce").fillna(1).astype(int)
        df["CGPA"] = pd.to_numeric(df["CGPA"], errors="coerce").fillna(0.0)
        df["CGPA_Scale"] = pd.to_numeric(df["CGPA_Scale"], errors="coerce").fillna(10.0)
        df["CGPA_Scale"] = df["CGPA_Scale"].replace(0.0, 10.0)
        df["cgpa_10"] = ((df["CGPA"] / df["CGPA_Scale"]) * 10.0).clip(0.0, 10.0).round(2)

        df["university_year"] = df["University_Year"].astype(float)
        df["programming_skill"] = pd.to_numeric(df["Programming_Skill"], errors="coerce").fillna(5.0)
        df["problem_solving"] = pd.to_numeric(df["Problem_Solving"], errors="coerce").fillna(5.0)
        df["communication_skills"] = pd.to_numeric(df["Communication_Skills"], errors="coerce").fillna(5.0)
        df["projects_completed"] = pd.to_numeric(df["Projects_Completed"], errors="coerce").fillna(0.0)
        df["certifications"] = pd.to_numeric(df["Certifications"], errors="coerce").fillna(0.0)
        df["internships"] = pd.to_numeric(df["Internships"], errors="coerce").fillna(0.0)

        # Parse canonical skills list for each student
        df["skills_list"] = df["Technical_Skills"].apply(parse_skills_input)

        self.students_df = df.reset_index(drop=True)

        # Collect sorted unique dataset skills and majors
        skill_set = set()
        for skills in self.students_df["skills_list"]:
            skill_set.update(skills)
        self.all_dataset_skills = sorted(skill_set)
        self.all_majors = sorted(
            [m for m in self.students_df["Major"].unique() if str(m).strip()]
        )

        # Fit skill TF-IDF vectorizer on exact canonical skill tokens
        self.skill_vectorizer = TfidfVectorizer(
            tokenizer=_identity_tokenizer,
            preprocessor=None,
            lowercase=False,
            token_pattern=None,
        )
        self.student_skills_matrix = (
            self.skill_vectorizer.fit_transform(self.students_df["skills_list"]).toarray()
        )

        # Fit StandardScaler on numeric profile features
        self.numeric_scaler = StandardScaler()
        raw_numeric = self.students_df[NUMERIC_FEATURES].to_numpy(dtype=float)
        # Divide by sqrt(d) so expected squared L2 norm of the standardized block is ~1.0
        self.student_numeric_matrix = self.numeric_scaler.fit_transform(raw_numeric) / np.sqrt(
            len(NUMERIC_FEATURES)
        )

        # Fit OneHotEncoder on Academic Major
        self.major_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.student_major_matrix = self.major_encoder.fit_transform(
            self.students_df[["Major"]]
        )

    def _assemble_weighted_matrix(
        self,
        skills_mat: np.ndarray,
        numeric_mat: np.ndarray,
        major_mat: np.ndarray,
        weight_preset: str,
    ) -> np.ndarray:
        """Combine feature blocks using sqrt(weight) scaling for balanced distance geometry."""
        w_skills, w_num, w_major = WEIGHT_PRESETS.get(
            weight_preset, WEIGHT_PRESETS["Balanced (Skills + Academics)"]
        )
        return np.hstack([
            skills_mat * np.sqrt(w_skills),
            numeric_mat * np.sqrt(w_num),
            major_mat * np.sqrt(w_major),
        ])

    def _transform_query(
        self, student_profile: Dict[str, Any], weight_preset: str
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Transform a user profile dictionary into the weighted KNN feature vector."""
        name = str(student_profile.get("student_name", "You") or "You").strip()
        major = str(student_profile.get("major", "Computer Science") or "Computer Science").strip()
        year = int(student_profile.get("university_year", 2))
        cgpa_raw = float(student_profile.get("cgpa", 8.0))
        cgpa_scale = float(student_profile.get("cgpa_scale", 10.0)) or 10.0
        cgpa_10 = round(float(np.clip((cgpa_raw / cgpa_scale) * 10.0, 0.0, 10.0)), 2)

        programming = float(student_profile.get("programming_skill", 7))
        problem_solving = float(student_profile.get("problem_solving", 7))
        communication = float(student_profile.get("communication_skills", 7))
        projects = float(student_profile.get("projects_completed", 2))
        certifications = float(student_profile.get("certifications", 1))
        internships = float(student_profile.get("internships", 0))
        skills_list = parse_skills_input(student_profile.get("technical_skills", []))

        skills_vec = self.skill_vectorizer.transform([skills_list]).toarray()
        numeric_raw = np.array(
            [[year, cgpa_10, programming, problem_solving, communication, projects, certifications, internships]],
            dtype=float,
        )
        numeric_vec = self.numeric_scaler.transform(numeric_raw) / np.sqrt(len(NUMERIC_FEATURES))
        major_df = pd.DataFrame({"Major": [major]})
        major_vec = self.major_encoder.transform(major_df)

        combined = self._assemble_weighted_matrix(skills_vec, numeric_vec, major_vec, weight_preset)

        normalized_profile = {
            "student_name": name,
            "major": major,
            "university_year": year,
            "cgpa": cgpa_raw,
            "cgpa_scale": cgpa_scale,
            "cgpa_10": cgpa_10,
            "technical_skills": skills_list,
            "programming_skill": programming,
            "problem_solving": problem_solving,
            "communication_skills": communication,
            "projects_completed": projects,
            "certifications": certifications,
            "internships": internships,
        }
        return combined, normalized_profile

    def get_all_students(self) -> pd.DataFrame:
        """Return a copy of the loaded synthetic student profiles dataset."""
        return self.students_df.copy()

    def get_available_skills(self) -> List[str]:
        """Return sorted skills present in the student dataset."""
        return list(self.all_dataset_skills)

    def get_dataset_summary(self) -> Dict[str, Any]:
        """Return descriptive statistics for the 30 synthetic student profiles."""
        df = self.students_df
        return {
            "total_students": len(df),
            "career_fields_count": int(df["Career_Field"].nunique()),
            "career_field_distribution": df["Career_Field"].value_counts().to_dict(),
            "technical_domains": df["Technical_Domain"].value_counts().to_dict(),
            "avg_cgpa_10": round(float(df["cgpa_10"].mean()), 2),
            "avg_projects": round(float(df["projects_completed"].mean()), 2),
            "avg_certifications": round(float(df["certifications"].mean()), 2),
            "with_internship_count": int((df["internships"] > 0).sum()),
            "unique_skills_count": len(self.all_dataset_skills),
        }

    def find_similar_students(
        self,
        student_profile: Dict[str, Any],
        k: int = 5,
        metric: str = "euclidean",
        weight_preset: str = "Balanced (Skills + Academics)",
        dataset_skill_freqs: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        Execute KNN search against all 30 synthetic student profiles and return
        matched peers, profile averages, career recommendations, skill gaps, and insights.
        """
        total_students = len(self.students_df)
        k_clamped = max(1, min(int(k), total_students))
        metric_clean = "cosine" if str(metric).lower().startswith("cos") else "euclidean"

        dataset_matrix = self._assemble_weighted_matrix(
            self.student_skills_matrix,
            self.student_numeric_matrix,
            self.student_major_matrix,
            weight_preset,
        )
        query_vector, norm_profile = self._transform_query(student_profile, weight_preset)

        nn = NearestNeighbors(n_neighbors=total_students, metric=metric_clean)
        nn.fit(dataset_matrix)
        distances, indices = nn.kneighbors(query_vector, n_neighbors=total_students)

        dists = distances[0]
        idxs = indices[0]
        user_skill_set = set(norm_profile["technical_skills"])

        all_ranked: List[Dict[str, Any]] = []
        for rank, (row_idx, dist) in enumerate(zip(idxs, dists), start=1):
            row = self.students_df.iloc[int(row_idx)]
            dist_val = float(dist)
            if metric_clean == "euclidean":
                sim_score = 1.0 / (1.0 + dist_val)
            else:
                sim_score = float(np.clip(1.0 - dist_val, 0.0, 1.0))

            peer_skills = list(row["skills_list"])
            peer_skill_set = set(peer_skills)
            matched_skills = sorted(list(user_skill_set.intersection(peer_skill_set)))
            additional_skills = sorted(list(peer_skill_set.difference(user_skill_set)))

            all_ranked.append({
                "rank": rank,
                "student_id": row["Student_ID"],
                "student_name": row["Student_Name"],
                "major": row["Major"],
                "university_year": int(row["University_Year"]),
                "cgpa": float(row["CGPA"]),
                "cgpa_scale": float(row["CGPA_Scale"]),
                "cgpa_10": float(row["cgpa_10"]),
                "technical_skills_raw": row["Technical_Skills"],
                "technical_skills": peer_skills,
                "programming_skill": int(row["Programming_Skill"]),
                "problem_solving": int(row["Problem_Solving"]),
                "communication_skills": int(row["Communication_Skills"]),
                "projects_completed": int(row["Projects_Completed"]),
                "certifications": int(row["Certifications"]),
                "internships": int(row["Internships"]),
                "career_field": row["Career_Field"],
                "technical_domain": row["Technical_Domain"],
                "internship_role": row["Internship_Role"] or "No prior internship",
                "has_internship": int(row["Internships"]) > 0,
                "linkedin_profile": row["LinkedIn_Profile"],
                "linkedin_status": row["LinkedIn_Status"],
                "data_source": row["Data_Source"],
                "profile_consent": row["Profile_Consent"],
                "distance": round(dist_val, 4),
                "similarity_score": round(sim_score, 4),
                "similarity_pct": round(sim_score * 100.0, 1),
                "matched_skills": matched_skills,
                "additional_skills": additional_skills,
                "is_top_k": rank <= k_clamped,
            })

        top_peers = all_ranked[:k_clamped]

        # Peer averages
        peer_average = self._compute_peer_average(top_peers)
        intern_peers = [p for p in top_peers if p["has_internship"]]
        intern_peer_average = self._compute_peer_average(intern_peers) if intern_peers else peer_average

        # Distance-weighted Career Recommendations
        career_recommendations = self._compute_career_recommendations(top_peers)

        # Skill Gap Analysis
        skill_gaps = self._compute_skill_gaps(top_peers, user_skill_set, dataset_skill_freqs or {})

        # Numeric & Experience Improvement Insights
        profile_improvements = self._compute_profile_improvements(
            norm_profile, peer_average, intern_peer_average, len(intern_peers)
        )

        # Recommended combined query skills for bridging to Internship Recommendation Engine
        top_career_skills = [g["skill"] for g in skill_gaps[:3]]
        in_vocab_skills = [
            s for s in norm_profile["technical_skills"]
            if s in set(self.skill_vectorizer.vocabulary_.keys())
        ]

        return {
            "status": "success",
            "k": k_clamped,
            "metric": metric_clean,
            "weight_preset": weight_preset,
            "weights": WEIGHT_PRESETS.get(weight_preset, WEIGHT_PRESETS["Balanced (Skills + Academics)"]),
            "user_profile": norm_profile,
            "peers": top_peers,
            "all_ranked": all_ranked,
            "peer_average": peer_average,
            "intern_peer_average": intern_peer_average,
            "intern_peer_count": len(intern_peers),
            "career_recommendations": career_recommendations,
            "skill_gaps": skill_gaps,
            "profile_improvements": profile_improvements,
            "top_suggested_skills": top_career_skills,
            "recognized_peer_vocab_skills": in_vocab_skills,
            "empty_skills_input": len(norm_profile["technical_skills"]) == 0,
        }

    def _compute_peer_average(self, peers: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute average numeric attributes across a list of peer dicts."""
        if not peers:
            return {
                "university_year": 0.0,
                "cgpa_10": 0.0,
                "programming_skill": 0.0,
                "problem_solving": 0.0,
                "communication_skills": 0.0,
                "projects_completed": 0.0,
                "certifications": 0.0,
                "internships": 0.0,
            }
        keys = [
            "university_year",
            "cgpa_10",
            "programming_skill",
            "problem_solving",
            "communication_skills",
            "projects_completed",
            "certifications",
            "internships",
        ]
        return {
            key: round(float(np.mean([p[key] for p in peers])), 2)
            for key in keys
        }

    def _compute_career_recommendations(self, top_peers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank career fields via similarity-weighted voting across the K nearest peers."""
        if not top_peers:
            return []

        weight_sums: Dict[str, float] = defaultdict(float)
        counts: Counter = Counter()
        domains: Dict[str, Counter] = defaultdict(Counter)
        roles: Dict[str, Counter] = defaultdict(Counter)
        sim_lists: Dict[str, List[float]] = defaultdict(list)

        total_weight = sum(max(p["similarity_score"], 1e-6) for p in top_peers)
        for p in top_peers:
            cf = p["career_field"]
            w = max(p["similarity_score"], 1e-6)
            weight_sums[cf] += w
            counts[cf] += 1
            domains[cf][p["technical_domain"]] += 1
            if p["internship_role"] and p["internship_role"] != "No prior internship":
                roles[cf][p["internship_role"]] += 1
            sim_lists[cf].append(p["similarity_pct"])

        recs: List[Dict[str, Any]] = []
        for cf, w_sum in sorted(weight_sums.items(), key=lambda x: (x[1], counts[x[0]]), reverse=True):
            top_domain = domains[cf].most_common(1)[0][0] if domains[cf] else "Technology"
            top_role = roles[cf].most_common(1)[0][0] if roles[cf] else f"{cf} Intern"
            recs.append({
                "career_field": cf,
                "technical_domain": top_domain,
                "recommended_internship_role": top_role,
                "peer_count": counts[cf],
                "total_k": len(top_peers),
                "weighted_share_pct": round((w_sum / total_weight) * 100.0, 1) if total_weight > 0 else 0.0,
                "avg_peer_similarity_pct": round(float(np.mean(sim_lists[cf])), 1),
            })
        return recs

    def _compute_skill_gaps(
        self,
        top_peers: List[Dict[str, Any]],
        user_skill_set: set,
        dataset_skill_freqs: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        """Identify skills present in nearest peers that the user does not currently list."""
        peer_skill_counts: Counter = Counter()
        weighted_peer_score: Dict[str, float] = defaultdict(float)
        for p in top_peers:
            for s in p["technical_skills"]:
                if s not in user_skill_set:
                    peer_skill_counts[s] += 1
                    weighted_peer_score[s] += p["similarity_score"]

        gaps: List[Dict[str, Any]] = []
        for skill, count in peer_skill_counts.items():
            internship_demand = int(dataset_skill_freqs.get(skill, 0))
            gaps.append({
                "skill": skill,
                "peer_count": count,
                "total_k": len(top_peers),
                "peer_prevalence_pct": round((count / len(top_peers)) * 100.0, 1) if top_peers else 0.0,
                "internship_dataset_listings": internship_demand,
                "priority_score": round(weighted_peer_score[skill] + 0.002 * internship_demand, 3),
            })

        gaps.sort(
            key=lambda g: (g["peer_count"], g["internship_dataset_listings"], g["priority_score"]),
            reverse=True,
        )
        return gaps

    def _compute_profile_improvements(
        self,
        user_profile: Dict[str, Any],
        peer_avg: Dict[str, float],
        intern_peer_avg: Dict[str, float],
        intern_peer_count: int,
    ) -> List[Dict[str, Any]]:
        """Generate grounded comparison feedback on numeric profile dimensions."""
        ref_avg = intern_peer_avg if intern_peer_count > 0 else peer_avg
        ref_label = (
            f"Internship-Experienced Peers (n={intern_peer_count})"
            if intern_peer_count > 0
            else "Matched K-Peer Average"
        )

        dimensions = [
            ("programming_skill", "Programming Proficiency", "/10", 1.0, "Practice algorithmic coding and language-specific projects."),
            ("problem_solving", "Problem-Solving Ability", "/10", 1.0, "Solve structured DSA / analytical case problems regularly."),
            ("communication_skills", "Communication Skills", "/10", 1.0, "Document project READMEs, present demos, and practice mock interviews."),
            ("projects_completed", "Completed Projects", " projects", 1.0, "Build 1–2 end-to-end portfolio projects aligned with your target domain."),
            ("certifications", "Certifications", " certs", 1.0, "Complete a focused, verifiable certification in your target track."),
            ("cgpa_10", "Normalized CGPA", "/10", 0.5, "Maintain steady academic coursework performance alongside practical builds."),
        ]

        suggestions: List[Dict[str, Any]] = []
        for key, label, unit, min_gap, action_tip in dimensions:
            user_val = float(user_profile.get(key, 0.0))
            target_val = float(ref_avg.get(key, 0.0))
            gap = round(target_val - user_val, 2)
            if gap >= min_gap:
                status = "Opportunity to Grow"
                message = f"Your score ({user_val:g}{unit}) is {gap:g} below {ref_label} ({target_val:g}{unit}). {action_tip}"
            elif gap > 0:
                status = "Near Benchmark"
                message = f"Close to {ref_label} ({user_val:g}{unit} vs {target_val:g}{unit}). Minor polish keeps you competitive."
            else:
                status = "Strong / Above Peer Benchmark"
                message = f"Matches or exceeds {ref_label} ({user_val:g}{unit} vs {target_val:g}{unit})."

            suggestions.append({
                "dimension": label,
                "user_value": user_val,
                "benchmark_value": target_val,
                "benchmark_label": ref_label,
                "gap": gap,
                "status": status,
                "recommendation": message,
            })

        return suggestions

    def run_leave_one_out_consistency_check(self, k: int = 5) -> Dict[str, Any]:
        """
        Internal consistency check on the 30 synthetic profiles:
        Leaves each synthetic student out and checks if weighted KNN (K=5)
        recovers their Career_Field among the remaining 29 profiles.
        Explicitly reported as an internal synthetic benchmark, NOT real-world accuracy.
        """
        df = self.students_df
        total = len(df)
        if total < 2:
            return {"evaluated": 0, "matches": 0, "consistency_pct": 0.0}

        full_mat = self._assemble_weighted_matrix(
            self.student_skills_matrix,
            self.student_numeric_matrix,
            self.student_major_matrix,
            "Balanced (Skills + Academics)",
        )

        k_eff = min(k, total - 1)
        matches = 0
        for i in range(total):
            mask = np.ones(total, dtype=bool)
            mask[i] = False
            train_mat = full_mat[mask]
            train_fields = df.loc[mask, "Career_Field"].tolist()
            query = full_mat[i : i + 1]

            nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
            nn.fit(train_mat)
            dists, idxs = nn.kneighbors(query, n_neighbors=k_eff)

            votes: Dict[str, float] = defaultdict(float)
            for d, idx in zip(dists[0], idxs[0]):
                votes[train_fields[int(idx)]] += 1.0 / (1.0 + float(d))

            predicted_field = max(votes.items(), key=lambda x: x[1])[0]
            if predicted_field == df.iloc[i]["Career_Field"]:
                matches += 1

        return {
            "evaluated": total,
            "matches": matches,
            "consistency_pct": round((matches / total) * 100.0, 1),
            "k": k_eff,
        }

    def run_knn_validation_checks(self) -> List[Dict[str, Any]]:
        """Run deterministic functional validation checks on the KNN peer matching pipeline."""
        checks: List[Dict[str, Any]] = []

        # 1. Dataset integrity
        n_rows = len(self.students_df)
        checks.append({
            "check": "Student Dataset Integrity (30 Profiles)",
            "status": "PASSED" if n_rows == 30 else "FAILED",
            "details": f"Loaded {n_rows} synthetic student profiles with {self.students_df['Career_Field'].nunique()} career fields.",
        })

        # 2. Distance monotonicity and K-bounds
        sample_profile = {
            "student_name": "Test Student",
            "major": "Computer Science",
            "university_year": 3,
            "cgpa": 8.5,
            "cgpa_scale": 10.0,
            "technical_skills": ["python", "sql", "pandas", "machine learning"],
            "programming_skill": 8,
            "problem_solving": 8,
            "communication_skills": 7,
            "projects_completed": 3,
            "certifications": 2,
            "internships": 1,
        }
        res = self.find_similar_students(sample_profile, k=5, metric="euclidean")
        peers = res["peers"]
        dists = [p["distance"] for p in peers]
        is_sorted = all(dists[i] <= dists[i + 1] for i in range(len(dists) - 1))
        checks.append({
            "check": "KNN Distance Ordering & K-Bound (K=5)",
            "status": "PASSED" if len(peers) == 5 and is_sorted else "FAILED",
            "details": f"Returned {len(peers)} peers with non-decreasing distances: {dists}.",
        })

        # 3. CGPA scale normalization equivalence (4.0/4.0 == 10.0/10.0)
        p4 = {**sample_profile, "cgpa": 4.0, "cgpa_scale": 4.0}
        p10 = {**sample_profile, "cgpa": 10.0, "cgpa_scale": 10.0}
        r4 = self.find_similar_students(p4, k=3)
        r10 = self.find_similar_students(p10, k=3)
        same_top = [p["student_id"] for p in r4["peers"]] == [p["student_id"] for p in r10["peers"]]
        checks.append({
            "check": "CGPA Scale Normalization (4.0 vs 10.0)",
            "status": "PASSED" if same_top else "FAILED",
            "details": "4.0/4.0 and 10.0/10.0 produce identical normalized CGPA (10.0) and neighbor ranking.",
        })

        # 4. Unseen major & empty skills edge case
        edge_profile = {
            "student_name": "Edge Student",
            "major": "Aerospace Engineering",
            "university_year": 1,
            "cgpa": 7.0,
            "cgpa_scale": 10.0,
            "technical_skills": [],
            "programming_skill": 5,
            "problem_solving": 5,
            "communication_skills": 5,
            "projects_completed": 0,
            "certifications": 0,
            "internships": 0,
        }
        edge_res = self.find_similar_students(edge_profile, k=3, metric="cosine")
        checks.append({
            "check": "Unseen Major & Empty Skills Edge Case",
            "status": "PASSED" if edge_res["status"] == "success" and len(edge_res["peers"]) == 3 else "FAILED",
            "details": "Handled unknown major and empty skills gracefully without errors.",
        })

        # 5. Internal LOO consistency on synthetic profiles
        loo = self.run_leave_one_out_consistency_check(k=5)
        checks.append({
            "check": "Synthetic Leave-One-Out Career Consistency (K=5)",
            "status": "PASSED" if loo["consistency_pct"] >= 70.0 else "FAILED",
            "details": (
                f"{loo['matches']}/{loo['evaluated']} ({loo['consistency_pct']}%) synthetic profiles "
                "recovered their Career_Field via weighted LOO-KNN (internal synthetic check only)."
            ),
        })

        return checks


# Singleton instance for efficient reuse across Streamlit reruns
_peer_engine_instance: Optional[PeerMatchEngine] = None


def get_peer_engine() -> PeerMatchEngine:
    global _peer_engine_instance
    if _peer_engine_instance is None:
        _peer_engine_instance = PeerMatchEngine()
    return _peer_engine_instance
