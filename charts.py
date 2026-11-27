"""
charts.py
---------
Modular, explainable ML visualization functions for the AI Internship Recommendation System.

Implements:
1. Skill Coverage Overview (Page 2)
2. Graph A: Top 5 Internship Match Scores (Page 3)
3. Graph B: Skill Match Breakdown (Page 3)
4. Graph C: Most In-Demand Skills in Dataset (Page 3)
5. Graph D: Advanced Similarity Heatmap (Page 3)
"""

from typing import List, Dict, Any, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Set consistent style parameters
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["axes.edgecolor"] = "#cbd5e1"
plt.rcParams["axes.linewidth"] = 0.8


def plot_skill_coverage_chart(coverage_data: Dict[str, Any]) -> plt.Figure:
    """
    Page 2: Skill Coverage Overview.
    Visualizes confirmed student skills present in dataset vocabulary vs. not in dataset vocabulary.
    """
    in_cnt = coverage_data.get("in_dataset_count", 0)
    out_cnt = coverage_data.get("not_in_dataset_count", 0)
    total = coverage_data.get("total_student_skills", 0)

    fig, ax = plt.subplots(figsize=(7, 2.2), dpi=130)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    categories = ["Recognized in Dataset", "Outside Dataset Vocab"]
    counts = [in_cnt, out_cnt]
    colors = ["#2563eb", "#94a3b8"]

    bars = ax.barh(categories, counts, color=colors, height=0.45)
    ax.set_xlim(0, max(total + 1, 4))
    ax.set_xlabel("Number of Confirmed Skills", fontsize=10, color="#334155", labelpad=6)
    
    # Value annotations on bars
    for bar, count in zip(bars, counts):
        pct = f"({(count / total * 100):.0f}%)" if total > 0 else ""
        ax.text(
            bar.get_width() + 0.15,
            bar.get_y() + bar.get_height() / 2,
            f"{count} {pct}",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="#0f172a"
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.grid(axis="x", linestyle="--", alpha=0.4, color="#e2e8f0")
    ax.tick_params(colors="#334155", labelsize=9.5)
    plt.tight_layout()
    return fig


def plot_top_match_scores_chart(recommendations: List[Dict[str, Any]]) -> plt.Figure:
    """
    GRAPH A: Top 5 Internship Match Scores.
    Horizontal bar chart showing actual cosine similarity scores (0.0 to 1.0) sorted descending.
    """
    if not recommendations:
        fig, ax = plt.subplots(figsize=(8, 2.5), dpi=130)
        ax.text(0.5, 0.5, "No recommendations available to display.", ha="center", va="center", color="#64748b")
        ax.axis("off")
        return fig

    # Reverse to display rank 1 at the top of the horizontal bar chart
    recs_rev = list(reversed(recommendations))
    labels = []
    scores = []

    for r in recs_rev:
        title = r["title"]
        loc = r["location"]
        short_title = title if len(title) <= 24 else title[:22] + ".."
        labels.append(f"{short_title} ({loc})")
        scores.append(float(r["similarity_score"]))

    fig, ax = plt.subplots(figsize=(8.5, max(3.0, len(recommendations) * 0.65)), dpi=130)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    bars = ax.barh(labels, scores, color="#2563eb", height=0.52)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Cosine Similarity Score (0.0 to 1.0)", fontsize=10, color="#334155", labelpad=8)

    # Display numerical score on each bar
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_width() + 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f} ({round(score * 100, 1)}%)",
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="600",
            color="#0f172a"
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.grid(axis="x", linestyle="--", alpha=0.5, color="#e2e8f0")
    ax.tick_params(colors="#334155", labelsize=9.5)
    plt.tight_layout()
    return fig


def plot_skill_match_breakdown_chart(
    recommendations: List[Dict[str, Any]],
    total_confirmed_skills: int
) -> plt.Figure:
    """
    GRAPH B: Skill Match Breakdown.
    Horizontal grouped bar chart comparing:
      1. Confirmed student skills count (profile reference)
      2. Overlapping skills count with each internship requirement
    """
    if not recommendations:
        fig, ax = plt.subplots(figsize=(8, 2.5), dpi=130)
        ax.text(0.5, 0.5, "No recommendations available to display.", ha="center", va="center", color="#64748b")
        ax.axis("off")
        return fig

    # Reverse to keep top match at the top
    recs_rev = list(reversed(recommendations))
    labels = []
    overlap_counts = []
    total_counts = [total_confirmed_skills] * len(recs_rev)

    for i, r in enumerate(recs_rev, start=1):
        title = r["title"]
        loc = r["location"]
        short_title = title if len(title) <= 22 else title[:20] + ".."
        labels.append(f"{short_title} ({loc})")
        overlap_counts.append(len(r.get("matched_skills", [])))

    fig, ax = plt.subplots(figsize=(8.8, max(3.4, len(recommendations) * 0.75)), dpi=130)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    y = np.arange(len(labels))
    height = 0.32

    # Grouped bars
    bars_total = ax.barh(y + height / 2, total_counts, height, label="Your Confirmed Skills", color="#cbd5e1")
    bars_overlap = ax.barh(y - height / 2, overlap_counts, height, label="Overlapping Skills", color="#0d9488")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5, color="#334155")
    max_val = max(total_confirmed_skills, max(overlap_counts) if overlap_counts else 1)
    ax.set_xlim(0, max_val + 2)
    ax.set_xlabel("Number of Skills", fontsize=10, color="#334155", labelpad=8)

    # Bar annotations
    for bar in bars_overlap:
        width = bar.get_width()
        ax.text(
            width + 0.15,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width)} overlap",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="600",
            color="#047857"
        )

    ax.legend(loc="lower right", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.grid(axis="x", linestyle="--", alpha=0.5, color="#e2e8f0")
    ax.tick_params(colors="#334155", labelsize=9.5)
    plt.tight_layout()
    return fig


def plot_dataset_skill_demand_chart(top_skills: List[Tuple[str, int]]) -> plt.Figure:
    """
    GRAPH C: Most In-Demand Skills in the Dataset.
    Horizontal bar chart showing the frequency of the top 10-15 recognized skills in dataset.
    """
    if not top_skills:
        fig, ax = plt.subplots(figsize=(8, 2.5), dpi=130)
        ax.text(0.5, 0.5, "Skill frequency data not available.", ha="center", va="center", color="#64748b")
        ax.axis("off")
        return fig

    # Reverse so highest is at top
    skills_rev = list(reversed(top_skills[:12]))
    names = [s[0].title() for s in skills_rev]
    freqs = [s[1] for s in skills_rev]

    fig, ax = plt.subplots(figsize=(8.5, 4.4), dpi=130)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    bars = ax.barh(names, freqs, color="#3b82f6", height=0.52)
    max_freq = max(freqs) if freqs else 100
    ax.set_xlim(0, max_freq * 1.15)
    ax.set_xlabel("Listing Frequency Across 388 Internships", fontsize=10, color="#334155", labelpad=8)

    for bar, count in zip(bars, freqs):
        ax.text(
            bar.get_width() + max_freq * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{count} listings",
            va="center",
            ha="left",
            fontsize=9,
            color="#0f172a",
            fontweight="500"
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.grid(axis="x", linestyle="--", alpha=0.5, color="#e2e8f0")
    ax.tick_params(colors="#334155", labelsize=9.5)
    plt.tight_layout()
    return fig


def plot_similarity_heatmap_chart(heatmap_df: pd.DataFrame) -> plt.Figure:
    """
    GRAPH D: Advanced Similarity Heatmap.
    Displays cosine similarity matrix between student profile / query skills and recommended listings.
    """
    if heatmap_df.empty:
        fig, ax = plt.subplots(figsize=(8, 2.5), dpi=130)
        ax.text(0.5, 0.5, "Heatmap data not available.", ha="center", va="center", color="#64748b")
        ax.axis("off")
        return fig

    num_rows = len(heatmap_df)
    num_cols = len(heatmap_df.columns)

    fig, ax = plt.subplots(figsize=(max(8.0, num_cols * 1.3), max(3.5, num_rows * 0.7)), dpi=130)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    data = heatmap_df.values
    # Clean blue-teal colormap
    cax = ax.imshow(data, cmap="Blues", aspect="auto", vmin=0.0, vmax=max(0.8, np.max(data) if data.size else 1.0))

    # Add colorbar
    cbar = fig.colorbar(cax, ax=ax, fraction=0.035, pad=0.04)
    cbar.ax.tick_params(labelsize=8.5, colors="#334155")
    cbar.set_label("Cosine Similarity", fontsize=9.5, color="#334155")

    # Tick marks
    ax.set_xticks(np.arange(num_cols))
    ax.set_yticks(np.arange(num_rows))
    ax.set_xticklabels([c.title() for c in heatmap_df.columns], rotation=25, ha="right", fontsize=9.5, color="#0f172a")
    ax.set_yticklabels(heatmap_df.index, fontsize=9.5, color="#0f172a")

    # Annotate cell values
    for r in range(num_rows):
        for c in range(num_cols):
            val = data[r, c]
            # Choose text color based on cell brightness
            text_color = "#ffffff" if val > 0.45 else "#0f172a"
            ax.text(c, r, f"{val:.3f}", ha="center", va="center", color=text_color, fontsize=9, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.tight_layout()
    return fig


def plot_peer_competency_radar(
    user_profile: Dict[str, Any],
    peer_avg: Dict[str, float],
    top_peer: Dict[str, Any],
) -> plt.Figure:
    """
    AI Peer Match: Polar radar chart comparing Your Profile vs. K-Peer Average
    vs. Top-1 Nearest Peer across core 0-10 academic & skill ratings.
    """
    categories = ["CGPA (/10)", "Programming", "Problem\nSolving", "Communication"]
    user_vals = [
        float(user_profile.get("cgpa_10", 0.0)),
        float(user_profile.get("programming_skill", 0.0)),
        float(user_profile.get("problem_solving", 0.0)),
        float(user_profile.get("communication_skills", 0.0)),
    ]
    avg_vals = [
        float(peer_avg.get("cgpa_10", 0.0)),
        float(peer_avg.get("programming_skill", 0.0)),
        float(peer_avg.get("problem_solving", 0.0)),
        float(peer_avg.get("communication_skills", 0.0)),
    ]
    top_vals = [
        float(top_peer.get("cgpa_10", 0.0)),
        float(top_peer.get("programming_skill", 0.0)),
        float(top_peer.get("problem_solving", 0.0)),
        float(top_peer.get("communication_skills", 0.0)),
    ]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    user_vals += user_vals[:1]
    avg_vals += avg_vals[:1]
    top_vals += top_vals[:1]

    fig, ax = plt.subplots(figsize=(5.6, 4.4), dpi=130, subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    ax.plot(angles, user_vals, color="#2563eb", linewidth=2.2, label="You")
    ax.fill(angles, user_vals, color="#2563eb", alpha=0.14)

    ax.plot(angles, avg_vals, color="#0d9488", linewidth=2.0, linestyle="--", label="K-Peer Avg")
    ax.fill(angles, avg_vals, color="#0d9488", alpha=0.08)

    ax.plot(angles, top_vals, color="#64748b", linewidth=1.6, linestyle=":", label=f"Top Peer ({top_peer.get('student_name', 'Peer')})")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9.5, color="#0f172a", fontweight="600")
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=8, color="#64748b")
    ax.grid(color="#e2e8f0", linestyle="--", linewidth=0.8)
    ax.spines["polar"].set_color("#cbd5e1")

    ax.legend(loc="upper right", bbox_to_anchor=(1.22, 1.12), frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=8.5)
    plt.tight_layout()
    return fig


def plot_peer_experience_comparison(
    user_profile: Dict[str, Any],
    peer_avg: Dict[str, float],
    top_peer: Dict[str, Any],
) -> plt.Figure:
    """
    AI Peer Match: Grouped horizontal bar chart comparing Projects Completed,
    Certifications, and Previous Internships (exact counts).
    """
    metrics = ["Internships", "Certifications", "Projects Completed"]
    user_counts = [
        float(user_profile.get("internships", 0.0)),
        float(user_profile.get("certifications", 0.0)),
        float(user_profile.get("projects_completed", 0.0)),
    ]
    avg_counts = [
        float(peer_avg.get("internships", 0.0)),
        float(peer_avg.get("certifications", 0.0)),
        float(peer_avg.get("projects_completed", 0.0)),
    ]
    top_counts = [
        float(top_peer.get("internships", 0.0)),
        float(top_peer.get("certifications", 0.0)),
        float(top_peer.get("projects_completed", 0.0)),
    ]

    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=130)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    y = np.arange(len(metrics))
    h = 0.24

    bars_u = ax.barh(y + h, user_counts, h, label="You", color="#2563eb")
    bars_a = ax.barh(y, avg_counts, h, label="K-Peer Average", color="#0d9488")
    bars_t = ax.barh(y - h, top_counts, h, label=f"Top Peer ({top_peer.get('student_name', 'Peer')})", color="#94a3b8")

    max_val = max(max(user_counts), max(avg_counts), max(top_counts), 2.0)
    ax.set_xlim(0, max_val * 1.25)
    ax.set_yticks(y)
    ax.set_yticklabels(metrics, fontsize=9.5, color="#0f172a", fontweight="600")
    ax.set_xlabel("Count", fontsize=9.5, color="#334155")

    for bars, txt_color in [(bars_u, "#1d4ed8"), (bars_a, "#0f766e"), (bars_t, "#475569")]:
        for bar in bars:
            w = bar.get_width()
            ax.text(w + max_val * 0.02, bar.get_y() + bar.get_height() / 2, f"{w:.1f}".rstrip("0").rstrip("."), va="center", ha="left", fontsize=8.5, color=txt_color, fontweight="600")

    ax.legend(loc="lower right", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.grid(axis="x", linestyle="--", alpha=0.45, color="#e2e8f0")
    ax.tick_params(colors="#334155", labelsize=9)
    plt.tight_layout()
    return fig


def plot_career_support_chart(career_recs: List[Dict[str, Any]]) -> plt.Figure:
    """
    AI Peer Match: Horizontal bar chart showing KNN distance-weighted support (%)
    for recommended career fields among the K nearest peers.
    """
    if not career_recs:
        fig, ax = plt.subplots(figsize=(7, 2.2), dpi=130)
        ax.text(0.5, 0.5, "No career recommendations available.", ha="center", va="center", color="#64748b")
        ax.axis("off")
        return fig

    recs_rev = list(reversed(career_recs))
    labels = [r["career_field"] for r in recs_rev]
    shares = [r["weighted_share_pct"] for r in recs_rev]
    counts = [f"{r['peer_count']}/{r['total_k']} peers ({r['weighted_share_pct']}%)" for r in recs_rev]

    fig, ax = plt.subplots(figsize=(7.6, max(2.6, len(recs_rev) * 0.65)), dpi=130)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    colors = ["#0d9488" if i == len(recs_rev) - 1 else "#3b82f6" for i in range(len(recs_rev))]
    bars = ax.barh(labels, shares, color=colors, height=0.5)
    ax.set_xlim(0, 118)
    ax.set_xlabel("Distance-Weighted Neighbor Support (%)", fontsize=9.5, color="#334155")

    for bar, txt in zip(bars, counts):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2, txt, va="center", ha="left", fontsize=9, fontweight="600", color="#0f172a")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.grid(axis="x", linestyle="--", alpha=0.45, color="#e2e8f0")
    ax.tick_params(colors="#334155", labelsize=9.5)
    plt.tight_layout()
    return fig


def plot_knn_neighbors_chart(all_ranked: List[Dict[str, Any]], k: int, metric: str = "euclidean") -> plt.Figure:
    """
    AI Peer Match (KNN Model Insights): Visualizes the top-15 ranked synthetic students
    by KNN similarity score, highlighting the K selected neighbors vs. cutoff boundary.
    """
    if not all_ranked:
        fig, ax = plt.subplots(figsize=(8, 2.5), dpi=130)
        ax.text(0.5, 0.5, "No neighbor ranking data available.", ha="center", va="center", color="#64748b")
        ax.axis("off")
        return fig

    subset = all_ranked[: min(15, len(all_ranked))]
    subset_rev = list(reversed(subset))
    labels = [f"#{p['rank']} {p['student_name']} ({p['career_field']})" for p in subset_rev]
    sims = [p["similarity_pct"] for p in subset_rev]
    dists = [p["distance"] for p in subset_rev]
    colors = ["#2563eb" if p["rank"] <= k else "#cbd5e1" for p in subset_rev]

    fig, ax = plt.subplots(figsize=(8.6, max(3.6, len(subset_rev) * 0.36)), dpi=130)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    bars = ax.barh(labels, sims, color=colors, height=0.58)
    ax.set_xlim(0, 115)
    ax.set_xlabel(f"KNN Similarity Score (%) — Metric: {metric.title()}", fontsize=9.5, color="#334155")

    for bar, sim, dist in zip(bars, sims, dists):
        ax.text(
            bar.get_width() + 1.2,
            bar.get_y() + bar.get_height() / 2,
            f"{sim:.1f}% (d={dist:.3f})",
            va="center",
            ha="left",
            fontsize=8.3,
            color="#0f172a",
            fontweight="500",
        )

    # Draw K-cutoff line separating top-K neighbors from rank K+1
    if 0 < k < len(subset_rev):
        cutoff_y = len(subset_rev) - k - 0.5
        ax.axhline(cutoff_y, color="#0d9488", linestyle="--", linewidth=1.5)
        ax.text(
            113,
            cutoff_y + 0.12,
            f"K={k} Cutoff",
            ha="right",
            va="bottom",
            fontsize=8.5,
            fontweight="700",
            color="#0d9488",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.grid(axis="x", linestyle="--", alpha=0.45, color="#e2e8f0")
    ax.tick_params(colors="#334155", labelsize=8.8)
    plt.tight_layout()
    return fig


def plot_interview_question_scores_chart(question_rows: List[Dict[str, Any]]) -> plt.Figure:
    """
    AI Mock Interview: Bar chart of question-wise practice scores (0-100),
    marking skipped questions distinctly.
    """
    if not question_rows:
        fig, ax = plt.subplots(figsize=(7.5, 2.5), dpi=130)
        ax.text(0.5, 0.5, "No interview questions recorded yet.", ha="center", va="center", color="#64748b")
        ax.axis("off")
        return fig

    labels = [f"Q{r['index']}" for r in question_rows]
    scores = [float(r["score"]) if r["score"] is not None else 0.0 for r in question_rows]
    colors = []
    for r in question_rows:
        if r["score"] is None:
            colors.append("#cbd5e1")
        elif r["score"] >= 75:
            colors.append("#0d9488")
        elif r["score"] >= 50:
            colors.append("#2563eb")
        else:
            colors.append("#f59e0b")

    fig, ax = plt.subplots(figsize=(max(6.8, len(labels) * 0.65), 3.6), dpi=130)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    bars = ax.bar(labels, scores, color=colors, width=0.55)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Practice Score (/100)", fontsize=9.5, color="#334155")

    attempted = [r["score"] for r in question_rows if r["score"] is not None]
    if attempted:
        avg_val = float(np.mean(attempted))
        ax.axhline(avg_val, color="#1e293b", linestyle="--", linewidth=1.2, label=f"Attempted Avg ({avg_val:.1f})")
        ax.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=8.5)

    for bar, r in zip(bars, question_rows):
        txt = "Skip" if r["score"] is None else f"{r['score']:.0f}"
        y_pos = 4 if r["score"] is None else bar.get_height() + 2.2
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            txt,
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="600",
            color="#475569" if r["score"] is None else "#0f172a",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.grid(axis="y", linestyle="--", alpha=0.45, color="#e2e8f0")
    ax.tick_params(colors="#334155", labelsize=9)
    plt.tight_layout()
    return fig


def plot_interview_rubric_dimensions_chart(dimension_averages: Dict[str, float]) -> plt.Figure:
    """
    AI Mock Interview: Horizontal bar chart of average scores across the 5 evaluation dimensions.
    """
    if not dimension_averages:
        fig, ax = plt.subplots(figsize=(7.5, 2.5), dpi=130)
        ax.text(0.5, 0.5, "Attempt at least one question to view dimension breakdown.", ha="center", va="center", color="#64748b")
        ax.axis("off")
        return fig

    order = [
        ("answer_clarity", "Answer Clarity (10%)"),
        ("completeness", "Completeness (10%)"),
        ("technical_correctness", "Technical Correctness (15%)"),
        ("concept_coverage", "Concept Coverage (30%)"),
        ("semantic_relevance", "Semantic Relevance (35%)"),
    ]
    labels = [label for key, label in order if key in dimension_averages]
    vals = [float(dimension_averages[key]) for key, _ in order if key in dimension_averages]

    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=130)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    bars = ax.barh(labels, vals, color="#2563eb", height=0.52)
    ax.set_xlim(0, 114)
    ax.set_xlabel("Average Dimension Score (/100)", fontsize=9.5, color="#334155")

    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 1.8, bar.get_y() + bar.get_height() / 2, f"{v:.1f}", va="center", ha="left", fontsize=9, fontweight="600", color="#0f172a")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.grid(axis="x", linestyle="--", alpha=0.45, color="#e2e8f0")
    ax.tick_params(colors="#334155", labelsize=9)
    plt.tight_layout()
    return fig


def plot_interview_topic_performance_chart(topic_averages: List[Dict[str, Any]]) -> plt.Figure:
    """
    AI Mock Interview: Horizontal bar chart showing average score per interview topic.
    """
    if not topic_averages:
        fig, ax = plt.subplots(figsize=(7.5, 2.5), dpi=130)
        ax.text(0.5, 0.5, "No topic metrics available.", ha="center", va="center", color="#64748b")
        ax.axis("off")
        return fig

    topics_rev = list(reversed(topic_averages))
    labels = [t["topic"] if len(t["topic"]) <= 28 else t["topic"][:25] + "..." for t in topics_rev]
    scores = [float(t["avg_score"]) for t in topics_rev]
    colors = ["#0d9488" if s >= 70 else ("#3b82f6" if s >= 50 else "#f59e0b") for s in scores]

    fig, ax = plt.subplots(figsize=(8.0, max(2.8, len(topics_rev) * 0.48)), dpi=130)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    bars = ax.barh(labels, scores, color=colors, height=0.54)
    ax.set_xlim(0, 114)
    ax.set_xlabel("Topic Average Score (/100)", fontsize=9.5, color="#334155")

    for bar, t in zip(bars, topics_rev):
        ax.text(
            bar.get_width() + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{t['avg_score']:.1f} (n={t['count']})",
            va="center",
            ha="left",
            fontsize=8.8,
            fontweight="600",
            color="#0f172a",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.grid(axis="x", linestyle="--", alpha=0.45, color="#e2e8f0")
    ax.tick_params(colors="#334155", labelsize=9)
    plt.tight_layout()
    return fig
