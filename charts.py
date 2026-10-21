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
