# src/utils.py
# EDA and visualization functions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

# --- Global style ---
sns.set_theme(style="whitegrid", palette="muted")
PASS_COLOR = "#2ecc71"   # green
FAIL_COLOR = "#e74c3c"   # red
SAVE_DIR   = "data"


def save_fig(filename: str):
    """Save figure to data/ folder and close it."""
    path = os.path.join(SAVE_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path}")


# ── Chart 1 ──────────────────────────────────────────────
def plot_performance_distribution(df: pd.DataFrame):
    """Bar chart: How many students Pass vs Fail."""
    counts = df["performance_label"].value_counts()
    colors = [PASS_COLOR if x == "Pass" else FAIL_COLOR for x in counts.index]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", width=0.5)

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1, str(val),
                ha="center", va="bottom", fontweight="bold", fontsize=12)

    ax.set_title("Student Performance Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Performance", fontsize=12)
    ax.set_ylabel("Number of Students", fontsize=12)
    ax.set_ylim(0, counts.max() + 20)
    save_fig("chart1_performance_distribution.png")


# ── Chart 2 ──────────────────────────────────────────────
def plot_attendance_vs_grade(df: pd.DataFrame):
    """Scatter plot: Attendance % vs Final Grade, coloured by Pass/Fail."""
    colors = df["performance_label"].map({"Pass": PASS_COLOR, "Fail": FAIL_COLOR})

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["attendance_pct"], df["final_grade"],
               c=colors, alpha=0.7, edgecolors="white", linewidth=0.5, s=60)

    # Trend line
    z = np.polyfit(df["attendance_pct"], df["final_grade"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df["attendance_pct"].min(), df["attendance_pct"].max(), 100)
    ax.plot(x_line, p(x_line), "k--", linewidth=1.5, alpha=0.6, label="Trend")

    # Legend
    pass_patch = mpatches.Patch(color=PASS_COLOR, label="Pass")
    fail_patch = mpatches.Patch(color=FAIL_COLOR, label="Fail")
    ax.legend(handles=[pass_patch, fail_patch], fontsize=10)

    ax.set_title("Attendance vs Final Grade", fontsize=14, fontweight="bold")
    ax.set_xlabel("Attendance (%)", fontsize=12)
    ax.set_ylabel("Final Grade", fontsize=12)
    save_fig("chart2_attendance_vs_grade.png")


# ── Chart 3 ──────────────────────────────────────────────
def plot_study_hours_boxplot(df: pd.DataFrame):
    """Boxplot: Study hours distribution for Pass vs Fail students."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(
        data=df, x="performance_label", y="study_hours_per_day",
        hue="performance_label",
        palette={"Pass": PASS_COLOR, "Fail": FAIL_COLOR},
        width=0.5, linewidth=1.5, legend=False, ax=ax
    )
    ax.set_title("Study Hours: Pass vs Fail", fontsize=14, fontweight="bold")
    ax.set_xlabel("Performance", fontsize=12)
    ax.set_ylabel("Study Hours per Day", fontsize=12)
    save_fig("chart3_study_hours_boxplot.png")


# ── Chart 4 ──────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame):
    """Heatmap: Correlation between all numeric features."""
    numeric_df = df.select_dtypes(include=[np.number])

    fig, ax = plt.subplots(figsize=(9, 6))
    mask = np.triu(np.ones_like(numeric_df.corr(), dtype=bool))  # upper triangle mask
    sns.heatmap(
        numeric_df.corr().round(2),
        mask=mask,
        annot=True, fmt=".2f",
        cmap="RdYlGn", center=0,
        linewidths=0.5, linecolor="white",
        ax=ax
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    save_fig("chart4_correlation_heatmap.png")


# ── Chart 5 ──────────────────────────────────────────────
def plot_parent_education_impact(df: pd.DataFrame):
    """Grouped bar: Pass rate by parent education level."""
    order = ["None", "School", "Graduate", "Postgraduate"]

    # Calculate pass rate per education level
    summary = (
        df.groupby("parent_education")["performance_label"]
        .apply(lambda x: (x == "Pass").mean() * 100)
        .reindex(order)
        .reset_index()
    )
    summary.columns = ["parent_education", "pass_rate"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        summary["parent_education"], summary["pass_rate"],
        color=["#3498db", "#9b59b6", "#1abc9c", "#e67e22"],
        edgecolor="white", width=0.5
    )

    for bar, val in zip(bars, summary["pass_rate"]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1, f"{val:.1f}%",
                ha="center", va="bottom", fontweight="bold", fontsize=11)

    ax.set_title("Pass Rate by Parent Education Level", fontsize=14, fontweight="bold")
    ax.set_xlabel("Parent Education", fontsize=12)
    ax.set_ylabel("Pass Rate (%)", fontsize=12)
    ax.set_ylim(0, 110)
    save_fig("chart5_parent_education_impact.png")


# ── Chart 6 ──────────────────────────────────────────────
def plot_at_risk_students(df: pd.DataFrame):
    """
    Identify and visualize at-risk students:
    attendance < 60% AND study_hours < 3 AND prev_grade < 50
    """
    df = df.copy()
    df["at_risk"] = (
        (df["attendance_pct"] < 60) &
        (df["study_hours_per_day"] < 3) &
        (df["prev_grade"] < 50)
    )

    risk_counts = df["at_risk"].value_counts().reindex([True, False])
    labels      = ["At Risk", "Not At Risk"]
    colors      = [FAIL_COLOR, PASS_COLOR]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart
    ax1.pie(
        risk_counts.values,
        labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2}
    )
    ax1.set_title("At-Risk Student Proportion", fontsize=13, fontweight="bold")

    # Scatter: attendance vs prev_grade, highlight at-risk
    scatter_colors = df["at_risk"].map({True: FAIL_COLOR, False: "#95a5a6"})
    ax2.scatter(df["attendance_pct"], df["prev_grade"],
                c=scatter_colors, alpha=0.7, s=60,
                edgecolors="white", linewidth=0.5)
    ax2.axvline(x=60, color="black", linestyle="--", linewidth=1.2, alpha=0.6)
    ax2.axhline(y=50, color="black", linestyle="--", linewidth=1.2, alpha=0.6)

    risk_patch    = mpatches.Patch(color=FAIL_COLOR, label=f"At Risk ({df['at_risk'].sum()})")
    no_risk_patch = mpatches.Patch(color="#95a5a6",  label="Not At Risk")
    ax2.legend(handles=[risk_patch, no_risk_patch], fontsize=10)

    ax2.set_title("At-Risk: Attendance vs Previous Grade", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Attendance (%)", fontsize=11)
    ax2.set_ylabel("Previous Grade", fontsize=11)

    plt.tight_layout()
    save_fig("chart6_at_risk_students.png")

    # Print at-risk student list
    at_risk_df = df[df["at_risk"]][["name", "attendance_pct",
                                    "study_hours_per_day", "prev_grade",
                                    "performance_label"]]
    print(f"\n[INFO] At-risk students identified: {len(at_risk_df)}")
    if len(at_risk_df) > 0:
        print(at_risk_df.to_string(index=False))
    return at_risk_df


# ── Run all charts ────────────────────────────────────────
def run_full_eda(df: pd.DataFrame):
    """Generate all 6 EDA charts."""
    print("\n" + "="*50)
    print(" RUNNING FULL EDA")
    print("="*50)

    plot_performance_distribution(df)
    plot_attendance_vs_grade(df)
    plot_study_hours_boxplot(df)
    plot_correlation_heatmap(df)
    plot_parent_education_impact(df)
    plot_at_risk_students(df)

    print("\n✅ All 6 charts saved to data/")