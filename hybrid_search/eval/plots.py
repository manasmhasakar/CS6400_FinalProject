import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_latency_vs_recall(df: pd.DataFrame, output_path: str) -> None:
    """
    Scatter plot of latency vs recall for all methods.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="recall", y="latency_ms", hue="method", style="bin", s=80, alpha=0.7)
    plt.xlabel("Recall@k")
    plt.ylabel("Latency (ms)")
    plt.title("Latency vs Recall")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_selectivity_vs_latency(df: pd.DataFrame, output_path: str) -> None:
    """
    Line plot of p95 latency vs selectivity bin.
    """
    # Group by method and bin
    grouped = df.groupby(["method", "bin"])["latency_ms"].quantile(0.95).reset_index()
    grouped.columns = ["method", "bin", "p95_latency_ms"]
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=grouped, x="bin", y="p95_latency_ms", hue="method", marker="o")
    plt.xlabel("Selectivity Bin")
    plt.ylabel("p95 Latency (ms)")
    plt.title("p95 Latency vs Selectivity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_candidates_scored(df: pd.DataFrame, output_path: str) -> None:
    """
    Bar chart of average candidates scored per method.
    """
    grouped = df.groupby("method")["candidates_scored"].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=grouped, x="method", y="candidates_scored")
    plt.xlabel("Method")
    plt.ylabel("Avg Candidates Scored")
    plt.title("Average Candidates Scored per Query")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_all_plots(df: pd.DataFrame, output_dir: str) -> None:
    """
    Generate all evaluation plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plot_latency_vs_recall(df, os.path.join(output_dir, "latency_vs_recall.png"))
    plot_selectivity_vs_latency(df, os.path.join(output_dir, "selectivity_vs_latency.png"))
    plot_candidates_scored(df, os.path.join(output_dir, "candidates_scored.png"))
    
    print(f"✓ Plots saved to {output_dir}")

