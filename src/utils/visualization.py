"""
Visualization utilities for creating plots and charts.

This module provides functions for creating publication-quality
visualizations for all experiment results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pandas as pd

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import PLOT_CONFIG


def setup_plot_style():
    """Set up the default plotting style."""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette(PLOT_CONFIG["color_palette"])
    plt.rcParams['figure.figsize'] = PLOT_CONFIG["figsize"]
    plt.rcParams['figure.dpi'] = PLOT_CONFIG["dpi"]
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10


def plot_accuracy_by_position(results: Dict[str, List[float]],
                               save_path: Optional[Path] = None,
                               title: str = "Accuracy by Fact Position"):
    """
    Plot bar chart showing accuracy by position (Experiment 1).

    Args:
        results: Dictionary mapping position to list of accuracy scores
        save_path: Path to save the figure
        title: Plot title
    """
    setup_plot_style()

    positions = list(results.keys())
    means = [np.mean(results[pos]) for pos in positions]
    stds = [np.std(results[pos]) for pos in positions]

    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize"])

    x_pos = np.arange(len(positions))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                  color=sns.color_palette(PLOT_CONFIG["color_palette"], len(positions)),
                  alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.2f}',
                ha='center', va='bottom', fontweight='bold')

    ax.set_xlabel('Fact Position', fontweight='bold')
    ax.set_ylabel('Accuracy Score', fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([pos.capitalize() for pos in positions])
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.close()


def plot_context_size_impact(results: List[Dict[str, Any]],
                             save_path: Optional[Path] = None,
                             title: str = "Context Size Impact on Performance"):
    """
    Plot line graphs showing accuracy and latency vs context size (Experiment 2).

    Args:
        results: List of result dictionaries with num_docs, accuracy, latency
        save_path: Path to save the figure
        title: Plot title
    """
    setup_plot_style()

    df = pd.DataFrame(results)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Accuracy vs Context Size
    ax1.plot(df['num_docs'], df['accuracy_mean'], marker='o',
             linewidth=2, markersize=8, color='#2ecc71')
    ax1.fill_between(df['num_docs'],
                      df['accuracy_mean'] - df['accuracy_std'],
                      df['accuracy_mean'] + df['accuracy_std'],
                      alpha=0.3, color='#2ecc71')
    ax1.set_xlabel('Number of Documents', fontweight='bold')
    ax1.set_ylabel('Accuracy Score', fontweight='bold')
    ax1.set_title('Accuracy vs Context Size', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)

    # Plot 2: Latency vs Context Size
    ax2.plot(df['num_docs'], df['latency_mean'], marker='s',
             linewidth=2, markersize=8, color='#e74c3c')
    ax2.fill_between(df['num_docs'],
                      df['latency_mean'] - df['latency_std'],
                      df['latency_mean'] + df['latency_std'],
                      alpha=0.3, color='#e74c3c')
    ax2.set_xlabel('Number of Documents', fontweight='bold')
    ax2.set_ylabel('Latency (seconds)', fontweight='bold')
    ax2.set_title('Latency vs Context Size', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Tokens vs Context Size
    ax3.plot(df['num_docs'], df['tokens_used'], marker='^',
             linewidth=2, markersize=8, color='#3498db')
    ax3.set_xlabel('Number of Documents', fontweight='bold')
    ax3.set_ylabel('Tokens Used', fontweight='bold')
    ax3.set_title('Token Consumption vs Context Size', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    fig.suptitle(title, fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.close()


def plot_rag_comparison(full_context_results: Dict[str, float],
                        rag_results: Dict[str, float],
                        save_path: Optional[Path] = None,
                        title: str = "RAG vs Full Context Comparison"):
    """
    Plot comparison between RAG and full context approaches (Experiment 3).

    Args:
        full_context_results: Metrics for full context approach
        rag_results: Metrics for RAG approach
        save_path: Path to save the figure
        title: Plot title
    """
    setup_plot_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Metrics to compare
    metrics = ['accuracy', 'latency']
    x = np.arange(len(metrics))
    width = 0.35

    # Plot 1: Metrics comparison
    full_values = [full_context_results.get(m, 0) for m in metrics]
    rag_values = [rag_results.get(m, 0) for m in metrics]

    # Normalize latency to 0-1 scale for comparison
    max_latency = max(full_values[1], rag_values[1])
    if max_latency > 0:
        full_values_norm = [full_values[0], full_values[1] / max_latency]
        rag_values_norm = [rag_values[0], rag_values[1] / max_latency]
    else:
        full_values_norm = full_values
        rag_values_norm = rag_values

    bars1 = ax1.bar(x - width/2, full_values_norm, width,
                    label='Full Context', color='#e74c3c', alpha=0.8)
    bars2 = ax1.bar(x + width/2, rag_values_norm, width,
                    label='RAG', color='#2ecc71', alpha=0.8)

    ax1.set_ylabel('Score (normalized)', fontweight='bold')
    ax1.set_title('Performance Metrics Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Accuracy', 'Speed (1/latency)'])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9)

    # Plot 2: Actual latency comparison
    methods = ['Full Context', 'RAG']
    latencies = [full_context_results.get('latency', 0),
                 rag_results.get('latency', 0)]

    bars = ax2.bar(methods, latencies,
                   color=['#e74c3c', '#2ecc71'], alpha=0.8)

    ax2.set_ylabel('Latency (seconds)', fontweight='bold')
    ax2.set_title('Response Time Comparison', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom', fontweight='bold')

    fig.suptitle(title, fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.close()


def plot_strategy_comparison(results: Dict[str, List[Dict[str, Any]]],
                             save_path: Optional[Path] = None,
                             title: str = "Context Management Strategies Comparison"):
    """
    Plot multi-line graph comparing strategies over time (Experiment 4).

    Args:
        results: Dictionary mapping strategy name to list of step results
        save_path: Path to save the figure
        title: Plot title
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {'select': '#2ecc71', 'compress': '#3498db', 'write': '#e74c3c'}
    markers = {'select': 'o', 'compress': 's', 'write': '^'}

    for strategy, strategy_results in results.items():
        steps = [r['step'] for r in strategy_results]
        accuracies = [r['accuracy'] for r in strategy_results]

        ax.plot(steps, accuracies,
                marker=markers.get(strategy, 'o'),
                linewidth=2.5,
                markersize=8,
                label=strategy.upper(),
                color=colors.get(strategy, '#95a5a6'))

    ax.set_xlabel('Action Step', fontweight='bold')
    ax.set_ylabel('Accuracy Score', fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.close()


def create_summary_table(data: Dict[str, Dict[str, Any]],
                         save_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Create a summary table for results.

    Args:
        data: Dictionary of experiment results
        save_path: Path to save the table (CSV)

    Returns:
        DataFrame with summary statistics
    """
    df = pd.DataFrame(data).T

    if save_path:
        df.to_csv(save_path)
        print(f"Table saved to {save_path}")

    return df


def plot_heatmap(data: pd.DataFrame,
                 save_path: Optional[Path] = None,
                 title: str = "Results Heatmap",
                 cmap: str = "RdYlGn"):
    """
    Create a heatmap visualization.

    Args:
        data: DataFrame with data to visualize
        save_path: Path to save the figure
        title: Plot title
        cmap: Colormap to use
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(data, annot=True, fmt='.2f', cmap=cmap,
                cbar_kws={'label': 'Score'},
                linewidths=0.5, ax=ax)

    ax.set_title(title, fontweight='bold', fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.close()
