"""
Create visualization for sensitivity analysis results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import EXP3_RESULTS_DIR


def plot_sensitivity_analysis():
    """Create heatmap visualization for sensitivity analysis."""

    # Load results
    results_file = EXP3_RESULTS_DIR / "sensitivity_analysis.json"

    if not results_file.exists():
        print(f"Error: Results file not found at {results_file}")
        return

    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract parameters
    chunk_sizes = data["parameters"]["chunk_sizes"]
    top_k_values = data["parameters"]["top_k_values"]
    aggregated = data["aggregated_results"]

    # Create matrix for heatmap
    response_lengths = np.zeros((len(top_k_values), len(chunk_sizes)))

    for i, top_k in enumerate(top_k_values):
        for j, chunk_size in enumerate(chunk_sizes):
            key = f"chunk{chunk_size}_k{top_k}"
            if key in aggregated:
                response_lengths[i, j] = aggregated[key]["avg_response_length"]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create heatmap
    im = ax.imshow(response_lengths, cmap='YlOrRd', aspect='auto')

    # Set ticks
    ax.set_xticks(np.arange(len(chunk_sizes)))
    ax.set_yticks(np.arange(len(top_k_values)))

    # Set tick labels
    ax.set_xticklabels(chunk_sizes)
    ax.set_yticklabels(top_k_values)

    # Labels
    ax.set_xlabel('Chunk Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Top-K Retrieved Documents', fontsize=12, fontweight='bold')
    ax.set_title('RAG Parameter Sensitivity Analysis:\nAverage Response Length',
                 fontsize=14, fontweight='bold', pad=20)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Avg Response Length (chars)', rotation=270, labelpad=20)

    # Add text annotations
    for i in range(len(top_k_values)):
        for j in range(len(chunk_sizes)):
            text = ax.text(j, i, f'{response_lengths[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=10)

    # Adjust layout
    plt.tight_layout()

    # Save plot
    output_file = EXP3_RESULTS_DIR / "sensitivity_analysis_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {output_file}")

    # Also save as PDF
    output_pdf = EXP3_RESULTS_DIR / "sensitivity_analysis_heatmap.pdf"
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"✓ PDF saved to {output_pdf}")

    plt.close()


if __name__ == "__main__":
    plot_sensitivity_analysis()
