"""
Minimal tests for visualization module to verify plot generation.
"""

import pytest
import sys
from pathlib import Path
import tempfile
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.visualization import plot_accuracy_by_position


class TestVisualizationOutput:
    """Test that visualization functions generate output files."""

    def test_plot_accuracy_by_position_creates_png(self):
        """Test that plot_accuracy_by_position creates a PNG file."""
        # Create minimal test data
        results = {
            "start": [0.5, 0.6, 0.4],
            "middle": [0.3, 0.4, 0.5],
            "end": [0.6, 0.5, 0.7]
        }

        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_plot.png"

            # Generate plot
            plot_accuracy_by_position(results, save_path=output_path)

            # Assert PNG file was created
            assert output_path.exists(), f"PNG file not created at {output_path}"

            # Assert file is not empty
            assert output_path.stat().st_size > 0, "PNG file is empty"

            # Assert file has reasonable size (at least 1KB)
            assert output_path.stat().st_size > 1000, "PNG file too small"

    def test_plot_creates_valid_png_format(self):
        """Test that generated file is a valid PNG."""
        results = {
            "start": [0.8, 0.7, 0.9],
            "middle": [0.6, 0.5, 0.4],
            "end": [0.7, 0.8, 0.6]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "format_test.png"

            plot_accuracy_by_position(results, save_path=output_path)

            # Check PNG magic number (first 8 bytes)
            with open(output_path, 'rb') as f:
                header = f.read(8)
                # PNG signature: 137 80 78 71 13 10 26 10
                assert header == b'\x89PNG\r\n\x1a\n', "Not a valid PNG file"
