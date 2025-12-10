# Experiment Template

**Purpose**: Reusable template for adding new experiments to the project.

**Version**: 1.0
**Date**: December 2025

---

## Overview

This template provides a standardized structure for creating new experiments, ensuring consistency, maintainability, and proper integration with the existing codebase.

---

## File Structure

```
src/
├── experiments/
│   └── exp{N}_{name}.py          # Main experiment script
├── config.py                      # Add EXP{N}_CONFIG section
data/
└── synthetic/{name}/              # Generated data directory
results/
└── exp{N}/                        # Results directory
tests/
└── test_exp{N}_{name}.py         # Unit tests
```

---

## Step 1: Configuration

Add to `src/config.py`:

```python
# ============================================================================
# EXPERIMENT {N}: {EXPERIMENT NAME}
# ============================================================================

EXP{N}_CONFIG = {
    "parameter_1": value1,  # Description
    "parameter_2": value2,  # Description
    "iterations": 10,  # Number of iterations for statistical reliability
    # Add all configurable parameters here
}

# Results directory
EXP{N}_RESULTS_DIR = RESULTS_DIR / "exp{N}"
```

---

## Step 2: Experiment Implementation

Create `src/experiments/exp{N}_{name}.py`:

```python
"""
Experiment {N}: {Experiment Name}

Brief description of what this experiment demonstrates.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from typing import List, Dict, Any
import json
from tqdm import tqdm
import numpy as np

from config import EXP{N}_CONFIG, EXP{N}_RESULTS_DIR
from data_generator import DataGenerator
from llm_interface import create_llm_interface
from evaluator import create_evaluator
from utils.visualization import plot_{metric_name}


class Experiment{N}:
    """Experiment to test {phenomenon}."""

    def __init__(self):
        """Initialize experiment."""
        self.config = EXP{N}_CONFIG
        self.results_dir = EXP{N}_RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.data_generator = DataGenerator()
        self.llm = create_llm_interface()
        self.evaluator = create_evaluator(use_embeddings=True)

        print("=" * 60)
        print(f"Experiment {N}: {Experiment Name}")
        print("=" * 60)

    def generate_data(self) -> List[Dict[str, Any]]:
        """
        Generate experimental data.

        Returns:
            List of data dictionaries
        """
        print("\n[1/3] Generating data...")
        # Implement data generation
        dataset = []
        # ... generation logic ...
        print(f"  ✓ Generated {len(dataset)} items")
        return dataset

    def run_single_trial(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single trial.

        Args:
            data: Input data dictionary

        Returns:
            Trial results
        """
        # Query LLM
        llm_result = self.llm.query(
            prompt=self.config["query_template"],
            context=data["context"]
        )

        # Evaluate response
        eval_result = self.evaluator.evaluate_{metric}(
            response=llm_result["response"],
            expected=data["expected"],
        )

        return {
            **llm_result,
            **eval_result,
        }

    def run_experiment(self, dataset: List[Dict]) -> Dict[str, Any]:
        """
        Run the complete experiment.

        Args:
            dataset: List of data items

        Returns:
            Experiment results
        """
        print("\n[2/3] Running experiment...")

        results = []
        for data in tqdm(dataset, desc="  Processing"):
            result = self.run_single_trial(data)
            results.append(result)

        # Calculate statistics
        print("\n  Calculating statistics...")
        aggregated = self._aggregate_results(results)

        return aggregated

    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate results and calculate statistics."""
        # Implement aggregation logic
        return {
            "mean_accuracy": np.mean([r["accuracy"] for r in results]),
            "std_accuracy": np.std([r["accuracy"] for r in results]),
            "results": results,
        }

    def visualize_results(self, results: Dict[str, Any]):
        """Create visualizations."""
        print("\n[3/3] Creating visualizations...")

        plot_path = self.results_dir / "plot.png"
        # Implement visualization
        plot_{metric_name}(results, save_path=plot_path)

        print(f"  ✓ Saved plot to {plot_path}")

    def save_results(self, results: Dict[str, Any]):
        """Save results to JSON."""
        summary = {
            "experiment": "{Experiment Name}",
            "config": self.config,
            "results": results,
        }

        save_path = self.results_dir / "results.json"
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Saved results to {save_path}")

    def run(self):
        """Run the complete experiment pipeline."""
        dataset = self.generate_data()
        results = self.run_experiment(dataset)
        self.visualize_results(results)
        self.save_results(results)

        print("\n" + "=" * 60)
        print(f"Experiment {N} Complete!")
        print("=" * 60)
        print(f"\n✓ All results saved to: {self.results_dir}")


def main():
    """Main entry point."""
    experiment = Experiment{N}()
    experiment.run()


if __name__ == "__main__":
    main()
```

---

## Step 3: Data Generation

Add method to `src/data_generator.py` if needed:

```python
def generate_{name}_data(self, ...) -> List[Dict]:
    """
    Generate data for Experiment {N}.

    Args:
        ...: Parameters

    Returns:
        List of data dictionaries
    """
    # Implementation
    pass
```

---

## Step 4: Evaluation

Add evaluation method to `src/evaluator.py` if needed:

```python
def evaluate_{metric}(self, response: str, expected: str, ...) -> Dict:
    """
    Evaluate for Experiment {N}.

    Args:
        response: LLM response
        expected: Expected answer
        ...: Additional parameters

    Returns:
        Evaluation metrics
    """
    # Implementation
    pass
```

---

## Step 5: Visualization

Add to `src/utils/visualization.py` if needed:

```python
def plot_{metric_name}(results: Dict, save_path: Path):
    """
    Create visualization for Experiment {N}.

    Args:
        results: Experiment results
        save_path: Path to save plot
    """
    # Implementation using matplotlib/seaborn
    pass
```

---

## Step 6: Testing

Create `tests/test_exp{N}_{name}.py`:

```python
import pytest
from src.experiments.exp{N}_{name} import Experiment{N}

def test_experiment_init():
    """Test experiment initialization."""
    exp = Experiment{N}()
    assert exp.config is not None
    assert exp.results_dir.exists()

def test_data_generation():
    """Test data generation."""
    exp = Experiment{N}()
    data = exp.generate_data()
    assert len(data) > 0
    assert "context" in data[0]

def test_single_trial():
    """Test single trial execution."""
    exp = Experiment{N}()
    data = {...}  # Mock data
    result = exp.run_single_trial(data)
    assert "accuracy" in result
```

---

## Step 7: Documentation

Update `docs/RESULTS.md`:

```markdown
## Experiment {N}: {Experiment Name}

### Objective
What phenomenon or hypothesis does this experiment test?

### Methodology
- Parameter 1: value
- Parameter 2: value
- Iterations: N

### Results

| Metric | Value |
|--------|-------|
| ... | ... |

### Visualization

![Plot](../results/exp{N}/plot.png)

### Analysis

Key findings and interpretation.

### Conclusions

1. Finding 1
2. Finding 2
```

---

## Step 8: Integration Checklist

- [ ] Configuration added to `config.py`
- [ ] Experiment script created in `src/experiments/`
- [ ] Data generation implemented (if needed)
- [ ] Evaluation method implemented (if needed)
- [ ] Visualization function implemented (if needed)
- [ ] Unit tests created in `tests/`
- [ ] Documentation added to `docs/RESULTS.md`
- [ ] Experiment runs successfully
- [ ] Results saved to `results/exp{N}/`
- [ ] Visualization generated
- [ ] Cost analysis updated in `COST_ANALYSIS.md`

---

## Example: Adding Experiment 5

```python
# config.py
EXP5_CONFIG = {
    "num_samples": 100,
    "context_length": 2000,
    "metric": "perplexity",
}

# src/experiments/exp5_perplexity_test.py
class Experiment5:
    def __init__(self):
        self.config = EXP5_CONFIG
        # ...

# Usage
python src/experiments/exp5_perplexity_test.py
```

---

## Cost Estimation Template

When adding a new experiment, estimate:

```python
# Cost estimation
TOKENS_PER_CALL = estimated_context + estimated_response
NUM_CALLS = iterations * conditions
TOTAL_TOKENS = TOKENS_PER_CALL * NUM_CALLS

# Costs
COST_OLLAMA = 0
COST_GPT35 = TOTAL_TOKENS * 0.0005 / 1000
COST_GPT4 = TOTAL_TOKENS * 0.03 / 1000
```

Document in `COST_ANALYSIS.md`.

---

## Quality Standards

- **Type Hints**: All functions have type annotations
- **Docstrings**: NumPy-style docstrings for all public methods
- **Error Handling**: Try-except with meaningful messages
- **Configuration**: No magic numbers, all in config
- **Testing**: Minimum 80% coverage
- **Documentation**: Update all relevant docs

---

## Best Practices

1. **Modularity**: Keep experiments independent
2. **Reusability**: Use existing components (LLM, evaluator, visualizer)
3. **Reproducibility**: Always set seed=42
4. **Logging**: Use print statements for progress
5. **Validation**: Test on small data first
6. **Documentation**: Explain WHY, not just WHAT

---

## Support

For questions or issues:
1. Check existing experiments as examples
2. Review documentation in `docs/`
3. Refer to ADRs in `docs/architecture/decisions/`

---

**Last Updated**: December 10, 2025
