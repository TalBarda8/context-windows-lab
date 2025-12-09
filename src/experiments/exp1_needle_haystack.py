"""
Experiment 1: Needle in Haystack (Lost in the Middle)

This experiment demonstrates that LLMs struggle to retrieve information
from the middle of long contexts, exhibiting the "Lost in the Middle" phenomenon.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from typing import List, Dict, Any
import json
from tqdm import tqdm
import numpy as np

from config import (
    EXP1_CONFIG,
    EXP1_RESULTS_DIR,
    NEEDLE_HAYSTACK_DIR,
)
from data_generator import DataGenerator
from llm_interface import create_llm_interface
from evaluator import create_evaluator
from utils.visualization import plot_accuracy_by_position


class NeedleHaystackExperiment:
    """Experiment to test Lost in the Middle phenomenon."""

    def __init__(self):
        """Initialize experiment."""
        self.config = EXP1_CONFIG
        self.results_dir = EXP1_RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.data_generator = DataGenerator()
        self.llm = create_llm_interface()
        self.evaluator = create_evaluator(use_embeddings=True)

        print("="*60)
        print("Experiment 1: Needle in Haystack")
        print("="*60)

    def generate_data(self) -> List[Dict[str, str]]:
        """
        Generate experimental data.

        Returns:
            List of document dictionaries
        """
        print("\n[1/3] Generating synthetic documents...")

        dataset = self.data_generator.generate_needle_haystack_dataset(
            num_docs=self.config["iterations_per_position"],
            words_per_doc=self.config["words_per_document"],
            num_haystack_docs=self.config["num_haystack_docs"],
            save_path=NEEDLE_HAYSTACK_DIR / "dataset.json"
        )

        print(f"  ✓ Generated {len(dataset)} documents")

        return dataset

    def run_single_trial(self, document: str, secret_value: str,
                        position: str) -> Dict[str, Any]:
        """
        Run a single trial.

        Args:
            document: Document text
            secret_value: Expected secret value
            position: Position of fact

        Returns:
            Trial results
        """
        # Query the LLM
        query = self.config["query_template"]

        llm_result = self.llm.query(
            prompt=query,
            context=document
        )

        # Evaluate response
        eval_result = self.evaluator.evaluate_needle_haystack(
            response=llm_result["response"],
            secret_value=secret_value,
            position=position,
        )

        # Combine results
        result = {
            "position": position,
            "secret_value": secret_value,
            "response": llm_result["response"],
            "latency": llm_result["latency"],
            **eval_result,
        }

        return result

    def run_experiment(self, dataset: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Run the complete experiment.

        Args:
            dataset: List of documents

        Returns:
            Experiment results
        """
        print("\n[2/3] Running experiment trials...")

        results_by_position = {
            "start": [],
            "middle": [],
            "end": [],
        }

        # Run trials with progress bar
        for doc_data in tqdm(dataset, desc="  Processing documents"):
            result = self.run_single_trial(
                document=doc_data["document"],
                secret_value=doc_data["secret_value"],
                position=doc_data["position"],
            )

            results_by_position[doc_data["position"]].append(result)

        # Calculate aggregate statistics
        print("\n  Calculating statistics...")

        aggregated_results = {}

        for position in ["start", "middle", "end"]:
            trials = results_by_position[position]

            # Extract accuracy scores
            accuracy_scores = [t["overall_score"] for t in trials]
            correct_count = sum(1 for t in trials if t["correct"])

            aggregated_results[position] = {
                "accuracy_scores": accuracy_scores,
                "mean_accuracy": sum(accuracy_scores) / len(accuracy_scores),
                "correct_count": correct_count,
                "total_count": len(trials),
                "success_rate": correct_count / len(trials),
                "trials": trials,
            }

            print(f"    {position.capitalize()}: "
                  f"{aggregated_results[position]['mean_accuracy']:.3f} "
                  f"({correct_count}/{len(trials)})")

        return aggregated_results

    def visualize_results(self, results: Dict[str, Any]):
        """
        Create visualizations.

        Args:
            results: Experiment results
        """
        print("\n[3/3] Creating visualizations...")

        # Prepare data for plotting
        plot_data = {
            position: results[position]["accuracy_scores"]
            for position in ["start", "middle", "end"]
        }

        # Create plot
        plot_path = self.results_dir / "accuracy_by_position.png"

        plot_accuracy_by_position(
            results=plot_data,
            save_path=plot_path,
            title="Experiment 1: Accuracy by Fact Position (Lost in the Middle)"
        )

        print(f"  ✓ Saved plot to {plot_path}")

    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        else:
            return obj

    def save_results(self, results: Dict[str, Any]):
        """
        Save results to JSON.

        Args:
            results: Experiment results
        """
        # Prepare summary
        summary = {
            "experiment": "Needle in Haystack",
            "config": self.config,
            "results_by_position": {
                position: {
                    "mean_accuracy": float(data["mean_accuracy"]),
                    "success_rate": float(data["success_rate"]),
                    "correct_count": int(data["correct_count"]),
                    "total_count": int(data["total_count"]),
                }
                for position, data in results.items()
            },
            "detailed_results": self._convert_to_json_serializable(results),
        }

        # Save to JSON
        save_path = self.results_dir / "results.json"

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Saved results to {save_path}")

    def run(self):
        """Run the complete experiment pipeline."""
        # Generate data
        dataset = self.generate_data()

        # Run experiment
        results = self.run_experiment(dataset)

        # Visualize
        self.visualize_results(results)

        # Save results
        self.save_results(results)

        # Print summary
        print("\n" + "="*60)
        print("Experiment 1 Complete!")
        print("="*60)

        print("\nKey Findings:")
        for position in ["start", "middle", "end"]:
            acc = results[position]["mean_accuracy"]
            rate = results[position]["success_rate"]
            print(f"  {position.capitalize()}: {acc:.3f} accuracy, "
                  f"{rate:.1%} success rate")

        print("\n✓ All results saved to:", self.results_dir)


def main():
    """Main entry point."""
    experiment = NeedleHaystackExperiment()
    experiment.run()


if __name__ == "__main__":
    main()
