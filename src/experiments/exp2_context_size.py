"""
Experiment 2: Context Window Size Impact

This experiment measures how increasing context window size affects
accuracy, latency, and token consumption.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from typing import List, Dict, Any
import json
import numpy as np
from tqdm import tqdm
import numpy as np

from config import (
    EXP2_CONFIG,
    EXP2_RESULTS_DIR,
    CONTEXT_SIZE_DIR,
)
from data_generator import DataGenerator
from llm_interface import create_llm_interface
from evaluator import create_evaluator
from utils.visualization import plot_context_size_impact


class ContextSizeExperiment:
    """Experiment to measure context size impact."""

    def __init__(self):
        """Initialize experiment."""
        self.config = EXP2_CONFIG
        self.results_dir = EXP2_RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.data_generator = DataGenerator()
        self.llm = create_llm_interface()
        self.evaluator = create_evaluator(use_embeddings=True)

        print("="*60)
        print("Experiment 2: Context Window Size Impact")
        print("="*60)

    def generate_data(self) -> Dict[int, List[Dict[str, str]]]:
        """
        Generate experimental data for different context sizes.

        Returns:
            Dictionary mapping document count to list of documents
        """
        print("\n[1/3] Generating datasets for different context sizes...")

        datasets = self.data_generator.generate_context_size_dataset(
            doc_counts=self.config["document_counts"],
            words_per_doc=self.config["words_per_document"],
            save_dir=CONTEXT_SIZE_DIR,
        )

        total_docs = sum(len(docs) for docs in datasets.values())
        print(f"  ✓ Generated {total_docs} documents across "
              f"{len(datasets)} different sizes")

        return datasets

    def run_single_trial(self, documents: List[Dict[str, str]],
                        num_docs: int) -> Dict[str, Any]:
        """
        Run a single trial with specified number of documents.

        Args:
            documents: List of documents
            num_docs: Number of documents in this trial

        Returns:
            Trial results
        """
        # Find the document with the revenue fact
        revenue_value = None
        for doc in documents:
            if doc.get("revenue_value"):
                revenue_value = doc["revenue_value"]
                break

        if not revenue_value:
            print(f"  Warning: No revenue value found in documents")
            return None

        # Concatenate all documents as context
        context = "\n\n".join([doc["document"] for doc in documents])

        # Count tokens
        tokens_used = self.llm.count_tokens(context)

        # Query the LLM
        query = self.config["question_template"]

        llm_result = self.llm.query(
            prompt=query,
            context=context
        )

        # Evaluate response
        eval_result = self.evaluator.evaluate_context_size(
            response=llm_result["response"],
            revenue_value=revenue_value,
            num_docs=num_docs,
            latency=llm_result["latency"],
        )

        # Combine results
        result = {
            "num_docs": num_docs,
            "tokens_used": tokens_used,
            "revenue_value": revenue_value,
            "response": llm_result["response"],
            **eval_result,
        }

        return result

    def run_experiment(self,
                      datasets: Dict[int, List[Dict[str, str]]]
                      ) -> List[Dict[str, Any]]:
        """
        Run the complete experiment.

        Args:
            datasets: Dictionary of datasets by size

        Returns:
            List of results for each size
        """
        print("\n[2/3] Running experiment trials...")

        results = []

        for num_docs in tqdm(self.config["document_counts"],
                            desc="  Testing context sizes"):

            documents = datasets[num_docs]

            # Run multiple iterations
            size_results = []

            for iteration in range(self.config["iterations_per_size"]):
                result = self.run_single_trial(documents, num_docs)

                if result:
                    size_results.append(result)

            # Aggregate results for this size
            if size_results:
                accuracy_scores = [r["overall_score"] for r in size_results]
                latencies = [r["latency"] for r in size_results]
                tokens = [r["tokens_used"] for r in size_results]

                aggregated = {
                    "num_docs": num_docs,
                    "accuracy_mean": float(np.mean(accuracy_scores)),
                    "accuracy_std": float(np.std(accuracy_scores)),
                    "latency_mean": float(np.mean(latencies)),
                    "latency_std": float(np.std(latencies)),
                    "tokens_used": int(np.mean(tokens)),
                    "iterations": len(size_results),
                    "trials": size_results,
                }

                results.append(aggregated)

                print(f"    {num_docs} docs: "
                      f"acc={aggregated['accuracy_mean']:.3f}, "
                      f"latency={aggregated['latency_mean']:.2f}s, "
                      f"tokens={aggregated['tokens_used']}")

        return results

    def visualize_results(self, results: List[Dict[str, Any]]):
        """
        Create visualizations.

        Args:
            results: Experiment results
        """
        print("\n[3/3] Creating visualizations...")

        # Create plot
        plot_path = self.results_dir / "context_size_impact.png"

        plot_context_size_impact(
            results=results,
            save_path=plot_path,
            title="Experiment 2: Context Window Size Impact on Performance"
        )

        print(f"  ✓ Saved plot to {plot_path}")

    def save_results(self, results: List[Dict[str, Any]]):
        """
        Save results to JSON.

        Args:
            results: Experiment results
        """
        # Prepare summary
        summary = {
            "experiment": "Context Window Size Impact",
            "config": self.config,
            "results_summary": [
                {
                    "num_docs": r["num_docs"],
                    "accuracy_mean": r["accuracy_mean"],
                    "accuracy_std": r["accuracy_std"],
                    "latency_mean": r["latency_mean"],
                    "latency_std": r["latency_std"],
                    "tokens_used": r["tokens_used"],
                }
                for r in results
            ],
            "detailed_results": results,
        }

        # Save to JSON
        save_path = self.results_dir / "results.json"

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Saved results to {save_path}")

    def run(self):
        """Run the complete experiment pipeline."""
        # Generate data
        datasets = self.generate_data()

        # Run experiment
        results = self.run_experiment(datasets)

        # Visualize
        self.visualize_results(results)

        # Save results
        self.save_results(results)

        # Print summary
        print("\n" + "="*60)
        print("Experiment 2 Complete!")
        print("="*60)

        print("\nKey Findings:")
        print(f"  Accuracy degradation: "
              f"{results[0]['accuracy_mean']:.3f} → "
              f"{results[-1]['accuracy_mean']:.3f}")

        print(f"  Latency increase: "
              f"{results[0]['latency_mean']:.2f}s → "
              f"{results[-1]['latency_mean']:.2f}s")

        print(f"  Token growth: "
              f"{results[0]['tokens_used']} → "
              f"{results[-1]['tokens_used']}")

        print("\n✓ All results saved to:", self.results_dir)


def main():
    """Main entry point."""
    experiment = ContextSizeExperiment()
    experiment.run()


if __name__ == "__main__":
    main()
