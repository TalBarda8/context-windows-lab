"""
Evaluator module for measuring experiment accuracy and performance.

This module provides high-level evaluation functions that combine
LLM responses with metrics to assess experiment outcomes.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
import json

# Import utilities
from utils.metrics import (
    extract_answer_from_response,
    calculate_accuracy_metrics,
    calculate_statistics,
    perform_t_test,
)

# Import LLM interface
from llm_interface import LLMInterface, EmbeddingInterface


class Evaluator:
    """Evaluator for measuring experiment accuracy."""

    def __init__(self,
                 llm_interface: Optional[LLMInterface] = None,
                 embedding_interface: Optional[EmbeddingInterface] = None):
        """
        Initialize evaluator.

        Args:
            llm_interface: LLM interface for generating embeddings
            embedding_interface: Embedding interface for semantic similarity
        """
        self.llm = llm_interface
        self.embedding = embedding_interface

    def evaluate_response(self,
                         response: str,
                         expected_answer: str,
                         keywords: Optional[List[str]] = None,
                         extract_pattern: Optional[str] = None
                         ) -> Dict[str, float]:
        """
        Evaluate a single response against expected answer.

        Args:
            response: LLM response to evaluate
            expected_answer: Expected correct answer
            keywords: Optional keywords that should be present
            extract_pattern: Optional regex pattern to extract answer

        Returns:
            Dictionary of accuracy metrics
        """
        # Extract answer from response if pattern provided
        if extract_pattern:
            extracted = extract_answer_from_response(response, extract_pattern)
        else:
            extracted = extract_answer_from_response(response)

        # Generate embeddings if embedding interface available
        predicted_embedding = None
        expected_embedding = None

        if self.embedding:
            try:
                predicted_embedding = self.embedding.embed_text(extracted)
                expected_embedding = self.embedding.embed_text(expected_answer)
            except Exception as e:
                print(f"Warning: Could not generate embeddings: {e}")

        # Calculate all metrics
        metrics = calculate_accuracy_metrics(
            predicted=extracted,
            expected=expected_answer,
            predicted_embedding=predicted_embedding,
            expected_embedding=expected_embedding,
            keywords=keywords,
        )

        # Add extracted answer for debugging
        metrics["extracted_answer"] = extracted

        return metrics

    def evaluate_multiple_responses(self,
                                    responses: List[str],
                                    expected_answer: str,
                                    keywords: Optional[List[str]] = None
                                    ) -> Dict[str, Any]:
        """
        Evaluate multiple responses (e.g., from repeated trials).

        Args:
            responses: List of LLM responses
            expected_answer: Expected correct answer
            keywords: Optional keywords

        Returns:
            Aggregated metrics with statistics
        """
        all_metrics = []

        for response in responses:
            metrics = self.evaluate_response(
                response=response,
                expected_answer=expected_answer,
                keywords=keywords,
            )
            all_metrics.append(metrics)

        # Extract each metric type
        metric_names = [k for k in all_metrics[0].keys()
                       if isinstance(all_metrics[0][k], (int, float))]

        aggregated = {}

        for metric_name in metric_names:
            values = [m[metric_name] for m in all_metrics]
            aggregated[metric_name] = calculate_statistics(values)

        return aggregated

    def save_results(self, results: Dict[str, Any],
                    save_path: Path,
                    pretty: bool = True):
        """
        Save evaluation results to JSON.

        Args:
            results: Results dictionary
            save_path: Path to save file
            pretty: Whether to pretty-print JSON
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(results, f, indent=2, ensure_ascii=False)
            else:
                json.dump(results, f, ensure_ascii=False)

        print(f"Results saved to {save_path}")

    def load_results(self, load_path: Path) -> Dict[str, Any]:
        """
        Load evaluation results from JSON.

        Args:
            load_path: Path to load file

        Returns:
            Results dictionary
        """
        with open(load_path, 'r', encoding='utf-8') as f:
            results = json.load(f)

        return results


class ExperimentEvaluator(Evaluator):
    """Extended evaluator with experiment-specific methods."""

    def evaluate_needle_haystack(self,
                                 response: str,
                                 secret_value: str,
                                 position: str) -> Dict[str, Any]:
        """
        Evaluate Experiment 1 (Needle in Haystack) response.

        Args:
            response: LLM response
            secret_value: Expected secret password
            position: Position of fact in document

        Returns:
            Evaluation metrics
        """
        # Keywords to look for
        keywords = [secret_value]

        metrics = self.evaluate_response(
            response=response,
            expected_answer=secret_value,
            keywords=keywords,
        )

        metrics["position"] = position
        metrics["correct"] = metrics["exact_match"] > 0.5 or \
                            metrics["partial_match"] > 0.8

        return metrics

    def evaluate_context_size(self,
                             response: str,
                             revenue_value: str,
                             num_docs: int,
                             latency: float) -> Dict[str, Any]:
        """
        Evaluate Experiment 2 (Context Size) response.

        Args:
            response: LLM response
            revenue_value: Expected revenue value
            num_docs: Number of documents in context
            latency: Response latency

        Returns:
            Evaluation metrics
        """
        metrics = self.evaluate_response(
            response=response,
            expected_answer=revenue_value,
            keywords=[revenue_value],
        )

        metrics["num_docs"] = num_docs
        metrics["latency"] = latency
        metrics["correct"] = metrics["partial_match"] > 0.6

        return metrics

    def evaluate_rag_response(self,
                             response: str,
                             expected_answer: str,
                             latency: float,
                             method: str = "rag") -> Dict[str, Any]:
        """
        Evaluate Experiment 3 (RAG) response.

        Args:
            response: LLM response
            expected_answer: Expected answer
            latency: Response latency
            method: "rag" or "full_context"

        Returns:
            Evaluation metrics
        """
        # For Hebrew text, extract key terms
        keywords = expected_answer.split()[:3]  # First 3 words as keywords

        metrics = self.evaluate_response(
            response=response,
            expected_answer=expected_answer,
            keywords=keywords,
        )

        metrics["method"] = method
        metrics["latency"] = latency
        metrics["correct"] = metrics["overall_score"] > 0.5

        return metrics

    def evaluate_strategy_step(self,
                               response: str,
                               expected_output: str,
                               strategy: str,
                               step: int,
                               context_size: int) -> Dict[str, Any]:
        """
        Evaluate Experiment 4 (Strategies) single step.

        Args:
            response: LLM response
            expected_output: Expected output for this step
            strategy: Strategy name (select/compress/write)
            step: Step number
            context_size: Current context size in tokens

        Returns:
            Evaluation metrics
        """
        metrics = self.evaluate_response(
            response=response,
            expected_answer=expected_output,
        )

        metrics["strategy"] = strategy
        metrics["step"] = step
        metrics["context_size"] = context_size
        metrics["correct"] = metrics["overall_score"] > 0.5

        return metrics

    def compare_groups(self,
                      group1_results: List[Dict[str, Any]],
                      group2_results: List[Dict[str, Any]],
                      metric_name: str = "overall_score") -> Dict[str, Any]:
        """
        Perform statistical comparison between two groups.

        Args:
            group1_results: Results from group 1
            group2_results: Results from group 2
            metric_name: Metric to compare

        Returns:
            Statistical comparison results
        """
        group1_values = [r[metric_name] for r in group1_results
                        if metric_name in r]
        group2_values = [r[metric_name] for r in group2_results
                        if metric_name in r]

        if not group1_values or not group2_values:
            return {
                "error": "Insufficient data for comparison"
            }

        # Perform t-test
        test_results = perform_t_test(group1_values, group2_values)

        # Add descriptive statistics
        test_results["group1_mean"] = float(np.mean(group1_values))
        test_results["group2_mean"] = float(np.mean(group2_values))
        test_results["group1_std"] = float(np.std(group1_values))
        test_results["group2_std"] = float(np.std(group2_values))
        test_results["difference"] = test_results["group1_mean"] - \
                                     test_results["group2_mean"]

        return test_results


def create_evaluator(use_embeddings: bool = True) -> ExperimentEvaluator:
    """
    Factory function to create evaluator.

    Args:
        use_embeddings: Whether to initialize embedding interface

    Returns:
        Initialized experiment evaluator
    """
    llm_interface = None  # We don't need LLM in evaluator

    embedding_interface = None
    if use_embeddings:
        try:
            embedding_interface = EmbeddingInterface()
        except Exception as e:
            print(f"Warning: Could not initialize embeddings: {e}")
            embedding_interface = None

    return ExperimentEvaluator(
        llm_interface=llm_interface,
        embedding_interface=embedding_interface,
    )


if __name__ == "__main__":
    # Test the evaluator
    print("Testing Evaluator...")

    evaluator = create_evaluator(use_embeddings=True)

    # Test simple evaluation
    response = "The secret password is ABC123XYZ."
    expected = "ABC123XYZ"

    metrics = evaluator.evaluate_response(response, expected, keywords=["ABC123XYZ"])

    print(f"Metrics: {metrics}")
    print(f"Overall score: {metrics['overall_score']:.2f}")
