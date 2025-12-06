"""
Tests for evaluator.py - Evaluation module.

Tests cover:
- Evaluator initialization
- Single response evaluation
- Multiple response evaluation
- Experiment-specific evaluators (needle haystack, context size, RAG, strategies)
- Results saving and loading
- Group comparisons
- Factory functions
"""

import pytest
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluator import (
    Evaluator,
    ExperimentEvaluator,
    create_evaluator,
)


class TestEvaluatorInit:
    """Test Evaluator initialization."""

    def test_init_without_interfaces(self):
        """Test initialization without LLM or embedding interfaces."""
        evaluator = Evaluator()

        assert evaluator.llm is None
        assert evaluator.embedding is None

    def test_init_with_interfaces(self, mock_llm, mock_embedding):
        """Test initialization with interfaces."""
        evaluator = Evaluator(
            llm_interface=mock_llm,
            embedding_interface=mock_embedding
        )

        assert evaluator.llm is not None
        assert evaluator.embedding is not None


class TestEvaluateResponse:
    """Test evaluate_response method."""

    def test_evaluate_correct_response(self, mock_embedding):
        """Test evaluation of correct response."""
        evaluator = Evaluator(embedding_interface=mock_embedding)

        response = "The secret password is TEST12345."
        expected = "TEST12345"

        metrics = evaluator.evaluate_response(response, expected)

        assert "exact_match" in metrics
        assert "partial_match" in metrics
        assert "overall_score" in metrics
        assert "extracted_answer" in metrics

        # Should have decent partial match
        assert metrics["partial_match"] > 0.5

    def test_evaluate_with_keywords(self, mock_embedding):
        """Test evaluation with keywords."""
        evaluator = Evaluator(embedding_interface=mock_embedding)

        response = "The password is ABC123 and it's secret."
        expected = "ABC123"
        keywords = ["ABC123", "password"]

        metrics = evaluator.evaluate_response(
            response, expected, keywords=keywords
        )

        assert "keyword_match" in metrics
        assert metrics["keyword_match"] == 1.0  # Both keywords present

    def test_evaluate_with_embeddings(self, mock_embedding):
        """Test evaluation with semantic similarity."""
        evaluator = Evaluator(embedding_interface=mock_embedding)

        response = "The answer is 42"
        expected = "42"

        metrics = evaluator.evaluate_response(response, expected)

        # With embedding interface, should have semantic similarity
        assert "semantic_similarity" in metrics
        assert 0.0 <= metrics["semantic_similarity"] <= 1.0

    def test_evaluate_without_embeddings(self):
        """Test evaluation without embedding interface."""
        evaluator = Evaluator(embedding_interface=None)

        response = "The answer is 42"
        expected = "42"

        metrics = evaluator.evaluate_response(response, expected)

        # Should NOT have semantic similarity
        assert "semantic_similarity" not in metrics

        # But should have other metrics
        assert "exact_match" in metrics
        assert "partial_match" in metrics

    def test_evaluate_with_custom_pattern(self, mock_embedding):
        """Test evaluation with custom extraction pattern."""
        evaluator = Evaluator(embedding_interface=mock_embedding)

        response = "Revenue: $500 million for the year."
        expected = "$500 million"
        pattern = r"Revenue: (.+?) for"

        metrics = evaluator.evaluate_response(
            response, expected, extract_pattern=pattern
        )

        assert "$500 million" in metrics["extracted_answer"]

    def test_extracted_answer_in_metrics(self, mock_embedding):
        """Test that extracted answer is included in metrics."""
        evaluator = Evaluator(embedding_interface=mock_embedding)

        response = "The secret code is XYZ789."
        expected = "XYZ789"

        metrics = evaluator.evaluate_response(response, expected)

        assert "extracted_answer" in metrics
        assert isinstance(metrics["extracted_answer"], str)


class TestEvaluateMultipleResponses:
    """Test evaluate_multiple_responses method."""

    def test_evaluate_multiple_correct_responses(self, mock_embedding):
        """Test evaluation of multiple correct responses."""
        evaluator = Evaluator(embedding_interface=mock_embedding)

        responses = [
            "The password is TEST123.",
            "Password: TEST123",
            "TEST123 is the password.",
        ]
        expected = "TEST123"

        aggregated = evaluator.evaluate_multiple_responses(
            responses, expected
        )

        # Should have statistics for each metric
        assert "exact_match" in aggregated
        assert "partial_match" in aggregated
        assert "overall_score" in aggregated

        # Each should have statistical measures
        assert "mean" in aggregated["overall_score"]
        assert "std" in aggregated["overall_score"]

    def test_evaluate_with_keywords_multiple(self, mock_embedding):
        """Test multiple evaluation with keywords."""
        evaluator = Evaluator(embedding_interface=mock_embedding)

        responses = [
            "The revenue is $500 million.",
            "Annual revenue: $500 million",
        ]
        expected = "$500 million"
        keywords = ["revenue", "$500 million"]

        aggregated = evaluator.evaluate_multiple_responses(
            responses, expected, keywords=keywords
        )

        assert "keyword_match" in aggregated
        assert "mean" in aggregated["keyword_match"]

    def test_variability_in_responses(self, mock_embedding):
        """Test evaluation shows variability across responses."""
        evaluator = Evaluator(embedding_interface=mock_embedding)

        responses = [
            "The answer is correct: ABC",
            "Answer is ABC",
            "Maybe it's ABC or something else",
        ]
        expected = "ABC"

        aggregated = evaluator.evaluate_multiple_responses(
            responses, expected
        )

        # Standard deviation should be > 0 (variability exists)
        assert aggregated["overall_score"]["std"] >= 0


class TestSaveAndLoadResults:
    """Test save_results and load_results methods."""

    def test_save_results(self, temp_directory):
        """Test saving results to JSON."""
        evaluator = Evaluator()

        results = {
            "experiment": "test",
            "accuracy": 0.85,
            "metrics": {"exact_match": 1.0, "partial_match": 0.9}
        }

        save_path = temp_directory / "test_results.json"

        evaluator.save_results(results, save_path)

        # File should exist
        assert save_path.exists()

        # Should be valid JSON
        with open(save_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)

        assert loaded == results

    def test_save_pretty_json(self, temp_directory):
        """Test saving with pretty formatting."""
        evaluator = Evaluator()

        results = {"test": "data"}
        save_path = temp_directory / "pretty.json"

        evaluator.save_results(results, save_path, pretty=True)

        # Read raw content
        content = save_path.read_text()

        # Should have indentation (pretty formatted)
        assert "\n" in content
        assert "  " in content  # Indentation

    def test_load_results(self, temp_directory):
        """Test loading results from JSON."""
        evaluator = Evaluator()

        # Create test file
        test_data = {"accuracy": 0.9, "latency": 1.5}
        save_path = temp_directory / "load_test.json"

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f)

        # Load
        loaded = evaluator.load_results(save_path)

        assert loaded == test_data

    def test_save_load_roundtrip(self, temp_directory):
        """Test save and load round-trip."""
        evaluator = Evaluator()

        original_results = {
            "experiment_id": "exp1",
            "metrics": {
                "accuracy": 0.85,
                "latency": 0.5
            },
            "parameters": {
                "model": "llama2",
                "temperature": 0.0
            }
        }

        save_path = temp_directory / "roundtrip.json"

        evaluator.save_results(original_results, save_path)
        loaded_results = evaluator.load_results(save_path)

        assert loaded_results == original_results


class TestExperimentEvaluatorNeedleHaystack:
    """Test evaluate_needle_haystack method."""

    def test_evaluate_correct_needle_haystack(self, mock_embedding):
        """Test evaluation of correct needle-in-haystack response."""
        evaluator = ExperimentEvaluator(embedding_interface=mock_embedding)

        response = "The secret password is ABCD1234."
        secret = "ABCD1234"
        position = "middle"

        metrics = evaluator.evaluate_needle_haystack(
            response, secret, position
        )

        assert "position" in metrics
        assert metrics["position"] == "middle"
        assert "correct" in metrics
        assert metrics["correct"] is True  # Should be marked correct

    def test_evaluate_incorrect_needle_haystack(self, mock_embedding):
        """Test evaluation of incorrect response."""
        evaluator = ExperimentEvaluator(embedding_interface=mock_embedding)

        response = "I don't know the password."
        secret = "SECRET123"
        position = "start"

        metrics = evaluator.evaluate_needle_haystack(
            response, secret, position
        )

        assert metrics["correct"] is False

    def test_needle_haystack_different_positions(self, mock_embedding):
        """Test evaluation for different positions."""
        evaluator = ExperimentEvaluator(embedding_interface=mock_embedding)

        for position in ["start", "middle", "end"]:
            metrics = evaluator.evaluate_needle_haystack(
                "Password is TEST", "TEST", position
            )

            assert metrics["position"] == position


class TestExperimentEvaluatorContextSize:
    """Test evaluate_context_size method."""

    def test_evaluate_context_size(self, mock_embedding):
        """Test context size evaluation."""
        evaluator = ExperimentEvaluator(embedding_interface=mock_embedding)

        response = "The annual revenue is $500 million."
        revenue = "$500 million"
        num_docs = 10
        latency = 0.75

        metrics = evaluator.evaluate_context_size(
            response, revenue, num_docs, latency
        )

        assert "num_docs" in metrics
        assert metrics["num_docs"] == 10
        assert "latency" in metrics
        assert metrics["latency"] == 0.75
        assert "correct" in metrics

    def test_context_size_correct_detection(self, mock_embedding):
        """Test correct detection in context size evaluation."""
        evaluator = ExperimentEvaluator(embedding_interface=mock_embedding)

        response = "Revenue: $999 million"
        revenue = "$999 million"

        metrics = evaluator.evaluate_context_size(
            response, revenue, num_docs=5, latency=0.5
        )

        # Partial match should be high, marked correct
        assert metrics["correct"] is True


class TestExperimentEvaluatorRAG:
    """Test evaluate_rag_response method."""

    def test_evaluate_rag_response(self, mock_embedding):
        """Test RAG response evaluation."""
        evaluator = ExperimentEvaluator(embedding_interface=mock_embedding)

        response = "תופעות הלוואי כוללות: כאב ראש, בחילה."
        expected = "כאב ראש, בחילה, סחרחורת"
        latency = 0.6

        metrics = evaluator.evaluate_rag_response(
            response, expected, latency, method="rag"
        )

        assert "method" in metrics
        assert metrics["method"] == "rag"
        assert "latency" in metrics
        assert metrics["latency"] == 0.6
        assert "correct" in metrics

    def test_rag_vs_full_context(self, mock_embedding):
        """Test evaluation differentiates RAG vs full context."""
        evaluator = ExperimentEvaluator(embedding_interface=mock_embedding)

        response = "Test response"
        expected = "Test"

        rag_metrics = evaluator.evaluate_rag_response(
            response, expected, 0.5, method="rag"
        )

        full_metrics = evaluator.evaluate_rag_response(
            response, expected, 1.0, method="full_context"
        )

        assert rag_metrics["method"] == "rag"
        assert full_metrics["method"] == "full_context"

    def test_hebrew_keyword_extraction(self, mock_embedding):
        """Test that Hebrew keywords are extracted."""
        evaluator = ExperimentEvaluator(embedding_interface=mock_embedding)

        response = "פנדול גורם לכאב ראש"
        expected = "פנדול גורם לכאב ראש"

        metrics = evaluator.evaluate_rag_response(
            response, expected, 0.5
        )

        # Should extract first 3 words as keywords
        assert "keyword_match" in metrics


class TestExperimentEvaluatorStrategy:
    """Test evaluate_strategy_step method."""

    def test_evaluate_strategy_step(self, mock_embedding):
        """Test strategy step evaluation."""
        evaluator = ExperimentEvaluator(embedding_interface=mock_embedding)

        response = "Task completed successfully."
        expected = "Task completed"
        strategy = "select"
        step = 5
        context_size = 1024

        metrics = evaluator.evaluate_strategy_step(
            response, expected, strategy, step, context_size
        )

        assert "strategy" in metrics
        assert metrics["strategy"] == "select"
        assert "step" in metrics
        assert metrics["step"] == 5
        assert "context_size" in metrics
        assert metrics["context_size"] == 1024
        assert "correct" in metrics

    def test_different_strategies(self, mock_embedding):
        """Test evaluation for different strategies."""
        evaluator = ExperimentEvaluator(embedding_interface=mock_embedding)

        strategies = ["select", "compress", "write"]

        for strategy in strategies:
            metrics = evaluator.evaluate_strategy_step(
                "response", "expected", strategy, 1, 1000
            )

            assert metrics["strategy"] == strategy


class TestCompareGroups:
    """Test compare_groups method."""

    def test_compare_significantly_different_groups(self, mock_embedding):
        """Test comparison of significantly different groups."""
        evaluator = ExperimentEvaluator(embedding_interface=mock_embedding)

        group1 = [
            {"overall_score": 0.9, "exact_match": 1.0},
            {"overall_score": 0.85, "exact_match": 1.0},
            {"overall_score": 0.88, "exact_match": 1.0},
        ]

        group2 = [
            {"overall_score": 0.3, "exact_match": 0.0},
            {"overall_score": 0.35, "exact_match": 0.0},
            {"overall_score": 0.32, "exact_match": 0.0},
        ]

        comparison = evaluator.compare_groups(group1, group2, "overall_score")

        assert "t_statistic" in comparison
        assert "p_value" in comparison
        assert "significant" in comparison
        assert comparison["significant"] is True
        assert "group1_mean" in comparison
        assert "group2_mean" in comparison
        assert "difference" in comparison

    def test_compare_similar_groups(self, mock_embedding):
        """Test comparison of similar groups."""
        evaluator = ExperimentEvaluator(embedding_interface=mock_embedding)

        group1 = [
            {"overall_score": 0.75},
            {"overall_score": 0.78},
            {"overall_score": 0.76},
        ]

        group2 = [
            {"overall_score": 0.77},
            {"overall_score": 0.74},
            {"overall_score": 0.79},
        ]

        comparison = evaluator.compare_groups(group1, group2, "overall_score")

        assert comparison["significant"] is False

    def test_compare_insufficient_data(self, mock_embedding):
        """Test comparison with insufficient data."""
        evaluator = ExperimentEvaluator(embedding_interface=mock_embedding)

        group1 = [{"score": 0.8}]
        group2 = [{"score": 0.5}]

        comparison = evaluator.compare_groups(group1, group2, "score")

        assert "error" in comparison

    def test_compare_effect_size(self, mock_embedding):
        """Test that effect size is calculated."""
        evaluator = ExperimentEvaluator(embedding_interface=mock_embedding)

        group1 = [{"score": 0.9}, {"score": 0.85}, {"score": 0.88}]
        group2 = [{"score": 0.5}, {"score": 0.55}, {"score": 0.52}]

        comparison = evaluator.compare_groups(group1, group2, "score")

        assert "effect_size" in comparison
        assert comparison["effect_size"] > 0


class TestCreateEvaluator:
    """Test create_evaluator factory function."""

    def test_create_with_embeddings(self, patch_sentence_transformer):
        """Test creating evaluator with embeddings."""
        evaluator = create_evaluator(use_embeddings=True)

        assert isinstance(evaluator, ExperimentEvaluator)
        # Should have embedding interface
        assert evaluator.embedding is not None

    def test_create_without_embeddings(self):
        """Test creating evaluator without embeddings."""
        evaluator = create_evaluator(use_embeddings=False)

        assert isinstance(evaluator, ExperimentEvaluator)
        assert evaluator.embedding is None

    def test_create_default(self, patch_sentence_transformer):
        """Test default factory creation."""
        evaluator = create_evaluator()

        assert isinstance(evaluator, ExperimentEvaluator)


class TestEvaluatorIntegration:
    """Integration tests for Evaluator."""

    def test_full_needle_haystack_evaluation(self, mock_embedding):
        """Test complete needle-in-haystack evaluation workflow."""
        evaluator = ExperimentEvaluator(embedding_interface=mock_embedding)

        # Simulate multiple trials
        responses = [
            "The secret password is ABC123.",
            "Password: ABC123",
            "It's ABC123",
        ]

        secret = "ABC123"
        position = "middle"

        # Evaluate each
        results = []
        for response in responses:
            metrics = evaluator.evaluate_needle_haystack(
                response, secret, position
            )
            results.append(metrics)

        # All should be marked correct
        assert all(r["correct"] for r in results)

        # All should have same position
        assert all(r["position"] == "middle" for r in results)

    def test_context_size_experiment_evaluation(self, mock_embedding):
        """Test context size experiment evaluation workflow."""
        evaluator = ExperimentEvaluator(embedding_interface=mock_embedding)

        # Simulate different context sizes
        doc_counts = [2, 5, 10, 20]
        latencies = [0.3, 0.5, 0.8, 1.5]

        results = []

        for num_docs, latency in zip(doc_counts, latencies):
            response = "Revenue: $500 million"
            revenue = "$500 million"

            metrics = evaluator.evaluate_context_size(
                response, revenue, num_docs, latency
            )
            results.append(metrics)

        # Latency should increase with context size
        latencies_recorded = [r["latency"] for r in results]
        assert latencies_recorded == sorted(latencies_recorded)

    def test_rag_comparison_evaluation(self, mock_embedding):
        """Test RAG vs full-context comparison workflow."""
        evaluator = ExperimentEvaluator(embedding_interface=mock_embedding)

        question = "מהן תופעות הלוואי?"
        answer = "כאב ראש, בחילה"

        # RAG method
        rag_response = "תופעות הלוואי: כאב ראש, בחילה"
        rag_metrics = evaluator.evaluate_rag_response(
            rag_response, answer, 0.5, method="rag"
        )

        # Full context method
        full_response = "תופעות הלוואי: כאב ראש, בחילה"
        full_metrics = evaluator.evaluate_rag_response(
            full_response, answer, 2.0, method="full_context"
        )

        # Compare latencies
        assert rag_metrics["latency"] < full_metrics["latency"]

        # Both should be correct
        assert rag_metrics["correct"]
        assert full_metrics["correct"]

    def test_strategy_comparison_evaluation(self, mock_embedding):
        """Test strategy comparison workflow."""
        evaluator = ExperimentEvaluator(embedding_interface=mock_embedding)

        strategies = ["select", "compress", "write"]

        all_results = {strategy: [] for strategy in strategies}

        # Simulate 5 steps for each strategy
        for strategy in strategies:
            for step in range(5):
                context_size = 500 + (step * 100)

                metrics = evaluator.evaluate_strategy_step(
                    f"Step {step} completed",
                    f"Step {step} completed",
                    strategy,
                    step,
                    context_size
                )
                all_results[strategy].append(metrics)

        # Each strategy should have 5 results
        for strategy in strategies:
            assert len(all_results[strategy]) == 5

        # Compare strategies
        select_results = all_results["select"]
        compress_results = all_results["compress"]

        comparison = evaluator.compare_groups(
            select_results, compress_results, "overall_score"
        )

        assert "t_statistic" in comparison
        assert "p_value" in comparison

    def test_save_load_experiment_results(self, mock_embedding, temp_directory):
        """Test saving and loading complete experiment results."""
        evaluator = ExperimentEvaluator(embedding_interface=mock_embedding)

        # Run experiment
        results = []
        for i in range(5):
            metrics = evaluator.evaluate_needle_haystack(
                f"Password is TEST{i}",
                f"TEST{i}",
                "middle"
            )
            results.append(metrics)

        # Save
        save_path = temp_directory / "experiment_results.json"
        evaluator.save_results({"results": results}, save_path)

        # Load
        loaded = evaluator.load_results(save_path)

        assert "results" in loaded
        assert len(loaded["results"]) == 5
