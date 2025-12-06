"""
Tests for utils/metrics.py - Metrics and statistical utilities.

Tests cover:
- Exact and partial matching
- Keyword matching
- Semantic similarity (with mock embeddings)
- Statistical functions (mean, std, CI, t-tests)
- Answer extraction from responses
- Comprehensive accuracy metrics
- Results aggregation
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.metrics import (
    exact_match,
    partial_match,
    keyword_match,
    semantic_similarity,
    calculate_mean_std,
    calculate_confidence_interval,
    calculate_statistics,
    perform_t_test,
    extract_answer_from_response,
    calculate_accuracy_metrics,
    aggregate_results,
)


class TestExactMatch:
    """Test exact_match function."""

    def test_identical_strings(self):
        """Test exact match with identical strings."""
        assert exact_match("hello world", "hello world") == 1.0

    def test_case_insensitive(self):
        """Test that matching is case-insensitive."""
        assert exact_match("Hello World", "hello world") == 1.0
        assert exact_match("HELLO", "hello") == 1.0

    def test_whitespace_normalization(self):
        """Test that extra whitespace is normalized."""
        assert exact_match("hello  world", "hello world") == 1.0
        assert exact_match("  hello world  ", "hello world") == 1.0
        assert exact_match("hello\n\nworld", "hello world") == 1.0

    def test_different_strings(self):
        """Test non-matching strings."""
        assert exact_match("hello", "goodbye") == 0.0
        assert exact_match("hello world", "hello") == 0.0

    def test_empty_strings(self):
        """Test with empty strings."""
        assert exact_match("", "") == 1.0
        assert exact_match("hello", "") == 0.0
        assert exact_match("", "hello") == 0.0


class TestPartialMatch:
    """Test partial_match function."""

    def test_identical_strings(self):
        """Test partial match with identical strings."""
        score = partial_match("hello world", "hello world")
        assert score == 1.0

    def test_similar_strings(self):
        """Test partial match with similar strings."""
        score = partial_match("hello world", "hello world!")
        assert 0.9 < score < 1.0

    def test_partial_overlap(self):
        """Test strings with partial overlap."""
        score = partial_match("hello world", "hello there")
        assert 0.4 < score < 0.7

    def test_completely_different(self):
        """Test completely different strings."""
        score = partial_match("hello", "xyz")
        assert score < 0.3

    def test_case_normalization(self):
        """Test that case is normalized."""
        score1 = partial_match("Hello World", "hello world")
        score2 = partial_match("hello world", "hello world")
        assert score1 == score2

    def test_substring_match(self):
        """Test substring matching."""
        score = partial_match("abc", "abcdef")
        assert score > 0.5  # Significant overlap


class TestKeywordMatch:
    """Test keyword_match function."""

    def test_all_keywords_present(self):
        """Test when all keywords are present."""
        score = keyword_match(
            "The secret password is ABC123",
            ["password", "ABC123"]
        )
        assert score == 1.0

    def test_some_keywords_present(self):
        """Test when some keywords are present."""
        score = keyword_match(
            "The password is secret",
            ["password", "ABC123", "secret"]
        )
        assert score == pytest.approx(2/3, rel=0.01)

    def test_no_keywords_present(self):
        """Test when no keywords are present."""
        score = keyword_match(
            "Hello world",
            ["password", "secret"]
        )
        assert score == 0.0

    def test_case_insensitive(self):
        """Test that keyword matching is case-insensitive."""
        score = keyword_match(
            "The PASSWORD is SECRET",
            ["password", "secret"]
        )
        assert score == 1.0

    def test_empty_keywords(self):
        """Test with empty keyword list."""
        score = keyword_match("hello world", [])
        assert score == 0.0

    def test_keywords_as_substrings(self):
        """Test that keywords can be substrings."""
        score = keyword_match(
            "The company's revenue was substantial",
            ["revenue", "company"]
        )
        assert score == 1.0


class TestSemanticSimilarity:
    """Test semantic_similarity function."""

    def test_identical_embeddings(self):
        """Test similarity of identical embeddings."""
        emb = np.random.randn(384).astype(np.float32)
        score = semantic_similarity(emb, emb.copy())
        assert score == pytest.approx(1.0, abs=0.01)

    def test_similar_embeddings(self):
        """Test similarity of similar embeddings."""
        np.random.seed(42)
        emb1 = np.random.randn(384).astype(np.float32)
        emb2 = emb1 + np.random.randn(384).astype(np.float32) * 0.1

        score = semantic_similarity(emb1, emb2)
        assert 0.5 < score < 1.0

    def test_orthogonal_embeddings(self):
        """Test orthogonal embeddings (should be ~0.5 after normalization)."""
        emb1 = np.zeros(384, dtype=np.float32)
        emb1[0] = 1.0

        emb2 = np.zeros(384, dtype=np.float32)
        emb2[1] = 1.0

        score = semantic_similarity(emb1, emb2)
        # Orthogonal vectors have cosine=0, normalized to 0.5
        assert score == pytest.approx(0.5, abs=0.01)

    def test_1d_embeddings(self):
        """Test that 1D embeddings are handled correctly."""
        emb1 = np.random.randn(384).astype(np.float32)
        emb2 = np.random.randn(384).astype(np.float32)

        score = semantic_similarity(emb1, emb2)
        assert 0.0 <= score <= 1.0

    def test_2d_embeddings(self):
        """Test that 2D embeddings work."""
        emb1 = np.random.randn(1, 384).astype(np.float32)
        emb2 = np.random.randn(1, 384).astype(np.float32)

        score = semantic_similarity(emb1, emb2)
        assert 0.0 <= score <= 1.0


class TestCalculateMeanStd:
    """Test calculate_mean_std function."""

    def test_simple_values(self):
        """Test with simple values."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean, std = calculate_mean_std(values)

        assert mean == 3.0
        assert std == pytest.approx(np.std([1, 2, 3, 4, 5]))

    def test_empty_list(self):
        """Test with empty list."""
        mean, std = calculate_mean_std([])
        assert mean == 0.0
        assert std == 0.0

    def test_single_value(self):
        """Test with single value."""
        mean, std = calculate_mean_std([5.0])
        assert mean == 5.0
        assert std == 0.0

    def test_constant_values(self):
        """Test with constant values."""
        mean, std = calculate_mean_std([3.0, 3.0, 3.0, 3.0])
        assert mean == 3.0
        assert std == 0.0


class TestCalculateConfidenceInterval:
    """Test calculate_confidence_interval function."""

    def test_simple_ci(self):
        """Test confidence interval calculation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean, lower, upper = calculate_confidence_interval(values)

        assert mean == 3.0
        assert lower < mean
        assert upper > mean
        assert upper > lower

    def test_95_percent_ci(self):
        """Test 95% confidence interval."""
        values = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
        mean, lower, upper = calculate_confidence_interval(values, confidence=0.95)

        # Mean should be 15
        assert mean == 15.0

        # CI should be symmetric around mean
        assert abs((mean - lower) - (upper - mean)) < 0.01

    def test_empty_list(self):
        """Test with empty list."""
        mean, lower, upper = calculate_confidence_interval([])
        assert mean == 0.0
        assert lower == 0.0
        assert upper == 0.0

    def test_single_value(self):
        """Test with single value."""
        mean, lower, upper = calculate_confidence_interval([5.0])
        assert mean == 0.0  # Function returns 0 for n < 2


class TestCalculateStatistics:
    """Test calculate_statistics function."""

    def test_comprehensive_statistics(self):
        """Test that all statistics are calculated."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        stats = calculate_statistics(values)

        # Check all keys present
        expected_keys = ["mean", "std", "min", "max", "median", "ci_lower", "ci_upper"]
        for key in expected_keys:
            assert key in stats

        # Check values
        assert stats["mean"] == 5.5
        assert stats["min"] == 1.0
        assert stats["max"] == 10.0
        assert stats["median"] == 5.5

    def test_empty_list(self):
        """Test with empty list."""
        stats = calculate_statistics([])

        assert stats["mean"] == 0.0
        assert stats["std"] == 0.0
        assert stats["min"] == 0.0
        assert stats["max"] == 0.0

    def test_single_value(self):
        """Test with single value."""
        stats = calculate_statistics([42.0])

        assert stats["mean"] == 42.0
        assert stats["std"] == 0.0
        assert stats["min"] == 42.0
        assert stats["max"] == 42.0
        assert stats["median"] == 42.0

    def test_confidence_intervals_in_stats(self):
        """Test that confidence intervals are included."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = calculate_statistics(values)

        assert "ci_lower" in stats
        assert "ci_upper" in stats
        assert stats["ci_lower"] < stats["mean"]
        assert stats["ci_upper"] > stats["mean"]


class TestPerformTTest:
    """Test perform_t_test function."""

    def test_significant_difference(self):
        """Test t-test with significantly different groups."""
        group1 = [0.8, 0.85, 0.9, 0.88, 0.82, 0.87, 0.91, 0.86]
        group2 = [0.3, 0.35, 0.32, 0.28, 0.33, 0.31, 0.29, 0.34]

        result = perform_t_test(group1, group2)

        assert result["p_value"] < 0.05
        assert result["significant"] is True
        assert result["effect_size"] > 0.5  # Large effect
        assert "t_statistic" in result

    def test_no_significant_difference(self):
        """Test t-test with similar groups."""
        group1 = [0.5, 0.52, 0.48, 0.51, 0.49]
        group2 = [0.51, 0.49, 0.50, 0.52, 0.48]

        result = perform_t_test(group1, group2)

        assert result["p_value"] > 0.05
        assert result["significant"] is False
        assert result["effect_size"] < 0.5  # Small effect

    def test_insufficient_data(self):
        """Test with insufficient data."""
        group1 = [0.5]
        group2 = [0.6]

        result = perform_t_test(group1, group2)

        assert result["t_statistic"] == 0.0
        assert result["p_value"] == 1.0
        assert result["significant"] is False

    def test_effect_size_calculation(self):
        """Test Cohen's d effect size calculation."""
        group1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        group2 = [3.0, 4.0, 5.0, 6.0, 7.0]

        result = perform_t_test(group1, group2)

        # Effect size should be positive
        assert result["effect_size"] > 0
        assert "effect_size" in result


class TestExtractAnswerFromResponse:
    """Test extract_answer_from_response function."""

    def test_extract_with_custom_pattern(self):
        """Test extraction with custom regex pattern."""
        response = "The secret code is ABC123 and that's final."
        pattern = r"code is (\w+)"

        extracted = extract_answer_from_response(response, pattern)
        assert extracted == "ABC123"

    def test_extract_password(self):
        """Test extracting password from response."""
        response = "The password is: SECRET123."

        extracted = extract_answer_from_response(response)
        assert "SECRET123" in extracted

    def test_extract_revenue(self):
        """Test extracting revenue from response."""
        response = "The annual revenue: $500 million."

        extracted = extract_answer_from_response(response)
        assert "$500 million" in extracted

    def test_extract_first_sentence(self):
        """Test fallback to first sentence."""
        response = "This is the answer. This is extra information."

        extracted = extract_answer_from_response(response)
        assert "This is the answer" in extracted

    def test_no_pattern_match(self):
        """Test when no pattern matches."""
        response = "Some random text without structure"

        extracted = extract_answer_from_response(response)
        # Should return trimmed response
        assert extracted == response.strip()

    def test_answer_prefix(self):
        """Test extraction with 'answer is' prefix."""
        response = "The answer is: 42."

        extracted = extract_answer_from_response(response)
        assert "42" in extracted


class TestCalculateAccuracyMetrics:
    """Test calculate_accuracy_metrics function."""

    def test_perfect_match(self):
        """Test metrics with perfect match."""
        metrics = calculate_accuracy_metrics(
            predicted="The answer is ABC123",
            expected="The answer is ABC123"
        )

        assert metrics["exact_match"] == 1.0
        assert metrics["partial_match"] == 1.0
        assert metrics["overall_score"] > 0.9

    def test_with_keywords(self):
        """Test metrics with keyword matching."""
        metrics = calculate_accuracy_metrics(
            predicted="The secret password is TEST123",
            expected="TEST123",
            keywords=["TEST123", "password"]
        )

        assert "keyword_match" in metrics
        assert metrics["keyword_match"] == 1.0

    def test_with_embeddings(self, sample_embeddings):
        """Test metrics with semantic similarity."""
        emb = sample_embeddings["similar"]
        emb_copy = sample_embeddings["similar_copy"]

        metrics = calculate_accuracy_metrics(
            predicted="hello world",
            expected="hello world",
            predicted_embedding=emb,
            expected_embedding=emb_copy
        )

        assert "semantic_similarity" in metrics
        assert metrics["semantic_similarity"] > 0.9

    def test_partial_match_only(self):
        """Test with partial match."""
        metrics = calculate_accuracy_metrics(
            predicted="The password is TEST",
            expected="The password is TEST123"
        )

        assert metrics["exact_match"] < 1.0
        assert metrics["partial_match"] > 0.7
        assert 0.0 <= metrics["overall_score"] <= 1.0

    def test_no_match(self):
        """Test with no match."""
        metrics = calculate_accuracy_metrics(
            predicted="completely different",
            expected="something else"
        )

        assert metrics["exact_match"] == 0.0
        assert metrics["partial_match"] < 0.5
        assert metrics["overall_score"] < 0.5

    def test_overall_score_calculation(self):
        """Test that overall score is properly weighted."""
        metrics = calculate_accuracy_metrics(
            predicted="test",
            expected="test"
        )

        # Should have exact_match and partial_match
        assert "overall_score" in metrics
        assert 0.0 <= metrics["overall_score"] <= 1.0

    def test_all_metrics_combined(self, sample_embeddings):
        """Test with all metrics enabled."""
        metrics = calculate_accuracy_metrics(
            predicted="The secret password is ABC123",
            expected="ABC123",
            predicted_embedding=sample_embeddings["embedding_1"],
            expected_embedding=sample_embeddings["embedding_2"],
            keywords=["ABC123", "password"]
        )

        # All metrics should be present
        assert "exact_match" in metrics
        assert "partial_match" in metrics
        assert "keyword_match" in metrics
        assert "semantic_similarity" in metrics
        assert "overall_score" in metrics

        # Overall score should be weighted combination
        assert 0.0 <= metrics["overall_score"] <= 1.0


class TestAggregateResults:
    """Test aggregate_results function."""

    def test_aggregate_multiple_results(self):
        """Test aggregating multiple result dictionaries."""
        results = [
            {"accuracy": 0.8, "latency": 0.5, "name": "test1"},
            {"accuracy": 0.9, "latency": 0.6, "name": "test2"},
            {"accuracy": 0.85, "latency": 0.55, "name": "test3"},
        ]

        aggregated = aggregate_results(results)

        # Should have statistics for numeric metrics
        assert "accuracy" in aggregated
        assert "latency" in aggregated

        # Each should have full statistics
        assert "mean" in aggregated["accuracy"]
        assert "std" in aggregated["accuracy"]

        # Check mean values
        assert aggregated["accuracy"]["mean"] == pytest.approx(0.85, abs=0.01)
        assert aggregated["latency"]["mean"] == pytest.approx(0.55, abs=0.01)

    def test_aggregate_empty_results(self):
        """Test with empty results list."""
        aggregated = aggregate_results([])
        assert aggregated == {}

    def test_aggregate_ignores_non_numeric(self):
        """Test that non-numeric fields are ignored."""
        results = [
            {"score": 0.8, "name": "test1", "status": "success"},
            {"score": 0.9, "name": "test2", "status": "success"},
        ]

        aggregated = aggregate_results(results)

        # Should only aggregate numeric fields
        assert "score" in aggregated
        assert "name" not in aggregated
        assert "status" not in aggregated

    def test_aggregate_handles_missing_keys(self):
        """Test aggregation with inconsistent keys."""
        results = [
            {"accuracy": 0.8, "latency": 0.5},
            {"accuracy": 0.9},  # Missing latency
            {"accuracy": 0.85, "latency": 0.6},
        ]

        aggregated = aggregate_results(results)

        # Should aggregate what's available
        assert "accuracy" in aggregated
        assert "latency" in aggregated

        # Accuracy should use all 3 values
        assert aggregated["accuracy"]["mean"] == pytest.approx(0.85, abs=0.01)

        # Latency should use 2 values
        assert aggregated["latency"]["mean"] == pytest.approx(0.55, abs=0.01)

    def test_aggregate_single_result(self):
        """Test with single result."""
        results = [{"score": 0.75, "time": 1.5}]

        aggregated = aggregate_results(results)

        assert "score" in aggregated
        assert aggregated["score"]["mean"] == 0.75
        assert aggregated["score"]["std"] == 0.0


class TestMetricsIntegration:
    """Integration tests for metrics module."""

    def test_full_evaluation_pipeline(self, sample_embeddings):
        """Test complete evaluation pipeline."""
        # Simulate LLM response
        response = "Based on the documents, the secret password is TEST12345."
        expected = "TEST12345"
        keywords = ["TEST12345", "password"]

        # Extract answer
        extracted = extract_answer_from_response(response)

        # Calculate metrics
        metrics = calculate_accuracy_metrics(
            predicted=extracted,
            expected=expected,
            predicted_embedding=sample_embeddings["embedding_1"],
            expected_embedding=sample_embeddings["embedding_2"],
            keywords=keywords
        )

        # Should have all metrics
        assert "exact_match" in metrics
        assert "partial_match" in metrics
        assert "keyword_match" in metrics
        assert "semantic_similarity" in metrics
        assert "overall_score" in metrics

        # Should detect the password
        assert metrics["keyword_match"] >= 0.5

    def test_statistical_comparison_workflow(self):
        """Test statistical comparison workflow."""
        # Simulate experiment results
        baseline_scores = [0.7, 0.72, 0.68, 0.71, 0.69, 0.73]
        improved_scores = [0.85, 0.87, 0.83, 0.86, 0.84, 0.88]

        # Calculate statistics
        baseline_stats = calculate_statistics(baseline_scores)
        improved_stats = calculate_statistics(improved_scores)

        # Perform t-test
        comparison = perform_t_test(improved_scores, baseline_scores)

        # Should show significant improvement
        assert comparison["significant"] is True
        assert improved_stats["mean"] > baseline_stats["mean"]

    def test_batch_evaluation_aggregation(self):
        """Test batch evaluation and aggregation."""
        # Simulate multiple evaluation runs
        evaluations = []

        test_cases = [
            ("answer is 42", "42", ["42"]),
            ("the result is 100", "100", ["100"]),
            ("value is 99", "99", ["99"]),
        ]

        for predicted, expected, keywords in test_cases:
            metrics = calculate_accuracy_metrics(
                predicted=predicted,
                expected=expected,
                keywords=keywords
            )
            evaluations.append(metrics)

        # Aggregate results
        aggregated = aggregate_results(evaluations)

        # Should have statistics for all metrics
        assert "exact_match" in aggregated
        assert "overall_score" in aggregated

        # All should be dictionaries with statistics
        assert "mean" in aggregated["overall_score"]
        assert "std" in aggregated["overall_score"]
