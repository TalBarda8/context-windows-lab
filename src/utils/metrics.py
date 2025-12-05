"""
Metrics and statistical utilities for evaluating experiment results.

This module provides functions for calculating accuracy, similarity,
and statistical measures across all experiments.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import re


def exact_match(predicted: str, expected: str) -> float:
    """
    Calculate exact match score (binary: 1.0 or 0.0).

    Args:
        predicted: Model's prediction
        expected: Expected answer

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    # Normalize whitespace and case
    pred_normalized = " ".join(predicted.lower().split())
    exp_normalized = " ".join(expected.lower().split())

    return 1.0 if pred_normalized == exp_normalized else 0.0


def partial_match(predicted: str, expected: str) -> float:
    """
    Calculate partial match using sequence matching (Levenshtein-like).

    Args:
        predicted: Model's prediction
        expected: Expected answer

    Returns:
        Similarity score between 0.0 and 1.0
    """
    # Normalize
    pred_normalized = predicted.lower().strip()
    exp_normalized = expected.lower().strip()

    # Use SequenceMatcher for fuzzy matching
    matcher = SequenceMatcher(None, pred_normalized, exp_normalized)
    return matcher.ratio()


def keyword_match(predicted: str, keywords: List[str]) -> float:
    """
    Check if keywords are present in prediction.

    Args:
        predicted: Model's prediction
        keywords: List of keywords to check for

    Returns:
        Proportion of keywords found (0.0 to 1.0)
    """
    if not keywords:
        return 0.0

    pred_lower = predicted.lower()
    found = sum(1 for kw in keywords if kw.lower() in pred_lower)

    return found / len(keywords)


def semantic_similarity(predicted_embedding: np.ndarray,
                       expected_embedding: np.ndarray) -> float:
    """
    Calculate cosine similarity between embeddings.

    Args:
        predicted_embedding: Embedding of prediction
        expected_embedding: Embedding of expected answer

    Returns:
        Cosine similarity score (0.0 to 1.0)
    """
    # Reshape to 2D if needed
    if predicted_embedding.ndim == 1:
        predicted_embedding = predicted_embedding.reshape(1, -1)
    if expected_embedding.ndim == 1:
        expected_embedding = expected_embedding.reshape(1, -1)

    similarity = cosine_similarity(predicted_embedding, expected_embedding)[0][0]

    # Normalize to 0-1 range (cosine can be -1 to 1)
    return (similarity + 1) / 2


def calculate_mean_std(values: List[float]) -> Tuple[float, float]:
    """
    Calculate mean and standard deviation.

    Args:
        values: List of numerical values

    Returns:
        Tuple of (mean, std)
    """
    if not values:
        return 0.0, 0.0

    return np.mean(values), np.std(values)


def calculate_confidence_interval(values: List[float],
                                  confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Calculate confidence interval for a list of values.

    Args:
        values: List of numerical values
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    if not values or len(values) < 2:
        return 0.0, 0.0, 0.0

    mean = np.mean(values)
    sem = stats.sem(values)  # Standard error of the mean

    # Calculate confidence interval
    interval = sem * stats.t.ppf((1 + confidence) / 2, len(values) - 1)

    return mean, mean - interval, mean + interval


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """
    Calculate comprehensive statistics for a list of values.

    Args:
        values: List of numerical values

    Returns:
        Dictionary with mean, std, min, max, median, and confidence intervals
    """
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
        }

    mean, ci_lower, ci_upper = calculate_confidence_interval(values)

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
    }


def perform_t_test(group1: List[float], group2: List[float]) -> Dict[str, Any]:
    """
    Perform independent t-test between two groups.

    Args:
        group1: First group of values
        group2: Second group of values

    Returns:
        Dictionary with t-statistic, p-value, and significance
    """
    if len(group1) < 2 or len(group2) < 2:
        return {
            "t_statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "effect_size": 0.0,
        }

    t_stat, p_value = stats.ttest_ind(group1, group2)

    # Calculate Cohen's d (effect size)
    cohens_d = (np.mean(group1) - np.mean(group2)) / np.sqrt(
        (np.std(group1) ** 2 + np.std(group2) ** 2) / 2
    )

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "effect_size": float(abs(cohens_d)),
    }


def extract_answer_from_response(response: str,
                                 pattern: str = None) -> str:
    """
    Extract answer from LLM response using pattern matching.

    Args:
        response: Full LLM response text
        pattern: Regex pattern to extract answer (optional)

    Returns:
        Extracted answer or full response if no pattern matches
    """
    if pattern:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1) if match.groups() else match.group(0)

    # Default: try to extract content after common prefixes
    common_prefixes = [
        r"the answer is[:\s]+(.+?)(?:\.|$)",
        r"(?:password|revenue|value|result)[:\s]+(.+?)(?:\.|$)",
        r"^(.+?)(?:\.|$)",  # First sentence
    ]

    for prefix_pattern in common_prefixes:
        match = re.search(prefix_pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()

    # Fallback: return trimmed response
    return response.strip()


def calculate_accuracy_metrics(predicted: str,
                               expected: str,
                               predicted_embedding: np.ndarray = None,
                               expected_embedding: np.ndarray = None,
                               keywords: List[str] = None) -> Dict[str, float]:
    """
    Calculate comprehensive accuracy metrics.

    Args:
        predicted: Model's prediction
        expected: Expected answer
        predicted_embedding: Embedding of prediction (optional)
        expected_embedding: Embedding of expected answer (optional)
        keywords: Keywords to check for (optional)

    Returns:
        Dictionary with all accuracy metrics
    """
    metrics = {
        "exact_match": exact_match(predicted, expected),
        "partial_match": partial_match(predicted, expected),
    }

    if keywords:
        metrics["keyword_match"] = keyword_match(predicted, keywords)

    if predicted_embedding is not None and expected_embedding is not None:
        metrics["semantic_similarity"] = semantic_similarity(
            predicted_embedding, expected_embedding
        )

    # Overall score (weighted average)
    weights = []
    scores = []

    if "exact_match" in metrics:
        weights.append(0.4)
        scores.append(metrics["exact_match"])

    if "partial_match" in metrics:
        weights.append(0.3)
        scores.append(metrics["partial_match"])

    if "keyword_match" in metrics:
        weights.append(0.15)
        scores.append(metrics["keyword_match"])

    if "semantic_similarity" in metrics:
        weights.append(0.15)
        scores.append(metrics["semantic_similarity"])

    # Normalize weights
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]
        metrics["overall_score"] = sum(w * s for w, s in zip(weights, scores))
    else:
        metrics["overall_score"] = 0.0

    return metrics


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate results from multiple iterations.

    Args:
        results: List of result dictionaries

    Returns:
        Aggregated statistics
    """
    if not results:
        return {}

    # Extract all numeric metrics
    metric_keys = set()
    for result in results:
        metric_keys.update(k for k, v in result.items()
                          if isinstance(v, (int, float)))

    aggregated = {}

    for key in metric_keys:
        values = [r[key] for r in results if key in r and isinstance(r[key], (int, float))]
        if values:
            aggregated[key] = calculate_statistics(values)

    return aggregated
