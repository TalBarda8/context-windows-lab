"""
Tests for experiment modules - lightweight integration tests.

These tests exercise the experiment code with mocked dependencies
to achieve coverage without actually running expensive LLM calls.
"""

import pytest
import json
from pathlib import Path
import sys
from unittest.mock import Mock, MagicMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.exp1_needle_haystack import NeedleHaystackExperiment
from experiments.exp2_context_size import ContextSizeExperiment
from experiments.exp3_rag_impact import RAGImpactExperiment
from experiments.exp4_strategies import ContextEngineeringExperiment


@pytest.fixture
def mock_create_llm():
    """Mock create_llm_interface."""
    with patch('experiments.exp1_needle_haystack.create_llm_interface') as mock1, \
         patch('experiments.exp2_context_size.create_llm_interface') as mock2, \
         patch('experiments.exp3_rag_impact.create_llm_interface') as mock3, \
         patch('experiments.exp4_strategies.create_llm_interface') as mock4:

        # Create mock LLM
        mock_llm = MagicMock()
        mock_llm.query.return_value = {
            "response": "The secret password is TEST123.",
            "latency": 0.5,
            "success": True,
            "error": None
        }

        mock1.return_value = mock_llm
        mock2.return_value = mock_llm
        mock3.return_value = mock_llm
        mock4.return_value = mock_llm

        yield mock_llm


@pytest.fixture
def mock_create_evaluator():
    """Mock create_evaluator."""
    with patch('experiments.exp1_needle_haystack.create_evaluator') as mock1, \
         patch('experiments.exp2_context_size.create_evaluator') as mock2, \
         patch('experiments.exp3_rag_impact.create_evaluator') as mock3, \
         patch('experiments.exp4_strategies.create_evaluator') as mock4:

        # Create mock evaluator
        mock_eval = MagicMock()
        mock_eval.evaluate_needle_haystack.return_value = {
            "exact_match": 1.0,
            "partial_match": 1.0,
            "overall_score": 1.0,
            "correct": True,
            "position": "middle"
        }
        mock_eval.evaluate_context_size.return_value = {
            "exact_match": 1.0,
            "partial_match": 1.0,
            "overall_score": 1.0,
            "correct": True,
            "num_docs": 10,
            "latency": 0.5
        }
        mock_eval.evaluate_rag_response.return_value = {
            "exact_match": 1.0,
            "partial_match": 1.0,
            "overall_score": 1.0,
            "correct": True,
            "method": "rag",
            "latency": 0.5
        }
        mock_eval.evaluate_strategy_step.return_value = {
            "exact_match": 1.0,
            "partial_match": 1.0,
            "overall_score": 1.0,
            "correct": True,
            "strategy": "select",
            "step": 1,
            "context_size": 1000
        }
        mock_eval.compare_groups.return_value = {
            "t_statistic": 2.5,
            "p_value": 0.01,
            "significant": True,
            "effect_size": 0.8
        }

        mock1.return_value = mock_eval
        mock2.return_value = mock_eval
        mock3.return_value = mock_eval
        mock4.return_value = mock_eval

        yield mock_eval


@pytest.fixture
def mock_create_rag():
    """Mock create_rag_system."""
    with patch('experiments.exp3_rag_impact.create_rag_system') as mock:
        mock_rag = MagicMock()
        mock_rag.add_documents.return_value = None
        mock_rag.query_with_rag.return_value = {
            "response": "תופעות הלוואי כוללות: כאב ראש, בחילה.",
            "latency": 0.5,
            "retrieve_time": 0.1,
            "total_time": 0.6,
            "retrieved_docs": [],
            "num_docs_retrieved": 3,
            "success": True,
            "error": None
        }

        mock.return_value = mock_rag
        yield mock_rag


@pytest.fixture
def mock_plot():
    """Mock visualization functions."""
    with patch('experiments.exp1_needle_haystack.plot_accuracy_by_position') as mock1, \
         patch('experiments.exp2_context_size.plot_context_size_impact') as mock2, \
         patch('experiments.exp3_rag_impact.plot_rag_comparison') as mock3, \
         patch('experiments.exp4_strategies.plot_strategy_comparison') as mock4:

        mock1.return_value = None
        mock2.return_value = None
        mock3.return_value = None
        mock4.return_value = None

        yield


class TestExp1NeedleHaystack:
    """Test Experiment 1: Needle in Haystack."""

    def test_initialization(self, mock_create_llm, mock_create_evaluator):
        """Test experiment initialization."""
        exp = NeedleHaystackExperiment()

        assert exp.config is not None
        assert exp.results_dir is not None
        assert exp.data_generator is not None
        assert exp.llm is not None
        assert exp.evaluator is not None

    def test_generate_data(self, mock_create_llm, mock_create_evaluator):
        """Test data generation."""
        exp = NeedleHaystackExperiment()

        dataset = exp.generate_data()

        assert isinstance(dataset, list)
        assert len(dataset) > 0

    def test_run_single_trial(self, mock_create_llm, mock_create_evaluator):
        """Test running a single trial."""
        exp = NeedleHaystackExperiment()

        result = exp.run_single_trial(
            document="The secret password is TEST123.",
            secret_value="TEST123",
            position="middle"
        )

        assert "position" in result
        assert "secret_value" in result
        assert "response" in result
        assert "latency" in result

    def test_run_all_trials(self, mock_create_llm, mock_create_evaluator):
        """Test running all trials."""
        exp = NeedleHaystackExperiment()

        # Generate small dataset
        exp.config["iterations_per_position"] = 1
        dataset = exp.generate_data()

        # Run trials
        results = exp.run_all_trials(dataset)

        assert isinstance(results, list)
        assert len(results) > 0

    def test_analyze_results(self, mock_create_llm, mock_create_evaluator):
        """Test results analysis."""
        exp = NeedleHaystackExperiment()

        # Mock results
        results = [
            {"position": "start", "correct": True, "latency": 0.5},
            {"position": "middle", "correct": False, "latency": 0.6},
            {"position": "end", "correct": True, "latency": 0.4},
        ]

        analysis = exp.analyze_results(results)

        assert "by_position" in analysis
        assert "overall" in analysis


class TestExp2ContextSize:
    """Test Experiment 2: Context Size Impact."""

    def test_initialization(self, mock_create_llm, mock_create_evaluator):
        """Test experiment initialization."""
        exp = ContextSizeExperiment()

        assert exp.config is not None
        assert exp.results_dir is not None
        assert exp.data_generator is not None

    def test_generate_data(self, mock_create_llm, mock_create_evaluator):
        """Test data generation for different context sizes."""
        exp = ContextSizeExperiment()

        exp.config["document_counts"] = [2, 5]
        datasets = exp.generate_data()

        assert isinstance(datasets, dict)
        assert 2 in datasets
        assert 5 in datasets

    def test_run_single_size(self, mock_create_llm, mock_create_evaluator):
        """Test running trials for a single context size."""
        exp = ContextSizeExperiment()

        documents = [{"document": "Test doc", "revenue_value": "$100M"}]

        results = exp.run_single_size(
            num_docs=1,
            documents=documents[:1],
            iterations=1
        )

        assert isinstance(results, list)
        assert len(results) > 0

    def test_run_all_trials(self, mock_create_llm, mock_create_evaluator):
        """Test running all context size trials."""
        exp = ContextSizeExperiment()

        exp.config["document_counts"] = [2]
        exp.config["iterations_per_size"] = 1

        datasets = exp.generate_data()
        results = exp.run_all_trials(datasets)

        assert isinstance(results, dict)

    def test_analyze_results(self, mock_create_llm, mock_create_evaluator):
        """Test results analysis."""
        exp = ContextSizeExperiment()

        results = {
            2: [{"num_docs": 2, "correct": True, "latency": 0.5}],
            5: [{"num_docs": 5, "correct": True, "latency": 1.0}],
        }

        analysis = exp.analyze_results(results)

        assert "by_size" in analysis
        assert "overall" in analysis


class TestExp3RAGImpact:
    """Test Experiment 3: RAG Impact."""

    def test_initialization(self, mock_create_llm, mock_create_evaluator, mock_create_rag):
        """Test experiment initialization."""
        exp = RAGImpactExperiment()

        assert exp.config is not None
        assert exp.results_dir is not None
        assert exp.data_generator is not None

    def test_generate_corpus(self, mock_create_llm, mock_create_evaluator, mock_create_rag):
        """Test Hebrew corpus generation."""
        exp = RAGImpactExperiment()

        exp.config["num_documents"] = 5
        corpus = exp.generate_corpus()

        assert isinstance(corpus, list)
        assert len(corpus) > 0

    def test_run_rag_query(self, mock_create_llm, mock_create_evaluator, mock_create_rag):
        """Test RAG query."""
        exp = RAGImpactExperiment()

        result = exp.run_rag_query(
            "What are the side effects?",
            "side effects: headache"
        )

        assert "response" in result
        assert "latency" in result

    def test_run_full_context_query(self, mock_create_llm, mock_create_evaluator, mock_create_rag):
        """Test full context query."""
        exp = RAGImpactExperiment()

        corpus = [{"content": "Test content"}]

        result = exp.run_full_context_query(
            "What is this about?",
            corpus,
            "test answer"
        )

        assert "response" in result
        assert "latency" in result

    def test_run_comparison(self, mock_create_llm, mock_create_evaluator, mock_create_rag):
        """Test running comparison."""
        exp = RAGImpactExperiment()

        exp.config["num_documents"] = 3
        corpus = exp.generate_corpus()

        # Mock question
        question = "Test question?"
        answer = "Test answer"

        results = exp.run_comparison(question, answer, corpus)

        assert "rag" in results
        assert "full_context" in results

    def test_analyze_results(self, mock_create_llm, mock_create_evaluator, mock_create_rag):
        """Test results analysis."""
        exp = RAGImpactExperiment()

        results = [
            {
                "rag": {"correct": True, "latency": 0.5},
                "full_context": {"correct": True, "latency": 2.0}
            }
        ]

        analysis = exp.analyze_results(results)

        assert "rag" in analysis
        assert "full_context" in analysis


class TestExp4Strategies:
    """Test Experiment 4: Context Management Strategies."""

    def test_initialization(self, mock_create_llm, mock_create_evaluator):
        """Test experiment initialization."""
        exp = ContextEngineeringExperiment()

        assert exp.config is not None
        assert exp.results_dir is not None

    def test_run_select_strategy(self, mock_create_llm, mock_create_evaluator):
        """Test running SELECT strategy."""
        exp = ContextEngineeringExperiment()

        exp.config["num_actions"] = 2

        results = exp.run_select_strategy()

        assert isinstance(results, list)

    def test_run_compress_strategy(self, mock_create_llm, mock_create_evaluator):
        """Test running COMPRESS strategy."""
        exp = ContextEngineeringExperiment()

        exp.config["num_actions"] = 2

        results = exp.run_compress_strategy()

        assert isinstance(results, list)

    def test_run_write_strategy(self, mock_create_llm, mock_create_evaluator):
        """Test running WRITE strategy."""
        exp = ContextEngineeringExperiment()

        exp.config["num_actions"] = 2

        results = exp.run_write_strategy()

        assert isinstance(results, list)

    def test_run_all_strategies(self, mock_create_llm, mock_create_evaluator):
        """Test running all strategies."""
        exp = ContextEngineeringExperiment()

        exp.config["num_actions"] = 2

        results = exp.run_all_strategies()

        assert isinstance(results, dict)

    def test_analyze_results(self, mock_create_llm, mock_create_evaluator):
        """Test results analysis."""
        exp = ContextEngineeringExperiment()

        results = {
            "select": [{"correct": True, "context_size": 1000}],
            "compress": [{"correct": True, "context_size": 500}],
            "write": [{"correct": True, "context_size": 800}],
        }

        analysis = exp.analyze_results(results)

        assert "by_strategy" in analysis


class TestExperimentsIntegration:
    """Integration tests across all experiments."""

    def test_all_experiments_initialize(self, mock_create_llm, mock_create_evaluator, mock_create_rag):
        """Test that all experiments can be initialized."""
        exp1 = NeedleHaystackExperiment()
        exp2 = ContextSizeExperiment()
        exp3 = RAGImpactExperiment()
        exp4 = ContextEngineeringExperiment()

        assert exp1 is not None
        assert exp2 is not None
        assert exp3 is not None
        assert exp4 is not None

    def test_all_experiments_have_config(self, mock_create_llm, mock_create_evaluator, mock_create_rag):
        """Test that all experiments have configuration."""
        experiments = [
            NeedleHaystackExperiment(),
            ContextSizeExperiment(),
            RAGImpactExperiment(),
            ContextEngineeringExperiment(),
        ]

        for exp in experiments:
            assert hasattr(exp, 'config')
            assert exp.config is not None

    def test_all_experiments_have_results_dir(self, mock_create_llm, mock_create_evaluator, mock_create_rag):
        """Test that all experiments have results directory."""
        experiments = [
            NeedleHaystackExperiment(),
            ContextSizeExperiment(),
            RAGImpactExperiment(),
            ContextEngineeringExperiment(),
        ]

        for exp in experiments:
            assert hasattr(exp, 'results_dir')
            assert exp.results_dir is not None
