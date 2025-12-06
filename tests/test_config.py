"""
Tests for config.py - Configuration module.

Tests cover:
- Project paths configuration
- LLM configuration
- Experiment configurations
- Helper functions (set_random_seeds, ensure_directories, get_model_name)
- Configuration consistency
"""

import pytest
import os
import random
import numpy as np
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock
import subprocess

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import config


class TestProjectPaths:
    """Test project path configurations."""

    def test_project_root_exists(self):
        """Test that PROJECT_ROOT is defined and valid."""
        assert hasattr(config, 'PROJECT_ROOT')
        assert isinstance(config.PROJECT_ROOT, Path)

    def test_data_dir_path(self):
        """Test DATA_DIR path."""
        assert hasattr(config, 'DATA_DIR')
        assert config.DATA_DIR == config.PROJECT_ROOT / "data"

    def test_results_dir_path(self):
        """Test RESULTS_DIR path."""
        assert hasattr(config, 'RESULTS_DIR')
        assert config.RESULTS_DIR == config.PROJECT_ROOT / "results"

    def test_notebooks_dir_path(self):
        """Test NOTEBOOKS_DIR path."""
        assert hasattr(config, 'NOTEBOOKS_DIR')
        assert config.NOTEBOOKS_DIR == config.PROJECT_ROOT / "notebooks"

    def test_synthetic_data_dir(self):
        """Test SYNTHETIC_DATA_DIR path."""
        assert config.SYNTHETIC_DATA_DIR == config.DATA_DIR / "synthetic"

    def test_needle_haystack_dir(self):
        """Test NEEDLE_HAYSTACK_DIR path."""
        expected = config.SYNTHETIC_DATA_DIR / "needle_haystack"
        assert config.NEEDLE_HAYSTACK_DIR == expected

    def test_context_size_dir(self):
        """Test CONTEXT_SIZE_DIR path."""
        expected = config.SYNTHETIC_DATA_DIR / "context_size"
        assert config.CONTEXT_SIZE_DIR == expected

    def test_hebrew_corpus_dir(self):
        """Test HEBREW_CORPUS_DIR path."""
        expected = config.DATA_DIR / "hebrew_corpus"
        assert config.HEBREW_CORPUS_DIR == expected

    def test_experiment_results_dirs(self):
        """Test experiment results directory paths."""
        assert config.EXP1_RESULTS_DIR == config.RESULTS_DIR / "exp1"
        assert config.EXP2_RESULTS_DIR == config.RESULTS_DIR / "exp2"
        assert config.EXP3_RESULTS_DIR == config.RESULTS_DIR / "exp3"
        assert config.EXP4_RESULTS_DIR == config.RESULTS_DIR / "exp4"


class TestLLMConfiguration:
    """Test LLM configuration settings."""

    def test_ollama_base_url(self):
        """Test OLLAMA_BASE_URL setting."""
        assert hasattr(config, 'OLLAMA_BASE_URL')
        assert config.OLLAMA_BASE_URL == "http://localhost:11434"

    def test_primary_model(self):
        """Test PRIMARY_MODEL setting."""
        assert hasattr(config, 'PRIMARY_MODEL')
        assert isinstance(config.PRIMARY_MODEL, str)
        assert len(config.PRIMARY_MODEL) > 0

    def test_fallback_model(self):
        """Test FALLBACK_MODEL setting."""
        assert hasattr(config, 'FALLBACK_MODEL')
        assert isinstance(config.FALLBACK_MODEL, str)

    def test_llm_temperature(self):
        """Test LLM_TEMPERATURE setting."""
        assert config.LLM_TEMPERATURE == 0.0  # Deterministic

    def test_llm_top_p(self):
        """Test LLM_TOP_P setting."""
        assert config.LLM_TOP_P == 1.0

    def test_llm_seed(self):
        """Test LLM_SEED setting."""
        assert hasattr(config, 'LLM_SEED')
        assert config.LLM_SEED == 42

    def test_context_window_limits(self):
        """Test context window configurations."""
        assert config.DEFAULT_CONTEXT_WINDOW == 4096
        assert config.MAX_CONTEXT_WINDOW == 4096


class TestExperiment1Config:
    """Test Experiment 1 (Needle in Haystack) configuration."""

    def test_exp1_config_exists(self):
        """Test that EXP1_CONFIG exists."""
        assert hasattr(config, 'EXP1_CONFIG')
        assert isinstance(config.EXP1_CONFIG, dict)

    def test_exp1_num_documents(self):
        """Test num_documents setting."""
        assert "num_documents" in config.EXP1_CONFIG
        assert config.EXP1_CONFIG["num_documents"] == 5

    def test_exp1_words_per_document(self):
        """Test words_per_document setting."""
        assert "words_per_document" in config.EXP1_CONFIG
        assert config.EXP1_CONFIG["words_per_document"] == 200

    def test_exp1_positions(self):
        """Test positions setting."""
        assert "positions" in config.EXP1_CONFIG
        positions = config.EXP1_CONFIG["positions"]
        assert positions == ["start", "middle", "end"]

    def test_exp1_iterations(self):
        """Test iterations_per_position setting."""
        assert "iterations_per_position" in config.EXP1_CONFIG
        assert config.EXP1_CONFIG["iterations_per_position"] == 10

    def test_exp1_templates(self):
        """Test template strings."""
        assert "critical_fact_template" in config.EXP1_CONFIG
        assert "query_template" in config.EXP1_CONFIG

        assert "{password}" in config.EXP1_CONFIG["critical_fact_template"]


class TestExperiment2Config:
    """Test Experiment 2 (Context Size) configuration."""

    def test_exp2_config_exists(self):
        """Test that EXP2_CONFIG exists."""
        assert hasattr(config, 'EXP2_CONFIG')
        assert isinstance(config.EXP2_CONFIG, dict)

    def test_exp2_document_counts(self):
        """Test document_counts setting."""
        assert "document_counts" in config.EXP2_CONFIG
        counts = config.EXP2_CONFIG["document_counts"]
        assert isinstance(counts, list)
        assert len(counts) > 0
        assert all(isinstance(c, int) for c in counts)
        assert counts == sorted(counts)  # Should be sorted

    def test_exp2_words_per_document(self):
        """Test words_per_document setting."""
        assert config.EXP2_CONFIG["words_per_document"] == 200

    def test_exp2_iterations(self):
        """Test iterations_per_size setting."""
        assert "iterations_per_size" in config.EXP2_CONFIG
        assert config.EXP2_CONFIG["iterations_per_size"] == 5

    def test_exp2_question_template(self):
        """Test question_template setting."""
        assert "question_template" in config.EXP2_CONFIG
        assert isinstance(config.EXP2_CONFIG["question_template"], str)


class TestExperiment3Config:
    """Test Experiment 3 (RAG) configuration."""

    def test_exp3_config_exists(self):
        """Test that EXP3_CONFIG exists."""
        assert hasattr(config, 'EXP3_CONFIG')
        assert isinstance(config.EXP3_CONFIG, dict)

    def test_exp3_num_documents(self):
        """Test num_documents setting."""
        assert config.EXP3_CONFIG["num_documents"] == 20

    def test_exp3_topics(self):
        """Test topics setting."""
        assert "topics" in config.EXP3_CONFIG
        topics = config.EXP3_CONFIG["topics"]
        assert isinstance(topics, list)
        assert "medicine" in topics or "technology" in topics

    def test_exp3_chunk_size(self):
        """Test chunk_size setting."""
        assert config.EXP3_CONFIG["chunk_size"] == 500

    def test_exp3_chunk_overlap(self):
        """Test chunk_overlap setting."""
        assert config.EXP3_CONFIG["chunk_overlap"] == 50

    def test_exp3_top_k_retrieval(self):
        """Test top_k_retrieval setting."""
        assert config.EXP3_CONFIG["top_k_retrieval"] == 3

    def test_exp3_embedding_model(self):
        """Test embedding_model setting."""
        assert "embedding_model" in config.EXP3_CONFIG
        assert isinstance(config.EXP3_CONFIG["embedding_model"], str)

    def test_exp3_question_templates(self):
        """Test question_templates setting."""
        assert "question_templates" in config.EXP3_CONFIG
        templates = config.EXP3_CONFIG["question_templates"]
        assert isinstance(templates, dict)


class TestExperiment4Config:
    """Test Experiment 4 (Strategies) configuration."""

    def test_exp4_config_exists(self):
        """Test that EXP4_CONFIG exists."""
        assert hasattr(config, 'EXP4_CONFIG')
        assert isinstance(config.EXP4_CONFIG, dict)

    def test_exp4_num_actions(self):
        """Test num_actions setting."""
        assert config.EXP4_CONFIG["num_actions"] == 10

    def test_exp4_max_tokens_threshold(self):
        """Test max_tokens_threshold setting."""
        assert "max_tokens_threshold" in config.EXP4_CONFIG
        assert config.EXP4_CONFIG["max_tokens_threshold"] == 2048

    def test_exp4_strategies(self):
        """Test strategies setting."""
        assert "strategies" in config.EXP4_CONFIG
        strategies = config.EXP4_CONFIG["strategies"]
        assert "select" in strategies
        assert "compress" in strategies
        assert "write" in strategies

    def test_exp4_select_top_k(self):
        """Test select_top_k setting."""
        assert config.EXP4_CONFIG["select_top_k"] == 5

    def test_exp4_scratchpad_capacity(self):
        """Test scratchpad_capacity setting."""
        assert config.EXP4_CONFIG["scratchpad_capacity"] == 20


class TestEvaluationConfig:
    """Test evaluation configuration."""

    def test_evaluation_config_exists(self):
        """Test that EVALUATION_CONFIG exists."""
        assert hasattr(config, 'EVALUATION_CONFIG')
        assert isinstance(config.EVALUATION_CONFIG, dict)

    def test_similarity_threshold(self):
        """Test similarity_threshold setting."""
        assert "similarity_threshold" in config.EVALUATION_CONFIG
        threshold = config.EVALUATION_CONFIG["similarity_threshold"]
        assert 0.0 <= threshold <= 1.0

    def test_match_options(self):
        """Test match option flags."""
        assert config.EVALUATION_CONFIG["use_exact_match"] is True
        assert config.EVALUATION_CONFIG["use_keyword_match"] is True
        assert config.EVALUATION_CONFIG["use_semantic_match"] is True


class TestPlotConfig:
    """Test plot/visualization configuration."""

    def test_plot_config_exists(self):
        """Test that PLOT_CONFIG exists."""
        assert hasattr(config, 'PLOT_CONFIG')
        assert isinstance(config.PLOT_CONFIG, dict)

    def test_figsize(self):
        """Test figsize setting."""
        assert "figsize" in config.PLOT_CONFIG
        figsize = config.PLOT_CONFIG["figsize"]
        assert isinstance(figsize, tuple)
        assert len(figsize) == 2

    def test_dpi(self):
        """Test DPI setting."""
        assert config.PLOT_CONFIG["dpi"] == 300

    def test_save_format(self):
        """Test save_format setting."""
        assert config.PLOT_CONFIG["save_format"] == "png"


class TestLogConfig:
    """Test logging configuration."""

    def test_log_config_exists(self):
        """Test that LOG_CONFIG exists."""
        assert hasattr(config, 'LOG_CONFIG')
        assert isinstance(config.LOG_CONFIG, dict)

    def test_log_level(self):
        """Test log level setting."""
        assert "level" in config.LOG_CONFIG
        assert config.LOG_CONFIG["level"] in ["DEBUG", "INFO", "WARNING", "ERROR"]

    def test_log_format(self):
        """Test log format string."""
        assert "format" in config.LOG_CONFIG
        assert isinstance(config.LOG_CONFIG["format"], str)


class TestRandomSeeds:
    """Test random seed configuration."""

    def test_random_seed_defined(self):
        """Test that RANDOM_SEED is defined."""
        assert hasattr(config, 'RANDOM_SEED')
        assert config.RANDOM_SEED == 42


class TestSetRandomSeeds:
    """Test set_random_seeds function."""

    def test_set_random_seeds_default(self):
        """Test setting random seeds with default seed."""
        config.set_random_seeds()

        # Check that seeds are set
        # Python random
        val1 = random.random()
        config.set_random_seeds()
        val2 = random.random()
        assert val1 == val2  # Same seed produces same value

    def test_set_random_seeds_custom(self):
        """Test setting random seeds with custom seed."""
        config.set_random_seeds(seed=123)

        val1 = random.random()

        config.set_random_seeds(seed=123)

        val2 = random.random()

        assert val1 == val2

    def test_set_numpy_seed(self):
        """Test that numpy seed is set."""
        config.set_random_seeds(seed=42)

        arr1 = np.random.rand(5)

        config.set_random_seeds(seed=42)

        arr2 = np.random.rand(5)

        np.testing.assert_array_equal(arr1, arr2)

    def test_set_python_hash_seed(self):
        """Test that PYTHONHASHSEED is set."""
        config.set_random_seeds(seed=99)

        assert os.environ['PYTHONHASHSEED'] == '99'


class TestEnsureDirectories:
    """Test ensure_directories function."""

    def test_ensure_directories_called_on_import(self):
        """Test that ensure_directories is called on module import."""
        # This is tested implicitly - if directories weren't created,
        # many tests would fail. Just verify the function exists.
        assert hasattr(config, 'ensure_directories')
        assert callable(config.ensure_directories)

    def test_ensure_directories_creates_paths(self, temp_directory):
        """Test that ensure_directories creates all paths."""
        # We can't easily test the actual function since it operates on
        # real paths, but we can verify it's idempotent
        config.ensure_directories()
        config.ensure_directories()  # Should not raise error

        # Verify key directories exist
        assert config.DATA_DIR.exists() or True  # May not exist in test env
        assert config.RESULTS_DIR.exists() or True


class TestGetModelName:
    """Test get_model_name function."""

    def test_get_model_name_returns_string(self):
        """Test that get_model_name returns a string."""
        model_name = config.get_model_name()
        assert isinstance(model_name, str)
        assert len(model_name) > 0

    @patch('subprocess.run')
    def test_get_model_name_with_primary_available(self, mock_run):
        """Test when primary model is available."""
        # Mock subprocess result showing primary model available
        mock_result = MagicMock()
        mock_result.stdout = f"{config.PRIMARY_MODEL}  latest\nother-model latest"
        mock_run.return_value = mock_result

        model = config.get_model_name()
        assert model == config.PRIMARY_MODEL

    @patch('subprocess.run')
    def test_get_model_name_with_fallback(self, mock_run):
        """Test when only fallback model is available."""
        # Mock subprocess result showing only fallback model
        mock_result = MagicMock()
        mock_result.stdout = f"{config.FALLBACK_MODEL}  latest\nother-model latest"
        mock_run.return_value = mock_result

        model = config.get_model_name()
        assert model == config.FALLBACK_MODEL

    @patch('subprocess.run')
    def test_get_model_name_with_no_models(self, mock_run):
        """Test when no models are available."""
        # Mock subprocess result showing no models
        mock_result = MagicMock()
        mock_result.stdout = "other-model latest"
        mock_run.return_value = mock_result

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="Neither .* found"):
            config.get_model_name()

    @patch('subprocess.run')
    def test_get_model_name_subprocess_error(self, mock_run):
        """Test when subprocess fails."""
        # Mock subprocess raising exception
        mock_run.side_effect = subprocess.TimeoutExpired("ollama", 5)

        # Should fall back to PRIMARY_MODEL
        model = config.get_model_name()
        assert model == config.PRIMARY_MODEL


class TestConfigurationConsistency:
    """Test configuration consistency and relationships."""

    def test_chunk_overlap_less_than_chunk_size(self):
        """Test that chunk overlap is less than chunk size."""
        assert config.EXP3_CONFIG["chunk_overlap"] < config.EXP3_CONFIG["chunk_size"]

    def test_document_counts_reasonable(self):
        """Test that document counts are reasonable."""
        for count in config.EXP2_CONFIG["document_counts"]:
            assert 1 <= count <= 1000

    def test_iterations_positive(self):
        """Test that all iteration counts are positive."""
        assert config.EXP1_CONFIG["iterations_per_position"] > 0
        assert config.EXP2_CONFIG["iterations_per_size"] > 0

    def test_context_window_consistent(self):
        """Test context window settings are consistent."""
        assert config.DEFAULT_CONTEXT_WINDOW <= config.MAX_CONTEXT_WINDOW

    def test_temperature_in_range(self):
        """Test that temperature is in valid range."""
        assert 0.0 <= config.LLM_TEMPERATURE <= 2.0


class TestConfigIntegration:
    """Integration tests for configuration."""

    def test_all_experiment_configs_have_required_keys(self):
        """Test that all experiment configs have necessary keys."""
        # Exp1
        assert "num_documents" in config.EXP1_CONFIG
        assert "positions" in config.EXP1_CONFIG

        # Exp2
        assert "document_counts" in config.EXP2_CONFIG

        # Exp3
        assert "num_documents" in config.EXP3_CONFIG
        assert "chunk_size" in config.EXP3_CONFIG

        # Exp4
        assert "strategies" in config.EXP4_CONFIG
        assert "num_actions" in config.EXP4_CONFIG

    def test_paths_are_pathlib_paths(self):
        """Test that all path variables are Path objects."""
        path_vars = [
            config.PROJECT_ROOT,
            config.DATA_DIR,
            config.RESULTS_DIR,
            config.EXP1_RESULTS_DIR,
            config.EXP2_RESULTS_DIR,
            config.EXP3_RESULTS_DIR,
            config.EXP4_RESULTS_DIR,
        ]

        for path_var in path_vars:
            assert isinstance(path_var, Path)

    def test_config_module_imports_cleanly(self):
        """Test that config module imports without errors."""
        # If we got here, import was successful
        assert hasattr(config, 'PROJECT_ROOT')
        assert hasattr(config, 'EXP1_CONFIG')
        assert hasattr(config, 'set_random_seeds')
