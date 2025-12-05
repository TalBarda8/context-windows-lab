"""
Configuration file for Context Windows Lab experiments.

This module contains all configurable parameters for the experiments,
including model settings, experiment parameters, and file paths.
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Data subdirectories
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
NEEDLE_HAYSTACK_DIR = SYNTHETIC_DATA_DIR / "needle_haystack"
CONTEXT_SIZE_DIR = SYNTHETIC_DATA_DIR / "context_size"
HEBREW_CORPUS_DIR = DATA_DIR / "hebrew_corpus"

# Results subdirectories
EXP1_RESULTS_DIR = RESULTS_DIR / "exp1"
EXP2_RESULTS_DIR = RESULTS_DIR / "exp2"
EXP3_RESULTS_DIR = RESULTS_DIR / "exp3"
EXP4_RESULTS_DIR = RESULTS_DIR / "exp4"

# ============================================================================
# LLM CONFIGURATION
# ============================================================================

# Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434"
PRIMARY_MODEL = "llama2"  # Primary model to use
FALLBACK_MODEL = "mistral"  # Fallback if primary unavailable

# LLM parameters
LLM_TEMPERATURE = 0.0  # Deterministic outputs for reproducibility
LLM_TOP_P = 1.0
LLM_SEED = 42  # Random seed for reproducibility

# Context window limits (tokens)
DEFAULT_CONTEXT_WINDOW = 4096  # Typical for llama2
MAX_CONTEXT_WINDOW = 4096  # llama2 standard limit

# ============================================================================
# EXPERIMENT 1: NEEDLE IN HAYSTACK
# ============================================================================

EXP1_CONFIG = {
    "num_documents": 5,  # Number of documents to generate
    "words_per_document": 200,  # Words in each document
    "positions": ["start", "middle", "end"],  # Fact positions to test
    "iterations_per_position": 3,  # Reduced for faster runtime (was 10)
    "critical_fact_template": "The secret password is {password}.",
    "query_template": "What is the secret password mentioned in the documents?",
}

# ============================================================================
# EXPERIMENT 2: CONTEXT WINDOW SIZE IMPACT
# ============================================================================

EXP2_CONFIG = {
    "document_counts": [2, 5, 10, 20],  # Varying context sizes (removed 50 for speed)
    "words_per_document": 150,  # Reduced from 200 for faster processing
    "iterations_per_size": 3,  # Reduced from 5 for faster runtime
    "question_template": "What is the company's annual revenue mentioned in the documents?",
}

# ============================================================================
# EXPERIMENT 3: RAG IMPACT
# ============================================================================

EXP3_CONFIG = {
    "num_documents": 10,  # Reduced from 20 for faster generation
    "topics": ["technology", "law", "medicine"],  # Document categories
    "chunk_size": 400,  # Reduced from 500 for faster processing
    "chunk_overlap": 50,  # Overlap between chunks
    "top_k_retrieval": 3,  # Number of chunks to retrieve
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "question_templates": {
        "technology": "What are the main features of the new software system?",
        "law": "What are the legal requirements for data privacy?",
        "medicine": "What are the side effects of {drug_name}?",
    },
}

# ============================================================================
# EXPERIMENT 4: CONTEXT ENGINEERING STRATEGIES
# ============================================================================

EXP4_CONFIG = {
    "num_actions": 10,  # Sequential agent actions
    "max_tokens_threshold": 2048,  # Threshold for compression strategy
    "strategies": ["select", "compress", "write"],
    "select_top_k": 5,  # For SELECT strategy RAG retrieval
    "scratchpad_capacity": 20,  # Max facts in WRITE strategy
    "agent_action_template": "Execute task {task_id}: {task_description}",
}

# ============================================================================
# EVALUATION SETTINGS
# ============================================================================

EVALUATION_CONFIG = {
    "similarity_threshold": 0.8,  # For semantic similarity matching
    "use_exact_match": True,  # Include exact string matching
    "use_keyword_match": True,  # Include keyword presence checking
    "use_semantic_match": True,  # Include embedding-based similarity
}

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

PLOT_CONFIG = {
    "figsize": (10, 6),  # Default figure size
    "dpi": 300,  # High resolution for publication
    "style": "seaborn-v0_8-darkgrid",  # Plot style
    "color_palette": "Set2",  # Color scheme
    "save_format": "png",  # Image format
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_CONFIG = {
    "level": "INFO",  # Logging level: DEBUG, INFO, WARNING, ERROR
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
}

# ============================================================================
# RANDOM SEEDS FOR REPRODUCIBILITY
# ============================================================================

RANDOM_SEED = 42  # Master random seed
import random
import numpy as np

def set_random_seeds(seed=RANDOM_SEED):
    """Set random seeds for reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ensure_directories():
    """Create all necessary directories if they don't exist."""
    directories = [
        DATA_DIR,
        SYNTHETIC_DATA_DIR,
        NEEDLE_HAYSTACK_DIR,
        CONTEXT_SIZE_DIR,
        HEBREW_CORPUS_DIR,
        RESULTS_DIR,
        EXP1_RESULTS_DIR,
        EXP2_RESULTS_DIR,
        EXP3_RESULTS_DIR,
        EXP4_RESULTS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def get_model_name():
    """Get the model name to use, with fallback logic."""
    try:
        # Try to import ollama and check available models
        import subprocess
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if PRIMARY_MODEL in result.stdout:
            return PRIMARY_MODEL
        elif FALLBACK_MODEL in result.stdout:
            print(f"Warning: {PRIMARY_MODEL} not found, using {FALLBACK_MODEL}")
            return FALLBACK_MODEL
        else:
            raise RuntimeError(
                f"Neither {PRIMARY_MODEL} nor {FALLBACK_MODEL} found. "
                f"Please run: ollama pull {PRIMARY_MODEL}"
            )
    except Exception as e:
        print(f"Warning: Could not check available models: {e}")
        return PRIMARY_MODEL  # Assume it's available

# Initialize on import
ensure_directories()
set_random_seeds()
