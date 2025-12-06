# API Documentation

**Context Windows Lab - Public Interface Reference**

**Version**: 1.0.0
**Last Updated**: 2025-12-06

---

## Table of Contents

1. [Overview](#overview)
2. [Module: config](#module-config)
3. [Module: llm_interface](#module-llm_interface)
4. [Module: data_generator](#module-data_generator)
5. [Module: evaluator](#module-evaluator)
6. [Module: utils.metrics](#module-utilsmetrics)
7. [Module: utils.visualization](#module-utilsvisualization)
8. [Building Block Specifications](#building-block-specifications)
9. [Error Handling](#error-handling)
10. [Usage Examples](#usage-examples)

---

## Overview

This document provides comprehensive API documentation for all public interfaces in the Context Windows Lab project. The documentation follows the **Building Block Pattern** (Input Data, Output Data, Setup Data) as specified in the software submission guidelines.

### API Design Principles

- **Type Safety**: All functions use type hints (Python 3.10+)
- **Reproducibility**: Random seeds and deterministic configurations
- **Error Handling**: Graceful degradation with informative error messages
- **Modularity**: Independent components with clear interfaces
- **Documentation**: Docstrings on all public functions and classes

### Import Conventions

```python
# Configuration
from config import (
    EXP1_CONFIG, EXP2_CONFIG, EXP3_CONFIG, EXP4_CONFIG,
    ensure_directories, set_random_seeds
)

# LLM Interface
from llm_interface import (
    LLMInterface, EmbeddingInterface, RAGSystem,
    create_llm_interface, create_rag_system
)

# Data Generation
from data_generator import DataGenerator

# Evaluation
from evaluator import Evaluator, ExperimentEvaluator, create_evaluator

# Metrics
from utils.metrics import (
    exact_match, partial_match, keyword_match,
    calculate_statistics, perform_t_test
)

# Visualization
from utils.visualization import (
    plot_accuracy_by_position, plot_context_size_impact,
    plot_rag_comparison, plot_strategy_comparison
)
```

---

## Module: config

**Location**: `src/config.py`

### Overview

Centralized configuration management for all experiments, including paths, model settings, and experiment parameters.

### Constants

#### Project Paths

```python
PROJECT_ROOT: Path              # Root directory of project
DATA_DIR: Path                  # data/ directory
RESULTS_DIR: Path               # results/ directory
SYNTHETIC_DATA_DIR: Path        # data/synthetic/
HEBREW_CORPUS_DIR: Path         # data/hebrew_corpus/
```

#### LLM Configuration

```python
OLLAMA_BASE_URL: str           # Default: "http://localhost:11434"
PRIMARY_MODEL: str             # Default: "llama2"
FALLBACK_MODEL: str            # Default: "mistral"
LLM_TEMPERATURE: float         # Default: 0.0 (deterministic)
LLM_TOP_P: float              # Default: 1.0
LLM_SEED: int                 # Default: 42
```

#### Experiment Configurations

```python
EXP1_CONFIG: Dict[str, Any]    # Needle in Haystack parameters
EXP2_CONFIG: Dict[str, Any]    # Context Size Impact parameters
EXP3_CONFIG: Dict[str, Any]    # RAG Impact parameters
EXP4_CONFIG: Dict[str, Any]    # Context Engineering parameters
```

**Experiment 1 Configuration:**
```python
{
    "num_documents": 5,
    "words_per_document": 200,
    "positions": ["start", "middle", "end"],
    "iterations_per_position": 10,
    "critical_fact_template": "The secret password is {password}.",
    "query_template": "What is the secret password mentioned in the documents?"
}
```

**Experiment 2 Configuration:**
```python
{
    "document_counts": [2, 5, 10, 20, 50],
    "words_per_document": 200,
    "iterations_per_size": 5,
    "question_template": "What is the company's annual revenue mentioned in the documents?"
}
```

**Experiment 3 Configuration:**
```python
{
    "num_documents": 20,
    "topics": ["technology", "law", "medicine"],
    "chunk_size": 500,
    "chunk_overlap": 50,
    "top_k_retrieval": 3,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

**Experiment 4 Configuration:**
```python
{
    "num_actions": 10,
    "max_tokens_threshold": 2048,
    "strategies": ["select", "compress", "write"],
    "select_top_k": 5,
    "scratchpad_capacity": 20
}
```

### Functions

#### `ensure_directories()`

**Building Block: Directory Initialization**

**Input Data:**
- None

**Output Data:**
- None (side effect: creates directories)

**Setup Data:**
- PROJECT_ROOT: Path to project root
- All directory paths defined in module

**Description:**
Creates all necessary project directories if they don't exist.

**Example:**
```python
from config import ensure_directories

ensure_directories()  # Creates data/, results/, etc.
```

**Exceptions:**
- `OSError`: If directory creation fails due to permissions

---

#### `set_random_seeds(seed: int = 42)`

**Building Block: Reproducibility Setup**

**Input Data:**
- `seed: int` - Random seed value (default: 42)

**Output Data:**
- None (side effect: sets global random state)

**Setup Data:**
- Python `random` module
- NumPy `np.random` module
- `PYTHONHASHSEED` environment variable

**Description:**
Sets random seeds across all libraries to ensure reproducibility.

**Example:**
```python
from config import set_random_seeds

set_random_seeds(42)  # All random operations are now deterministic
```

**Thread Safety:**
- Not thread-safe (affects global state)

---

#### `get_model_name() -> str`

**Building Block: Model Selection**

**Input Data:**
- None

**Output Data:**
- `str`: Name of LLM model to use

**Setup Data:**
- `PRIMARY_MODEL`: Preferred model name
- `FALLBACK_MODEL`: Backup model name
- Ollama installation with available models

**Description:**
Determines which LLM model to use based on availability.

**Logic:**
1. Check if Ollama is running
2. List available models
3. Return PRIMARY_MODEL if available
4. Return FALLBACK_MODEL if primary unavailable
5. Raise error if neither available

**Example:**
```python
from config import get_model_name

model = get_model_name()  # Returns "llama2" or "mistral"
```

**Exceptions:**
- `RuntimeError`: If no suitable model found
- `subprocess.TimeoutExpired`: If Ollama doesn't respond

---

## Module: llm_interface

**Location**: `src/llm_interface.py`

### Class: LLMInterface

**Building Block: LLM Query Interface**

Unified interface for querying Large Language Models via Ollama.

#### Constructor

```python
def __init__(
    self,
    model_name: Optional[str] = None,
    temperature: float = LLM_TEMPERATURE
)
```

**Input Data:**
- `model_name: Optional[str]` - Model to use (auto-detect if None)
- `temperature: float` - Sampling temperature (0.0 = deterministic)

**Setup Data:**
- Ollama server running at `OLLAMA_BASE_URL`
- Model downloaded via `ollama pull <model_name>`

**Raises:**
- `ConnectionError`: If Ollama server unreachable
- `ValueError`: If model not found

---

#### Method: `query(prompt: str, context: Optional[str] = None) -> Dict[str, Any]`

**Building Block: LLM Inference**

**Input Data:**
- `prompt: str` - Question or instruction
- `context: Optional[str]` - Additional context to prepend

**Output Data:**
```python
{
    "response": str,         # LLM's generated response
    "latency": float,        # Response time in seconds
    "model": str,            # Model name used
    "success": bool,         # True if no errors
    "error": Optional[str]   # Error message if failed
}
```

**Setup Data:**
- Initialized LLMInterface instance
- Active network connection to Ollama

**Example:**
```python
llm = LLMInterface(model_name="llama2", temperature=0.0)

result = llm.query(
    prompt="What is 2+2?",
    context="You are a helpful math tutor."
)

print(result["response"])  # "The answer is 4."
print(f"Latency: {result['latency']:.2f}s")
```

**Error Handling:**
- Returns `success=False` with error message instead of raising
- Captures all exceptions (network, timeout, parsing errors)

---

#### Method: `query_with_template(template: str, **kwargs) -> Dict[str, Any]`

**Building Block: Templated Query**

**Input Data:**
- `template: str` - Prompt template with placeholders
- `**kwargs` - Variables to fill template

**Output Data:**
- Same as `query()` method

**Example:**
```python
llm = LLMInterface()

result = llm.query_with_template(
    template="What is the capital of {country}?",
    country="France"
)

print(result["response"])  # "The capital of France is Paris."
```

---

#### Method: `count_tokens(text: str) -> int`

**Building Block: Token Estimation**

**Input Data:**
- `text: str` - Text to count tokens for

**Output Data:**
- `int`: Estimated token count

**Setup Data:**
- None (uses heuristic: ~4 characters per token)

**Limitations:**
- Approximation only (actual tokenization varies by model)
- Suitable for estimation, not exact billing

**Example:**
```python
llm = LLMInterface()

tokens = llm.count_tokens("Hello, world!")
print(tokens)  # Approximately 3-4
```

---

### Class: EmbeddingInterface

**Building Block: Text Embedding Generation**

Interface for generating semantic embeddings using sentence-transformers.

#### Constructor

```python
def __init__(self, model_name: str = "all-MiniLM-L6-v2")
```

**Input Data:**
- `model_name: str` - Sentence-transformers model name

**Setup Data:**
- Internet connection for first-time model download
- ~100MB disk space for model weights

**Common Models:**
- `all-MiniLM-L6-v2`: Fast, 384 dimensions (default)
- `all-mpnet-base-v2`: Higher quality, 768 dimensions
- `paraphrase-multilingual-MiniLM-L12-v2`: Multilingual support

---

#### Method: `embed_text(text: str) -> np.ndarray`

**Building Block: Single Text Embedding**

**Input Data:**
- `text: str` - Text to embed

**Output Data:**
- `np.ndarray`: Embedding vector (shape: [embedding_dim])

**Example:**
```python
embedder = EmbeddingInterface()

embedding = embedder.embed_text("Hello, world!")
print(embedding.shape)  # (384,)
print(type(embedding))  # <class 'numpy.ndarray'>
```

---

#### Method: `embed_documents(texts: List[str]) -> List[np.ndarray]`

**Building Block: Batch Embedding**

**Input Data:**
- `texts: List[str]` - List of texts to embed

**Output Data:**
- `List[np.ndarray]`: List of embedding vectors

**Performance:**
- Batch processing is 5-10x faster than individual calls
- Recommended for >10 documents

**Example:**
```python
embedder = EmbeddingInterface()

docs = ["Document 1", "Document 2", "Document 3"]
embeddings = embedder.embed_documents(docs)

print(len(embeddings))        # 3
print(embeddings[0].shape)    # (384,)
```

---

### Class: RAGSystem

**Building Block: Retrieval-Augmented Generation**

Complete RAG implementation using ChromaDB for vector storage and retrieval.

#### Constructor

```python
def __init__(
    self,
    llm_interface: LLMInterface,
    embedding_interface: EmbeddingInterface,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    collection_name: str = "documents"
)
```

**Input Data:**
- `llm_interface: LLMInterface` - LLM for generation
- `embedding_interface: EmbeddingInterface` - Embeddings for retrieval
- `chunk_size: int` - Characters per chunk (default: 500)
- `chunk_overlap: int` - Overlap between chunks (default: 50)
- `collection_name: str` - ChromaDB collection name

**Setup Data:**
- Initialized LLM and embedding interfaces
- In-memory ChromaDB (no persistence)

---

#### Method: `add_documents(documents: List[str], metadatas: Optional[List[Dict]] = None)`

**Building Block: Document Indexing**

**Input Data:**
- `documents: List[str]` - Documents to index
- `metadatas: Optional[List[Dict]]` - Metadata for each document

**Output Data:**
- None (side effect: updates vector store)

**Processing:**
1. Split documents into chunks using RecursiveCharacterTextSplitter
2. Generate embeddings for all chunks
3. Store in ChromaDB with metadata

**Example:**
```python
rag = RAGSystem(llm, embedding)

docs = [
    "Paris is the capital of France.",
    "Python is a programming language.",
    "The Earth orbits the Sun."
]

metadatas = [
    {"topic": "geography"},
    {"topic": "programming"},
    {"topic": "astronomy"}
]

rag.add_documents(docs, metadatas)
# Added 3 chunks from 3 documents
```

---

#### Method: `retrieve(query: str, top_k: int = 3) -> List[Dict[str, Any]]`

**Building Block: Semantic Retrieval**

**Input Data:**
- `query: str` - Query text
- `top_k: int` - Number of chunks to retrieve

**Output Data:**
```python
[
    {
        "content": str,           # Chunk text
        "metadata": dict,         # Document metadata
        "similarity_score": float # Distance score
    },
    ...
]
```

**Raises:**
- `ValueError`: If no documents in vector store

**Example:**
```python
results = rag.retrieve("What is the capital of France?", top_k=2)

for doc in results:
    print(doc["content"])
    print(f"Score: {doc['similarity_score']:.3f}")
```

---

#### Method: `query_with_rag(query: str, top_k: int = 3) -> Dict[str, Any]`

**Building Block: RAG Query Pipeline**

**Input Data:**
- `query: str` - Question to answer
- `top_k: int` - Documents to retrieve

**Output Data:**
```python
{
    "response": str,              # LLM's answer
    "latency": float,             # LLM inference time
    "retrieve_time": float,       # Retrieval time
    "total_time": float,          # retrieve_time + latency
    "retrieved_docs": List[Dict], # Retrieved documents
    "num_docs_retrieved": int,    # Count of docs
    "success": bool,              # Success status
    "error": Optional[str]        # Error if failed
}
```

**Pipeline:**
1. Embed query
2. Retrieve top-k most similar chunks from ChromaDB
3. Concatenate chunks as context
4. Query LLM with context
5. Return response with timing metrics

**Example:**
```python
result = rag.query_with_rag("What is the capital of France?", top_k=3)

print(result["response"])
print(f"Total time: {result['total_time']:.2f}s")
print(f"Retrieved {result['num_docs_retrieved']} documents")
```

---

### Factory Functions

#### `create_llm_interface() -> LLMInterface`

Creates LLMInterface with default configuration.

```python
llm = create_llm_interface()
```

---

#### `create_rag_system(chunk_size: int = 500, chunk_overlap: int = 50) -> RAGSystem`

Creates complete RAG system with LLM and embeddings.

```python
rag = create_rag_system(chunk_size=500, chunk_overlap=50)
```

---

## Module: data_generator

**Location**: `src/data_generator.py`

### Class: DataGenerator

**Building Block: Synthetic Data Generation**

Generates realistic synthetic text data for all experiments.

#### Constructor

```python
def __init__(self, seed: int = 42)
```

**Input Data:**
- `seed: int` - Random seed for reproducibility

**Setup Data:**
- Faker library for text generation
- Both English and Hebrew locales

---

#### Method: `generate_filler_text(num_words: int, language: str = 'en') -> str`

**Building Block: Filler Text Generation**

**Input Data:**
- `num_words: int` - Target word count
- `language: str` - Language code ('en' or 'he')

**Output Data:**
- `str`: Generated filler text

**Characteristics:**
- Realistic sentence structure
- Mix of sentence lengths
- Approximately matches target word count (±5%)

**Example:**
```python
gen = DataGenerator(seed=42)

text = gen.generate_filler_text(num_words=100, language='en')
print(len(text.split()))  # Approximately 100
```

---

#### Method: `embed_fact_in_text(text: str, fact: str, position: str = 'middle') -> str`

**Building Block: Fact Injection**

**Input Data:**
- `text: str` - Base text
- `fact: str` - Fact to embed
- `position: str` - Position ('start', 'middle', 'end')

**Output Data:**
- `str`: Text with embedded fact

**Algorithm:**
1. Split text into sentences
2. Insert fact at specified position
3. Rejoin sentences

**Example:**
```python
gen = DataGenerator()

base = "Sentence 1. Sentence 2. Sentence 3."
result = gen.embed_fact_in_text(base, "SECRET FACT", position='middle')

print(result)
# "Sentence 1. SECRET FACT. Sentence 2. Sentence 3."
```

---

#### Method: `generate_needle_haystack_document(...) -> Dict[str, str]`

**Building Block: Experiment 1 Data**

**Input Data:**
- `words: int` - Document length (default: 200)
- `position: str` - Fact position ('start'|'middle'|'end')
- `secret_value: Optional[str]` - Secret to embed (random if None)

**Output Data:**
```python
{
    "document": str,        # Full document text
    "fact": str,            # Embedded fact sentence
    "position": str,        # Position used
    "secret_value": str     # Secret password value
}
```

**Example:**
```python
gen = DataGenerator(seed=42)

doc = gen.generate_needle_haystack_document(
    words=200,
    position='middle',
    secret_value='ABC123XYZ'
)

print("Secret:", doc["secret_value"])
print("Position:", doc["position"])
print(doc["document"][:100])
```

---

#### Method: `generate_context_size_document(...) -> Dict[str, str]`

**Building Block: Experiment 2 Data**

**Input Data:**
- `words: int` - Document length (default: 200)
- `revenue_value: Optional[str]` - Revenue fact (random if None)

**Output Data:**
```python
{
    "document": str,        # Business-themed document
    "revenue_value": str,   # Revenue amount (e.g., "$500 million")
    "company_name": str,    # Generated company name
    "year": int             # Fiscal year
}
```

**Example:**
```python
gen = DataGenerator()

doc = gen.generate_context_size_document(
    words=200,
    revenue_value="$500 million"
)

print(f"Company: {doc['company_name']}")
print(f"Revenue: {doc['revenue_value']}")
```

---

#### Method: `generate_hebrew_corpus(num_docs: int = 20, ...) -> List[Dict]`

**Building Block: Experiment 3 Data**

**Input Data:**
- `num_docs: int` - Number of Hebrew documents (default: 20)
- `save_dir: Optional[Path]` - Directory to save JSON

**Output Data:**
```python
[
    {
        "doc_id": str,            # Document ID
        "topic": str,             # "medicine" | "technology" | "law"
        "content": str,           # Hebrew text
        "drug_name": str,         # (if medicine topic)
        "side_effects": List[str] # (if medicine topic)
    },
    ...
]
```

**Topics:**
- **medicine**: Drug information with side effects
- **technology**: Technical content
- **law**: Legal text

**Example:**
```python
gen = DataGenerator(seed=42)

corpus = gen.generate_hebrew_corpus(num_docs=20)

for doc in corpus[:3]:
    print(f"{doc['doc_id']}: {doc['topic']}")
    print(doc['content'][:100])
```

---

## Module: evaluator

**Location**: `src/evaluator.py`

### Class: Evaluator

**Building Block: Response Evaluation**

Base evaluator for measuring LLM response accuracy.

#### Constructor

```python
def __init__(
    self,
    llm_interface: Optional[LLMInterface] = None,
    embedding_interface: Optional[EmbeddingInterface] = None
)
```

**Input Data:**
- `llm_interface: Optional[LLMInterface]` - LLM (not used in base class)
- `embedding_interface: Optional[EmbeddingInterface]` - For semantic similarity

---

#### Method: `evaluate_response(...) -> Dict[str, float]`

**Building Block: Single Response Evaluation**

**Input Data:**
- `response: str` - LLM response to evaluate
- `expected_answer: str` - Ground truth answer
- `keywords: Optional[List[str]]` - Keywords to check for
- `extract_pattern: Optional[str]` - Regex to extract answer

**Output Data:**
```python
{
    "exact_match": float,         # 1.0 if exact match, else 0.0
    "partial_match": float,       # Sequence similarity (0.0-1.0)
    "keyword_match": float,       # Proportion of keywords found
    "semantic_similarity": float, # Embedding similarity (if available)
    "overall_score": float,       # Weighted average
    "extracted_answer": str       # Extracted answer for debugging
}
```

**Scoring Weights:**
- Exact match: 40%
- Partial match: 30%
- Keyword match: 15%
- Semantic similarity: 15%

**Example:**
```python
from evaluator import create_evaluator

evaluator = create_evaluator(use_embeddings=True)

metrics = evaluator.evaluate_response(
    response="The secret password is ABC123XYZ.",
    expected_answer="ABC123XYZ",
    keywords=["ABC123XYZ"]
)

print(f"Overall score: {metrics['overall_score']:.2f}")
print(f"Exact match: {metrics['exact_match']}")
```

---

#### Method: `evaluate_multiple_responses(...) -> Dict[str, Any]`

**Building Block: Multi-Trial Evaluation**

**Input Data:**
- `responses: List[str]` - List of responses from multiple trials
- `expected_answer: str` - Ground truth
- `keywords: Optional[List[str]]` - Keywords to check

**Output Data:**
```python
{
    "exact_match": {
        "mean": float,
        "std": float,
        "min": float,
        "max": float,
        "median": float,
        "ci_lower": float,
        "ci_upper": float
    },
    "partial_match": { ... },
    "overall_score": { ... },
    ...
}
```

**Example:**
```python
evaluator = create_evaluator()

responses = [
    "The answer is 42.",
    "I think it's 42.",
    "The value is 42."
]

stats = evaluator.evaluate_multiple_responses(
    responses=responses,
    expected_answer="42",
    keywords=["42"]
)

print(f"Mean accuracy: {stats['overall_score']['mean']:.2f}")
print(f"Std dev: {stats['overall_score']['std']:.2f}")
```

---

### Class: ExperimentEvaluator

**Building Block: Experiment-Specific Evaluation**

Extends `Evaluator` with experiment-specific methods.

#### Method: `evaluate_needle_haystack(...) -> Dict[str, Any]`

**Experiment 1 Evaluator**

**Input Data:**
- `response: str` - LLM response
- `secret_value: str` - Expected secret password
- `position: str` - Fact position

**Output Data:**
```python
{
    # All metrics from evaluate_response()
    "position": str,   # Position used
    "correct": bool    # True if exact_match > 0.5 or partial_match > 0.8
}
```

---

#### Method: `evaluate_context_size(...) -> Dict[str, Any]`

**Experiment 2 Evaluator**

**Input Data:**
- `response: str` - LLM response
- `revenue_value: str` - Expected revenue
- `num_docs: int` - Number of documents in context
- `latency: float` - Response time

**Output Data:**
```python
{
    # All metrics from evaluate_response()
    "num_docs": int,
    "latency": float,
    "correct": bool    # True if partial_match > 0.6
}
```

---

#### Method: `evaluate_rag_response(...) -> Dict[str, Any]`

**Experiment 3 Evaluator**

**Input Data:**
- `response: str` - LLM response
- `expected_answer: str` - Expected answer
- `latency: float` - Response time
- `method: str` - "rag" or "full_context"

**Output Data:**
```python
{
    # All metrics from evaluate_response()
    "method": str,
    "latency": float,
    "correct": bool    # True if overall_score > 0.5
}
```

---

#### Method: `compare_groups(...) -> Dict[str, Any]`

**Statistical Comparison**

**Input Data:**
- `group1_results: List[Dict]` - Results from group 1
- `group2_results: List[Dict]` - Results from group 2
- `metric_name: str` - Metric to compare (default: "overall_score")

**Output Data:**
```python
{
    "t_statistic": float,   # t-test statistic
    "p_value": float,       # p-value
    "significant": bool,    # True if p < 0.05
    "effect_size": float,   # Cohen's d
    "group1_mean": float,
    "group2_mean": float,
    "group1_std": float,
    "group2_std": float,
    "difference": float     # group1_mean - group2_mean
}
```

**Example:**
```python
evaluator = ExperimentEvaluator()

rag_results = [{"overall_score": 0.85}, {"overall_score": 0.82}]
full_results = [{"overall_score": 0.65}, {"overall_score": 0.68}]

comparison = evaluator.compare_groups(
    group1_results=rag_results,
    group2_results=full_results,
    metric_name="overall_score"
)

print(f"p-value: {comparison['p_value']:.4f}")
print(f"Significant: {comparison['significant']}")
print(f"Effect size: {comparison['effect_size']:.2f}")
```

---

## Module: utils.metrics

**Location**: `src/utils/metrics.py`

### Functions

#### `exact_match(predicted: str, expected: str) -> float`

**Building Block: Exact Matching**

**Input Data:**
- `predicted: str` - Predicted answer
- `expected: str` - Expected answer

**Output Data:**
- `float`: 1.0 if exact match (case-insensitive, whitespace-normalized), else 0.0

**Example:**
```python
from utils.metrics import exact_match

score = exact_match("ABC123", "abc123")
print(score)  # 1.0

score = exact_match("ABC123", "XYZ789")
print(score)  # 0.0
```

---

#### `partial_match(predicted: str, expected: str) -> float`

**Building Block: Fuzzy Matching**

**Input Data:**
- `predicted: str` - Predicted answer
- `expected: str` - Expected answer

**Output Data:**
- `float`: Sequence similarity score (0.0-1.0) using Levenshtein-like algorithm

**Algorithm:**
Uses Python's `difflib.SequenceMatcher` for longest contiguous matching subsequence.

**Example:**
```python
from utils.metrics import partial_match

score = partial_match("ABC123XYZ", "ABC123")
print(score)  # Approximately 0.67

score = partial_match("Hello World", "Hello Wrld")
print(score)  # Approximately 0.9
```

---

#### `keyword_match(predicted: str, keywords: List[str]) -> float`

**Building Block: Keyword Presence**

**Input Data:**
- `predicted: str` - Predicted answer
- `keywords: List[str]` - Keywords to check for

**Output Data:**
- `float`: Proportion of keywords found (0.0-1.0)

**Example:**
```python
from utils.metrics import keyword_match

score = keyword_match(
    "Paris is the capital of France",
    keywords=["Paris", "France", "capital"]
)
print(score)  # 1.0 (all keywords found)

score = keyword_match(
    "Paris is beautiful",
    keywords=["Paris", "France", "capital"]
)
print(score)  # 0.33 (1 of 3 keywords found)
```

---

#### `semantic_similarity(predicted_embedding: np.ndarray, expected_embedding: np.ndarray) -> float`

**Building Block: Semantic Matching**

**Input Data:**
- `predicted_embedding: np.ndarray` - Embedding of prediction
- `expected_embedding: np.ndarray` - Embedding of expected answer

**Output Data:**
- `float`: Cosine similarity normalized to [0, 1]

**Formula:**
```
similarity = (cosine_similarity + 1) / 2
```

**Example:**
```python
from utils.metrics import semantic_similarity
from llm_interface import EmbeddingInterface

embedder = EmbeddingInterface()

emb1 = embedder.embed_text("Paris is the capital of France")
emb2 = embedder.embed_text("The capital of France is Paris")

score = semantic_similarity(emb1, emb2)
print(score)  # Approximately 0.95 (very similar)
```

---

#### `calculate_statistics(values: List[float]) -> Dict[str, float]`

**Building Block: Statistical Summary**

**Input Data:**
- `values: List[float]` - Numerical values

**Output Data:**
```python
{
    "mean": float,      # Average
    "std": float,       # Standard deviation
    "min": float,       # Minimum
    "max": float,       # Maximum
    "median": float,    # Median
    "ci_lower": float,  # 95% CI lower bound
    "ci_upper": float   # 95% CI upper bound
}
```

**Example:**
```python
from utils.metrics import calculate_statistics

values = [0.8, 0.85, 0.82, 0.9, 0.78]
stats = calculate_statistics(values)

print(f"Mean: {stats['mean']:.2f}")
print(f"95% CI: [{stats['ci_lower']:.2f}, {stats['ci_upper']:.2f}]")
```

---

#### `perform_t_test(group1: List[float], group2: List[float]) -> Dict[str, Any]`

**Building Block: Statistical Significance Testing**

**Input Data:**
- `group1: List[float]` - First group of values
- `group2: List[float]` - Second group of values

**Output Data:**
```python
{
    "t_statistic": float,  # t-test statistic
    "p_value": float,      # p-value
    "significant": bool,   # True if p < 0.05
    "effect_size": float   # Cohen's d (absolute value)
}
```

**Statistical Test:**
- Independent samples t-test
- Significance level: α = 0.05
- Effect size: Cohen's d

**Example:**
```python
from utils.metrics import perform_t_test

group_a = [0.8, 0.85, 0.82, 0.88, 0.79]
group_b = [0.6, 0.65, 0.62, 0.68, 0.59]

result = perform_t_test(group_a, group_b)

print(f"t-statistic: {result['t_statistic']:.2f}")
print(f"p-value: {result['p_value']:.4f}")
print(f"Significant: {result['significant']}")
print(f"Effect size (Cohen's d): {result['effect_size']:.2f}")
```

---

#### `extract_answer_from_response(response: str, pattern: str = None) -> str`

**Building Block: Answer Extraction**

**Input Data:**
- `response: str` - Full LLM response text
- `pattern: str` - Optional regex pattern

**Output Data:**
- `str`: Extracted answer

**Extraction Strategy:**
1. If pattern provided, use regex
2. Try common prefixes ("the answer is:", "password:", etc.)
3. Extract first sentence
4. Fallback: return trimmed full response

**Example:**
```python
from utils.metrics import extract_answer_from_response

response = "The secret password is ABC123. Remember to keep it safe."

answer = extract_answer_from_response(response)
print(answer)  # "ABC123"

# With custom pattern
answer = extract_answer_from_response(
    response,
    pattern=r"password is (\w+)"
)
print(answer)  # "ABC123"
```

---

## Module: utils.visualization

**Location**: `src/utils/visualization.py`

### Functions

#### `setup_plot_style()`

**Building Block: Plot Styling**

Configures matplotlib/seaborn with publication-quality defaults.

**Setup Data:**
- `PLOT_CONFIG` from config module

**Side Effects:**
- Sets global matplotlib rcParams
- Configures seaborn style and palette

**Example:**
```python
from utils.visualization import setup_plot_style

setup_plot_style()
# All subsequent plots use consistent styling
```

---

#### `plot_accuracy_by_position(...)`

**Building Block: Experiment 1 Visualization**

**Input Data:**
- `results: Dict[str, List[float]]` - Position → accuracy scores
- `save_path: Optional[Path]` - Where to save figure
- `title: str` - Plot title (default: "Accuracy by Fact Position")

**Output Data:**
- PNG file saved to `save_path` (if provided)
- Figure closed after saving

**Visualization:**
- Bar chart with error bars (standard deviation)
- Value labels on bars
- Publication-quality formatting (300 DPI)

**Example:**
```python
from pathlib import Path
from utils.visualization import plot_accuracy_by_position

results = {
    "start": [0.9, 0.85, 0.88, 0.92],
    "middle": [0.6, 0.65, 0.58, 0.62],
    "end": [0.85, 0.88, 0.82, 0.87]
}

plot_accuracy_by_position(
    results=results,
    save_path=Path("results/exp1/accuracy_by_position.png"),
    title="Lost in the Middle - Position Impact"
)
```

---

#### `plot_context_size_impact(...)`

**Building Block: Experiment 2 Visualization**

**Input Data:**
- `results: List[Dict[str, Any]]` - List with num_docs, accuracy, latency, tokens
- `save_path: Optional[Path]` - Where to save figure
- `title: str` - Plot title

**Output Data:**
- 3-panel figure (PNG):
  1. Accuracy vs Context Size
  2. Latency vs Context Size
  3. Token Consumption vs Context Size

**Visualization Features:**
- Line plots with confidence intervals (shaded regions)
- Separate colors for each metric
- Grid and legend

**Example:**
```python
from utils.visualization import plot_context_size_impact

results = [
    {"num_docs": 2, "accuracy_mean": 0.9, "accuracy_std": 0.05,
     "latency_mean": 0.8, "latency_std": 0.1, "tokens_used": 800},
    {"num_docs": 5, "accuracy_mean": 0.75, "accuracy_std": 0.08,
     "latency_mean": 1.2, "latency_std": 0.15, "tokens_used": 2000},
    # ...
]

plot_context_size_impact(
    results=results,
    save_path=Path("results/exp2/context_size_impact.png")
)
```

---

#### `plot_rag_comparison(...)`

**Building Block: Experiment 3 Visualization**

**Input Data:**
- `full_context_results: Dict[str, float]` - Metrics for full context
- `rag_results: Dict[str, float]` - Metrics for RAG
- `save_path: Optional[Path]` - Where to save figure
- `title: str` - Plot title

**Output Data:**
- 2-panel figure (PNG):
  1. Normalized metrics comparison (accuracy + speed)
  2. Absolute latency comparison

**Example:**
```python
from utils.visualization import plot_rag_comparison

full_context = {"accuracy": 0.65, "latency": 2.5}
rag = {"accuracy": 0.82, "latency": 1.2}

plot_rag_comparison(
    full_context_results=full_context,
    rag_results=rag,
    save_path=Path("results/exp3/rag_comparison.png")
)
```

---

#### `plot_strategy_comparison(...)`

**Building Block: Experiment 4 Visualization**

**Input Data:**
- `results: Dict[str, List[Dict]]` - Strategy name → list of step results
- `save_path: Optional[Path]` - Where to save figure
- `title: str` - Plot title

**Output Data:**
- Multi-line plot showing accuracy over time for each strategy

**Visualization:**
- Different colors and markers for each strategy
- Legend with strategy names
- Grid and axis labels

**Example:**
```python
from utils.visualization import plot_strategy_comparison

results = {
    "select": [
        {"step": 0, "accuracy": 0.8},
        {"step": 1, "accuracy": 0.82},
        # ...
    ],
    "compress": [
        {"step": 0, "accuracy": 0.75},
        {"step": 1, "accuracy": 0.73},
        # ...
    ],
    "write": [
        {"step": 0, "accuracy": 0.85},
        {"step": 1, "accuracy": 0.88},
        # ...
    ]
}

plot_strategy_comparison(
    results=results,
    save_path=Path("results/exp4/strategy_comparison.png")
)
```

---

## Building Block Specifications

This section summarizes the Building Block pattern (Input/Output/Setup Data) for key components as required by the guidelines.

### Building Block: Data Generator

**Purpose**: Generate synthetic documents with embedded facts

**Input Data:**
- `num_documents: int` - How many documents to generate
- `words_per_document: int` - Target length of each document
- `position: str` - Where to embed fact ('start'|'middle'|'end')
- `language: str` - Language code ('en'|'he')

**Output Data:**
- `List[Dict]` - List of document dictionaries containing:
  - `document: str` - Full text
  - `fact: str` - Embedded fact
  - `secret_value: str` - Ground truth answer

**Setup Data:**
- `seed: int = 42` - Random seed for reproducibility
- Faker library with English and Hebrew locales
- Topic templates and word pools

**Validation:**
- `num_documents` must be > 0
- `words_per_document` must be in range [50, 1000]
- `position` must be in ['start', 'middle', 'end']
- `language` must be in ['en', 'he']

**Error Handling:**
- Raises `ValueError` on invalid input
- Degrades gracefully if Faker fails (uses simple templates)

---

### Building Block: LLM Query Interface

**Purpose**: Query LLM with context and measure latency

**Input Data:**
- `prompt: str` - Question or instruction
- `context: Optional[str]` - Additional context to prepend
- `temperature: float` - Sampling temperature [0.0, 2.0]

**Output Data:**
- `Dict` with:
  - `response: str` - Generated text
  - `latency: float` - Response time in seconds
  - `success: bool` - Whether query succeeded
  - `error: Optional[str]` - Error message if failed

**Setup Data:**
- Ollama server running at configured URL
- Model downloaded and available
- Network connectivity

**Validation:**
- `prompt` must be non-empty string
- `temperature` must be in [0.0, 2.0]

**Error Handling:**
- Returns `success=False` instead of raising exceptions
- Includes error message for debugging
- Captures network, timeout, and parsing errors

---

### Building Block: RAG System

**Purpose**: Retrieve relevant context and generate answer

**Input Data:**
- `documents: List[str]` - Documents to index
- `query: str` - Question to answer
- `top_k: int` - Number of documents to retrieve [1, 10]

**Output Data:**
- `Dict` with:
  - `response: str` - Generated answer
  - `retrieved_docs: List[Dict]` - Retrieved chunks with scores
  - `total_time: float` - Retrieval + generation time

**Setup Data:**
- ChromaDB in-memory vector store
- Sentence-transformers embedding model
- LLM interface initialized
- Text splitter configured (chunk_size, overlap)

**Pipeline:**
1. **Indexing Phase**:
   - Split documents into chunks
   - Generate embeddings for all chunks
   - Store in ChromaDB with metadata

2. **Retrieval Phase**:
   - Embed query
   - Search for top-k similar chunks
   - Return chunks with similarity scores

3. **Generation Phase**:
   - Concatenate retrieved chunks as context
   - Query LLM with context
   - Return response with timing

**Validation:**
- `top_k` must be ≤ total number of indexed chunks
- `documents` must be non-empty
- `query` must be non-empty string

**Error Handling:**
- Raises `ValueError` if querying before indexing
- Handles embedding failures gracefully
- Returns partial results if LLM fails

---

### Building Block: Accuracy Evaluator

**Purpose**: Measure LLM response accuracy with multiple metrics

**Input Data:**
- `response: str` - LLM's generated response
- `expected: str` - Ground truth answer
- `keywords: Optional[List[str]]` - Keywords to check for

**Output Data:**
- `Dict[str, float]` with:
  - `exact_match: float` - Binary exact match (0.0 or 1.0)
  - `partial_match: float` - Sequence similarity [0.0, 1.0]
  - `keyword_match: float` - Proportion of keywords found
  - `semantic_similarity: float` - Embedding similarity
  - `overall_score: float` - Weighted combination

**Setup Data:**
- Embedding interface (optional, for semantic similarity)
- Regex patterns for answer extraction
- Scoring weights for overall score

**Metrics Calculation:**
1. **Exact Match**: Normalize whitespace/case, compare strings
2. **Partial Match**: Use SequenceMatcher for fuzzy matching
3. **Keyword Match**: Count keyword occurrences
4. **Semantic Similarity**: Cosine similarity of embeddings
5. **Overall Score**: Weighted average of available metrics

**Weights:**
- Exact match: 0.4
- Partial match: 0.3
- Keyword match: 0.15
- Semantic similarity: 0.15

**Validation:**
- All text inputs must be strings
- `keywords` must be list of strings (if provided)

**Error Handling:**
- Handles missing embeddings (skips semantic similarity)
- Returns 0.0 scores if evaluation fails
- Logs warnings for missing metrics

---

## Error Handling

### Exception Hierarchy

```python
# Base exceptions
ValueError          # Invalid input parameters
TypeError          # Wrong type passed
RuntimeError       # Runtime failures (e.g., model not found)

# Network exceptions
ConnectionError    # Cannot connect to Ollama
TimeoutError       # Request timeout

# Data exceptions
KeyError           # Missing required key in dict
IndexError         # Out of bounds access
```

### Common Error Patterns

#### LLM Query Failure

```python
result = llm.query("Question?")

if not result["success"]:
    print(f"Error: {result['error']}")
    # Fallback logic here
```

#### Missing Embeddings

```python
try:
    embedder = EmbeddingInterface()
except Exception as e:
    print(f"Warning: Embeddings unavailable: {e}")
    embedder = None  # Continue without semantic similarity
```

#### RAG Without Documents

```python
try:
    results = rag.retrieve("query", top_k=3)
except ValueError as e:
    print(f"Error: {e}")  # "No documents in vector store"
    # Add documents first
    rag.add_documents(docs)
```

---

## Usage Examples

### Complete Example: Experiment 1

```python
from config import EXP1_CONFIG, set_random_seeds
from data_generator import DataGenerator
from llm_interface import create_llm_interface
from evaluator import create_evaluator
from utils.visualization import plot_accuracy_by_position
from pathlib import Path

# Setup
set_random_seeds(42)

# Generate data
gen = DataGenerator(seed=42)
doc_data = gen.generate_needle_haystack_document(
    words=200,
    position='middle'
)

# Query LLM
llm = create_llm_interface()
result = llm.query(
    prompt="What is the secret password mentioned in the documents?",
    context=doc_data["document"]
)

# Evaluate
evaluator = create_evaluator(use_embeddings=True)
metrics = evaluator.evaluate_response(
    response=result["response"],
    expected_answer=doc_data["secret_value"],
    keywords=[doc_data["secret_value"]]
)

print(f"Overall accuracy: {metrics['overall_score']:.2f}")
print(f"Latency: {result['latency']:.2f}s")

# Visualize (after running multiple trials)
results_by_position = {
    "start": [0.9, 0.85, 0.88],
    "middle": [0.6, 0.65, 0.62],
    "end": [0.85, 0.88, 0.82]
}

plot_accuracy_by_position(
    results=results_by_position,
    save_path=Path("results/exp1/accuracy_by_position.png")
)
```

---

### Complete Example: RAG Experiment

```python
from llm_interface import create_rag_system
from data_generator import DataGenerator
from evaluator import create_evaluator

# Generate Hebrew corpus
gen = DataGenerator(seed=42)
corpus = gen.generate_hebrew_corpus(num_docs=20)

# Create RAG system
rag = create_rag_system(chunk_size=500, chunk_overlap=50)

# Index documents
documents = [doc["content"] for doc in corpus]
metadatas = [{"doc_id": doc["doc_id"], "topic": doc["topic"]}
             for doc in corpus]

rag.add_documents(documents, metadatas)

# Query with RAG
question = "What are the side effects of אספירין?"
result = rag.query_with_rag(question, top_k=3)

print(f"Response: {result['response']}")
print(f"Total time: {result['total_time']:.2f}s")
print(f"Retrieved {result['num_docs_retrieved']} documents")

# Show retrieved documents
for i, doc in enumerate(result['retrieved_docs'], 1):
    print(f"\nDocument {i} (score: {doc['similarity_score']:.3f}):")
    print(doc['content'][:200])
```

---

### Complete Example: Statistical Comparison

```python
from evaluator import create_evaluator
from utils.metrics import perform_t_test, calculate_statistics

# Simulate results from two methods
rag_accuracies = [0.85, 0.82, 0.88, 0.90, 0.84]
full_context_accuracies = [0.65, 0.68, 0.62, 0.70, 0.67]

# Calculate statistics
rag_stats = calculate_statistics(rag_accuracies)
full_stats = calculate_statistics(full_context_accuracies)

print("RAG Approach:")
print(f"  Mean: {rag_stats['mean']:.2f} ± {rag_stats['std']:.2f}")
print(f"  95% CI: [{rag_stats['ci_lower']:.2f}, {rag_stats['ci_upper']:.2f}]")

print("\nFull Context Approach:")
print(f"  Mean: {full_stats['mean']:.2f} ± {full_stats['std']:.2f}")
print(f"  95% CI: [{full_stats['ci_lower']:.2f}, {full_stats['ci_upper']:.2f}]")

# Statistical test
test_result = perform_t_test(rag_accuracies, full_context_accuracies)

print(f"\nStatistical Comparison:")
print(f"  t-statistic: {test_result['t_statistic']:.2f}")
print(f"  p-value: {test_result['p_value']:.4f}")
print(f"  Significant: {test_result['significant']}")
print(f"  Effect size: {test_result['effect_size']:.2f}")

if test_result['significant']:
    print("\n✅ RAG shows statistically significant improvement!")
else:
    print("\n⚠️  No statistically significant difference detected.")
```

---

## API Versioning and Compatibility

**Current Version**: 1.0.0

**Python Compatibility**: 3.10, 3.11, 3.12

**Breaking Changes Policy**:
- Major version (X.0.0): Breaking API changes
- Minor version (1.X.0): New features, backward compatible
- Patch version (1.0.X): Bug fixes only

**Deprecation Policy**:
- Features marked deprecated remain for 2 minor versions
- Warnings logged when using deprecated features

---

## Performance Considerations

### Optimization Guidelines

1. **Batch Processing**:
   ```python
   # Good: Batch embedding
   embeddings = embedder.embed_documents(texts)

   # Bad: Individual calls
   embeddings = [embedder.embed_text(t) for t in texts]
   ```

2. **RAG Chunking**:
   - Smaller chunks (200-300 tokens): Better retrieval precision
   - Larger chunks (500-800 tokens): More context per chunk
   - Recommended: 500 tokens with 50 overlap

3. **Caching**:
   - Cache embeddings for reused documents
   - Reuse LLM interface instances

### Typical Performance

| Operation | Time (avg) | Notes |
|-----------|-----------|-------|
| LLM query (200 words context) | 0.8-1.2s | llama2 on M1 Mac |
| Embedding single text | 5-10ms | all-MiniLM-L6-v2 |
| Embedding batch (100 docs) | 0.3-0.5s | GPU accelerated |
| RAG retrieval (20 docs) | 20-50ms | In-memory ChromaDB |
| RAG end-to-end | 1.0-1.5s | Retrieval + generation |

---

## Thread Safety

**Thread-Safe Components**:
- `EmbeddingInterface` (read-only model)
- Metric calculation functions
- Visualization functions

**Not Thread-Safe**:
- `DataGenerator` (shared Faker instances)
- `RAGSystem` (mutable vector store)
- `config.set_random_seeds()` (global state)

**Recommendation**: Create separate instances per thread or use process-based parallelism.

---

## Testing

### Unit Testing

```python
import pytest
from llm_interface import LLMInterface

def test_llm_query():
    llm = LLMInterface(temperature=0.0)
    result = llm.query("What is 2+2?")

    assert result["success"] is True
    assert "4" in result["response"].lower()
    assert result["latency"] > 0
```

### Integration Testing

```python
def test_rag_pipeline():
    rag = create_rag_system()

    docs = ["Paris is the capital of France."]
    rag.add_documents(docs)

    result = rag.query_with_rag("What is the capital of France?", top_k=1)

    assert result["success"] is True
    assert len(result["retrieved_docs"]) == 1
    assert "paris" in result["response"].lower()
```

---

## API Change Log

### Version 1.0.0 (2025-12-06)
- Initial release
- All 4 experiments implemented
- Complete LLM, RAG, evaluation, and visualization APIs
- Building block pattern documentation

---

**For questions or issues, please refer to the project README or contact the development team.**
