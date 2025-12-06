"""
Pytest configuration and shared fixtures for testing.

This module provides mocks and fixtures to ensure tests run without
external dependencies (Ollama, embeddings, ChromaDB, etc.).
"""

import pytest
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, MagicMock, patch


# ============================================================================
# MOCK LLM INTERFACE
# ============================================================================

class MockLLM:
    """Mock LLM that returns deterministic responses without calling Ollama."""

    def __init__(self, model_name: str = "mock-llama2", temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature
        self.call_count = 0
        self.last_prompt = None

    def invoke(self, prompt: str) -> str:
        """Return deterministic response based on prompt keywords."""
        self.call_count += 1
        self.last_prompt = prompt

        prompt_lower = prompt.lower()

        # Needle in haystack responses
        if "secret password" in prompt_lower:
            if "ABC123XYZ" in prompt:
                return "The secret password is ABC123XYZ."
            elif "password" in prompt_lower:
                return "The secret password mentioned is TEST12345."

        # Context size / revenue responses
        if "revenue" in prompt_lower or "annual revenue" in prompt_lower:
            if "$500 million" in prompt:
                return "The company's annual revenue is $500 million."
            else:
                return "The annual revenue mentioned is $250 million."

        # Hebrew medical responses
        if "תופעות לוואי" in prompt or "side effects" in prompt_lower:
            return "תופעות הלוואי כוללות: כאב ראש, בחילה, סחרחורת."

        # RAG responses
        if "capital of france" in prompt_lower:
            return "The capital of France is Paris."

        # Default response
        return "This is a mock response to the query."


class MockLLMInterface:
    """Mock for src.llm_interface.LLMInterface."""

    def __init__(self, model_name: Optional[str] = None, temperature: float = 0.0):
        self.model_name = model_name or "mock-llama2"
        self.temperature = temperature
        self.llm = MockLLM(model_name=self.model_name, temperature=temperature)

    def query(self, prompt: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Mock query method."""
        if context:
            full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}\n\nAnswer:"
        else:
            full_prompt = prompt

        response = self.llm.invoke(full_prompt)

        return {
            "response": response,
            "latency": 0.05,  # Deterministic latency
            "model": self.model_name,
            "success": True,
            "error": None,
        }

    def query_with_template(self, template: str, **kwargs) -> Dict[str, Any]:
        """Mock template query."""
        prompt = template.format(**kwargs)
        return self.query(prompt)

    def count_tokens(self, text: str) -> int:
        """Estimate token count (4 chars per token)."""
        return len(text) // 4


# ============================================================================
# MOCK EMBEDDING INTERFACE
# ============================================================================

class MockEmbeddingInterface:
    """Mock for src.llm_interface.EmbeddingInterface."""

    def __init__(self, model_name: str = "mock-embeddings", embedding_dim: int = 384):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.call_count = 0

    def embed_text(self, text: str) -> np.ndarray:
        """Generate deterministic embedding based on text hash."""
        self.call_count += 1

        # Use hash to create deterministic but unique embeddings
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(self.embedding_dim).astype(np.float32)

        # Normalize
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        embeddings = np.array([self.embed_text(text) for text in texts])
        return embeddings


# ============================================================================
# MOCK RAG SYSTEM
# ============================================================================

class MockVectorStore:
    """Mock for ChromaDB vector store."""

    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.metadatas = []

    def add_documents(self, documents, embeddings, metadatas):
        """Add documents to mock store."""
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        self.metadatas.extend(metadatas)

    def similarity_search_with_score(self, query: str, k: int = 3):
        """Mock similarity search - returns first k documents."""
        results = []

        for i, doc in enumerate(self.documents[:k]):
            mock_doc = MagicMock()
            mock_doc.page_content = doc
            mock_doc.metadata = self.metadatas[i] if i < len(self.metadatas) else {}

            # Mock similarity score (higher is more similar in ChromaDB)
            score = 0.9 - (i * 0.1)

            results.append((mock_doc, score))

        return results


class MockRAGSystem:
    """Mock for src.llm_interface.RAGSystem."""

    def __init__(self, llm_interface, embedding_interface,
                 chunk_size: int = 500, chunk_overlap: int = 50,
                 collection_name: str = "documents"):
        self.llm = llm_interface
        self.embedding = embedding_interface
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = collection_name
        self.vectorstore = MockVectorStore()

    def add_documents(self, documents: List[str],
                     metadatas: Optional[List[Dict[str, Any]]] = None):
        """Mock add documents."""
        embeddings = self.embedding.embed_documents(documents)

        if metadatas is None:
            metadatas = [{} for _ in documents]

        self.vectorstore.add_documents(documents, embeddings, metadatas)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Mock retrieval."""
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)

        retrieved_docs = []
        for doc, score in results:
            retrieved_docs.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": float(score),
            })

        return retrieved_docs

    def query_with_rag(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Mock RAG query."""
        retrieved_docs = self.retrieve(query, top_k=top_k)
        context = "\n\n".join([doc["content"] for doc in retrieved_docs])

        llm_result = self.llm.query(prompt=query, context=context)

        return {
            "response": llm_result["response"],
            "latency": llm_result["latency"],
            "retrieve_time": 0.01,
            "total_time": 0.06,
            "retrieved_docs": retrieved_docs,
            "num_docs_retrieved": len(retrieved_docs),
            "success": True,
            "error": None,
        }


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_llm():
    """Provide mock LLM interface."""
    return MockLLMInterface()


@pytest.fixture
def mock_embedding():
    """Provide mock embedding interface."""
    return MockEmbeddingInterface()


@pytest.fixture
def mock_rag(mock_llm, mock_embedding):
    """Provide mock RAG system."""
    return MockRAGSystem(
        llm_interface=mock_llm,
        embedding_interface=mock_embedding,
        chunk_size=500,
        chunk_overlap=50,
    )


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    return [
        "The capital of France is Paris. It is known for the Eiffel Tower.",
        "Python is a programming language. It is widely used for data science.",
        "The Earth orbits around the Sun. It takes approximately 365 days.",
        "Machine learning is a subset of artificial intelligence.",
        "The secret password is ABC123XYZ. Remember this code.",
    ]


@pytest.fixture
def sample_hebrew_documents():
    """Provide sample Hebrew documents."""
    return [
        "פנדול היא תרופה נגד כאבים. תופעות הלוואי כוללות: כאב ראש, בחילה.",
        "ישראל היא מדינה במזרח התיכון. הבירה שלה היא ירושלים.",
        "פייתון היא שפת תכנות פופולרית. היא משמשת לניתוח נתונים.",
    ]


@pytest.fixture
def sample_needle_haystack_data():
    """Provide sample data for needle in haystack experiment."""
    return {
        "document": "This is filler text. The secret password is TEST12345. More filler text here.",
        "fact": "The secret password is TEST12345.",
        "position": "middle",
        "secret_value": "TEST12345",
        "doc_id": "middle_0",
    }


@pytest.fixture
def sample_context_size_data():
    """Provide sample data for context size experiment."""
    return {
        "document": "The company Acme Corp reported annual revenue of $500 million for 2023.",
        "revenue_value": "$500 million",
        "company_name": "Acme Corp",
        "year": 2023,
        "doc_id": "size_5_doc_0",
    }


@pytest.fixture
def sample_responses():
    """Provide sample LLM responses for evaluation."""
    return {
        "correct": "The secret password is TEST12345.",
        "partial": "The password mentioned is TEST12345",
        "incorrect": "I don't know the password.",
        "with_context": "Based on the document, the secret password is TEST12345.",
    }


@pytest.fixture
def sample_embeddings():
    """Provide sample embeddings for testing."""
    np.random.seed(42)
    return {
        "embedding_1": np.random.randn(384).astype(np.float32),
        "embedding_2": np.random.randn(384).astype(np.float32),
        "similar": np.array([1.0] * 384, dtype=np.float32),
        "similar_copy": np.array([1.0] * 384, dtype=np.float32),
    }


@pytest.fixture
def temp_directory():
    """Provide temporary directory for file I/O tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_statistical_data():
    """Provide sample data for statistical tests."""
    np.random.seed(42)
    return {
        "group1": [0.8, 0.85, 0.9, 0.88, 0.82, 0.87, 0.91, 0.86, 0.89, 0.84],
        "group2": [0.6, 0.65, 0.62, 0.58, 0.63, 0.61, 0.59, 0.64, 0.60, 0.66],
        "single_group": [0.75, 0.80, 0.78, 0.82, 0.77, 0.79, 0.81, 0.76, 0.83, 0.74],
    }


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    import random
    random.seed(42)
    np.random.seed(42)


# ============================================================================
# MOCK PATCHES
# ============================================================================

@pytest.fixture
def patch_ollama_llm(monkeypatch):
    """Patch LangChain OllamaLLM to use mock."""
    original_init = None

    def mock_ollama_init(self, **kwargs):
        # Call object.__init__ to avoid issues
        object.__setattr__(self, 'model', kwargs.get('model', 'mock-llama2'))
        object.__setattr__(self, 'base_url', kwargs.get('base_url', 'http://localhost:11434'))
        object.__setattr__(self, 'temperature', kwargs.get('temperature', 0.0))
        object.__setattr__(self, '_mock_llm', MockLLM(model_name=self.model, temperature=self.temperature))
        # Return None as __init__ should
        return None

    def mock_invoke(self, prompt):
        if hasattr(self, '_mock_llm'):
            return self._mock_llm.invoke(prompt)
        else:
            # Fallback
            return MockLLM().invoke(prompt)

    try:
        from langchain_ollama import OllamaLLM
        monkeypatch.setattr(OllamaLLM, "__init__", mock_ollama_init)
        monkeypatch.setattr(OllamaLLM, "invoke", mock_invoke)
    except ImportError:
        pass


@pytest.fixture
def patch_sentence_transformer(monkeypatch):
    """Patch SentenceTransformer to use mock embeddings."""
    def mock_st_init(self, model_name):
        self.model_name = model_name
        self.embedding_dim = 384

    def mock_encode(self, texts, convert_to_numpy=False, **kwargs):
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            np.random.seed(hash(text) % (2**32))
            emb = np.random.randn(self.embedding_dim).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)

        result = np.array(embeddings)

        if len(texts) == 1:
            result = result[0]

        return result

    monkeypatch.setattr("sentence_transformers.SentenceTransformer.__init__", mock_st_init)
    monkeypatch.setattr("sentence_transformers.SentenceTransformer.encode", mock_encode)


@pytest.fixture
def patch_chromadb(monkeypatch):
    """Patch ChromaDB to use in-memory mock."""
    def mock_from_documents(documents, embedding, collection_name, **kwargs):
        mock_vs = MockVectorStore()

        # Extract content from Document objects
        docs_content = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Generate embeddings
        embeddings = [embedding.embed_query(doc) for doc in docs_content]

        mock_vs.add_documents(docs_content, embeddings, metadatas)

        return mock_vs

    monkeypatch.setattr("langchain_community.vectorstores.Chroma.from_documents",
                       mock_from_documents)


@pytest.fixture
def all_patches(patch_ollama_llm, patch_sentence_transformer, patch_chromadb):
    """Apply all patches at once."""
    pass


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def assert_dict_contains(actual: Dict, expected_keys: List[str]):
    """Assert that dictionary contains all expected keys."""
    for key in expected_keys:
        assert key in actual, f"Key '{key}' not found in dictionary"


def assert_valid_metrics(metrics: Dict[str, float]):
    """Assert that metrics dictionary has valid values."""
    assert_dict_contains(metrics, ["exact_match", "partial_match", "overall_score"])

    for key, value in metrics.items():
        if isinstance(value, (int, float)) and key != "extracted_answer":
            assert 0.0 <= value <= 1.0, f"Metric '{key}' out of range: {value}"


def assert_valid_response(response: Dict[str, Any]):
    """Assert that LLM response has valid structure."""
    assert_dict_contains(response, ["response", "latency", "model", "success"])
    assert isinstance(response["response"], str)
    assert response["latency"] >= 0
    assert response["success"] in [True, False]
