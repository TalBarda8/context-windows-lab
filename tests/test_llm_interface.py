"""
Tests for llm_interface.py - LLM, embeddings, and RAG interfaces.

Tests cover:
- LLMInterface with mocked Ollama calls
- EmbeddingInterface with mocked embeddings
- RAGSystem with mocked vector store
- Query execution and response handling
- Token counting
- Document retrieval
- End-to-end RAG pipeline

All tests use mocks - NO external dependencies (Ollama, sentence-transformers, ChromaDB).
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_interface import (
    LLMInterface,
    EmbeddingInterface,
    RAGSystem,
    create_llm_interface,
    create_rag_system,
)


class TestLLMInterfaceInit:
    """Test LLMInterface initialization."""

    def test_init_with_default_model(self, patch_ollama_llm):
        """Test initialization with default model."""
        llm = LLMInterface()

        assert llm.model_name is not None
        assert llm.temperature == 0.0
        assert llm.llm is not None

    def test_init_with_custom_model(self, patch_ollama_llm):
        """Test initialization with custom model."""
        llm = LLMInterface(model_name="llama3.2", temperature=0.5)

        assert llm.model_name == "llama3.2"
        assert llm.temperature == 0.5

    def test_init_with_custom_temperature(self, patch_ollama_llm):
        """Test initialization with custom temperature."""
        llm = LLMInterface(temperature=0.7)

        assert llm.temperature == 0.7


class TestLLMInterfaceQuery:
    """Test LLMInterface query method."""

    def test_simple_query(self, patch_ollama_llm):
        """Test simple query without context."""
        llm = LLMInterface()

        result = llm.query("What is 2+2?")

        assert "response" in result
        assert "latency" in result
        assert "model" in result
        assert "success" in result
        assert "error" in result

        assert result["success"] is True
        assert result["error"] is None
        assert isinstance(result["response"], str)
        assert result["latency"] >= 0

    def test_query_with_context(self, patch_ollama_llm):
        """Test query with context."""
        llm = LLMInterface()

        context = "The sky is blue because of Rayleigh scattering."
        prompt = "Why is the sky blue?"

        result = llm.query(prompt, context=context)

        assert result["success"] is True
        assert isinstance(result["response"], str)

    def test_query_response_structure(self, patch_ollama_llm):
        """Test that query response has correct structure."""
        llm = LLMInterface()

        result = llm.query("Test query")

        # Validate all required fields
        assert_valid_response(result)

    def test_query_latency_measurement(self, patch_ollama_llm):
        """Test that latency is measured."""
        llm = LLMInterface()

        result = llm.query("Test query")

        # Latency should be non-negative
        assert result["latency"] >= 0
        assert isinstance(result["latency"], (int, float))

    def test_query_model_name_in_response(self, patch_ollama_llm):
        """Test that model name is included in response."""
        llm = LLMInterface(model_name="test-model")

        result = llm.query("Test")

        assert result["model"] == "test-model"


class TestLLMInterfaceQueryWithTemplate:
    """Test LLMInterface query_with_template method."""

    def test_query_with_template(self, patch_ollama_llm):
        """Test templated query."""
        llm = LLMInterface()

        template = "What is the capital of {country}?"
        result = llm.query_with_template(template, country="France")

        assert result["success"] is True
        assert isinstance(result["response"], str)

    def test_template_with_multiple_variables(self, patch_ollama_llm):
        """Test template with multiple variables."""
        llm = LLMInterface()

        template = "The {item} costs ${price} in {location}."
        result = llm.query_with_template(
            template,
            item="book",
            price="25",
            location="Paris"
        )

        assert result["success"] is True

    def test_template_response_structure(self, patch_ollama_llm):
        """Test template response has correct structure."""
        llm = LLMInterface()

        result = llm.query_with_template("Hello {name}!", name="World")

        assert_valid_response(result)


class TestLLMInterfaceTokenCounting:
    """Test LLMInterface count_tokens method."""

    def test_count_tokens_simple_text(self, patch_ollama_llm):
        """Test token counting for simple text."""
        llm = LLMInterface()

        text = "This is a test"
        token_count = llm.count_tokens(text)

        # Rough estimate: ~4 chars per token
        expected = len(text) // 4
        assert token_count == expected

    def test_count_tokens_long_text(self, patch_ollama_llm):
        """Test token counting for longer text."""
        llm = LLMInterface()

        text = "A" * 1000  # 1000 characters
        token_count = llm.count_tokens(text)

        assert token_count == 250  # 1000 / 4

    def test_count_tokens_empty_string(self, patch_ollama_llm):
        """Test token counting for empty string."""
        llm = LLMInterface()

        token_count = llm.count_tokens("")
        assert token_count == 0


class TestEmbeddingInterfaceInit:
    """Test EmbeddingInterface initialization."""

    def test_init_default_model(self, patch_sentence_transformer):
        """Test initialization with default model."""
        embedding = EmbeddingInterface()

        assert embedding.model_name == "all-MiniLM-L6-v2"
        assert embedding.model is not None

    def test_init_custom_model(self, patch_sentence_transformer):
        """Test initialization with custom model."""
        embedding = EmbeddingInterface(model_name="custom-model")

        assert embedding.model_name == "custom-model"


class TestEmbeddingInterfaceEmbedText:
    """Test EmbeddingInterface embed_text method."""

    def test_embed_single_text(self, patch_sentence_transformer):
        """Test embedding single text."""
        embedding = EmbeddingInterface()

        text = "This is a test sentence."
        emb = embedding.embed_text(text)

        assert isinstance(emb, np.ndarray)
        assert emb.shape == (384,)  # Default embedding dimension
        assert emb.dtype == np.float32

    def test_embed_different_texts(self, patch_sentence_transformer):
        """Test that different texts produce different embeddings."""
        embedding = EmbeddingInterface()

        emb1 = embedding.embed_text("hello")
        emb2 = embedding.embed_text("goodbye")

        # Embeddings should be different
        assert not np.array_equal(emb1, emb2)

    def test_embed_reproducibility(self, patch_sentence_transformer):
        """Test that same text produces same embedding."""
        embedding = EmbeddingInterface()

        text = "reproducible text"
        emb1 = embedding.embed_text(text)
        emb2 = embedding.embed_text(text)

        # Should be identical (deterministic hashing in mock)
        np.testing.assert_array_equal(emb1, emb2)

    def test_embed_empty_string(self, patch_sentence_transformer):
        """Test embedding empty string."""
        embedding = EmbeddingInterface()

        emb = embedding.embed_text("")

        assert isinstance(emb, np.ndarray)
        assert emb.shape == (384,)


class TestEmbeddingInterfaceEmbedDocuments:
    """Test EmbeddingInterface embed_documents method."""

    def test_embed_multiple_documents(self, patch_sentence_transformer):
        """Test embedding multiple documents."""
        embedding = EmbeddingInterface()

        texts = ["First document", "Second document", "Third document"]
        embeddings = embedding.embed_documents(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 384)
        assert embeddings.dtype == np.float32

    def test_embed_documents_different(self, patch_sentence_transformer):
        """Test that different documents produce different embeddings."""
        embedding = EmbeddingInterface()

        texts = ["doc1", "doc2", "doc3"]
        embeddings = embedding.embed_documents(texts)

        # All should be different
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                assert not np.array_equal(embeddings[i], embeddings[j])

    def test_embed_single_document_in_list(self, patch_sentence_transformer):
        """Test embedding single document in a list."""
        embedding = EmbeddingInterface()

        embeddings = embedding.embed_documents(["single document"])

        assert embeddings.shape == (1, 384)


class TestRAGSystemInit:
    """Test RAGSystem initialization."""

    def test_init_with_interfaces(self, mock_llm, mock_embedding):
        """Test initialization with LLM and embedding interfaces."""
        rag = RAGSystem(
            llm_interface=mock_llm,
            embedding_interface=mock_embedding
        )

        assert rag.llm is not None
        assert rag.embedding is not None
        assert rag.chunk_size == 500
        assert rag.chunk_overlap == 50

    def test_init_custom_chunk_size(self, mock_llm, mock_embedding):
        """Test initialization with custom chunk size."""
        rag = RAGSystem(
            llm_interface=mock_llm,
            embedding_interface=mock_embedding,
            chunk_size=1000,
            chunk_overlap=100
        )

        assert rag.chunk_size == 1000
        assert rag.chunk_overlap == 100

    def test_init_text_splitter(self, mock_llm, mock_embedding):
        """Test that text splitter is initialized."""
        rag = RAGSystem(
            llm_interface=mock_llm,
            embedding_interface=mock_embedding
        )

        assert rag.text_splitter is not None


class TestRAGSystemAddDocuments:
    """Test RAGSystem add_documents method."""

    def test_add_documents(self, mock_rag, sample_documents):
        """Test adding documents to RAG system."""
        # Add documents
        mock_rag.add_documents(sample_documents)

        # Vector store should have documents
        assert len(mock_rag.vectorstore.documents) > 0

    def test_add_documents_with_metadata(self, mock_rag):
        """Test adding documents with metadata."""
        documents = ["Doc 1", "Doc 2"]
        metadatas = [
            {"source": "file1.txt", "topic": "tech"},
            {"source": "file2.txt", "topic": "law"}
        ]

        mock_rag.add_documents(documents, metadatas=metadatas)

        # Should have documents with metadata
        assert len(mock_rag.vectorstore.documents) == 2
        assert len(mock_rag.vectorstore.metadatas) == 2

    def test_add_empty_documents(self, mock_rag):
        """Test adding empty document list."""
        mock_rag.add_documents([])

        # Should handle gracefully
        assert len(mock_rag.vectorstore.documents) == 0


class TestRAGSystemRetrieve:
    """Test RAGSystem retrieve method."""

    def test_retrieve_documents(self, mock_rag, sample_documents):
        """Test retrieving relevant documents."""
        # Add documents first
        mock_rag.add_documents(sample_documents)

        # Retrieve
        query = "What is the capital of France?"
        results = mock_rag.retrieve(query, top_k=3)

        assert isinstance(results, list)
        assert len(results) <= 3

        # Each result should have required fields
        for result in results:
            assert "content" in result
            assert "metadata" in result
            assert "similarity_score" in result

    def test_retrieve_top_k(self, mock_rag, sample_documents):
        """Test that top_k limits results."""
        mock_rag.add_documents(sample_documents)

        results = mock_rag.retrieve("test query", top_k=2)

        assert len(results) <= 2

    def test_retrieve_similarity_scores(self, mock_rag, sample_documents):
        """Test that similarity scores are included."""
        mock_rag.add_documents(sample_documents)

        results = mock_rag.retrieve("test query", top_k=3)

        for result in results:
            score = result["similarity_score"]
            assert isinstance(score, (int, float))
            assert 0.0 <= score <= 1.0

    def test_retrieve_without_documents(self, mock_llm, mock_embedding):
        """Test retrieval without adding documents first."""
        rag = RAGSystem(mock_llm, mock_embedding)

        # Should raise error
        with pytest.raises(ValueError, match="No documents in vector store"):
            rag.retrieve("test query")


class TestRAGSystemQueryWithRAG:
    """Test RAGSystem query_with_rag method."""

    def test_query_with_rag(self, mock_rag, sample_documents):
        """Test full RAG query pipeline."""
        # Add documents
        mock_rag.add_documents(sample_documents)

        # Query
        query = "What is the capital of France?"
        result = mock_rag.query_with_rag(query, top_k=3)

        # Check response structure
        assert "response" in result
        assert "latency" in result
        assert "retrieve_time" in result
        assert "total_time" in result
        assert "retrieved_docs" in result
        assert "num_docs_retrieved" in result
        assert "success" in result
        assert "error" in result

        assert result["success"] is True

    def test_rag_retrieved_docs_count(self, mock_rag, sample_documents):
        """Test that correct number of documents are retrieved."""
        mock_rag.add_documents(sample_documents)

        result = mock_rag.query_with_rag("test query", top_k=2)

        assert result["num_docs_retrieved"] <= 2
        assert len(result["retrieved_docs"]) <= 2

    def test_rag_timing(self, mock_rag, sample_documents):
        """Test that timing is measured."""
        mock_rag.add_documents(sample_documents)

        result = mock_rag.query_with_rag("test query")

        assert result["retrieve_time"] >= 0
        assert result["latency"] >= 0
        assert result["total_time"] >= 0

        # Total should be sum of retrieve and generation
        expected_total = result["retrieve_time"] + result["latency"]
        assert abs(result["total_time"] - expected_total) < 0.1

    def test_rag_uses_context(self, mock_rag, sample_documents):
        """Test that RAG uses retrieved docs as context."""
        mock_rag.add_documents(sample_documents)

        # The mock LLM will return different responses based on context
        result = mock_rag.query_with_rag("What is the secret password?")

        # Response should be non-empty
        assert len(result["response"]) > 0


class TestFactoryFunctions:
    """Test factory functions for creating interfaces."""

    def test_create_llm_interface(self, patch_ollama_llm):
        """Test create_llm_interface factory."""
        llm = create_llm_interface()

        assert isinstance(llm, LLMInterface)
        assert llm.llm is not None

    def test_create_rag_system(self, all_patches):
        """Test create_rag_system factory."""
        rag = create_rag_system(chunk_size=1000, chunk_overlap=100)

        assert isinstance(rag, RAGSystem)
        assert rag.chunk_size == 1000
        assert rag.chunk_overlap == 100

    def test_create_rag_system_defaults(self, all_patches):
        """Test create_rag_system with defaults."""
        rag = create_rag_system()

        assert isinstance(rag, RAGSystem)
        assert rag.chunk_size == 500
        assert rag.chunk_overlap == 50


class TestLLMInterfaceIntegration:
    """Integration tests for LLMInterface."""

    def test_multiple_queries(self, patch_ollama_llm):
        """Test multiple sequential queries."""
        llm = LLMInterface()

        queries = [
            "What is 1+1?",
            "What is the capital of France?",
            "Explain gravity.",
        ]

        results = [llm.query(q) for q in queries]

        # All should succeed
        assert all(r["success"] for r in results)

        # All should have responses
        assert all(len(r["response"]) > 0 for r in results)

    def test_query_with_different_contexts(self, patch_ollama_llm):
        """Test queries with different contexts."""
        llm = LLMInterface()

        contexts = [
            "Context about Paris",
            "Context about London",
            "Context about Tokyo",
        ]

        for context in contexts:
            result = llm.query("What city?", context=context)
            assert result["success"]

    def test_token_counting_consistency(self, patch_ollama_llm):
        """Test that token counting is consistent."""
        llm = LLMInterface()

        text = "This is a test" * 10  # Repeated text

        count1 = llm.count_tokens(text)
        count2 = llm.count_tokens(text)

        assert count1 == count2


class TestEmbeddingInterfaceIntegration:
    """Integration tests for EmbeddingInterface."""

    def test_batch_embedding(self, patch_sentence_transformer):
        """Test embedding batch of documents."""
        embedding = EmbeddingInterface()

        documents = [f"Document {i}" for i in range(20)]
        embeddings = embedding.embed_documents(documents)

        assert embeddings.shape == (20, 384)

    def test_embedding_consistency(self, patch_sentence_transformer):
        """Test that embeddings are consistent across calls."""
        embedding = EmbeddingInterface()

        text = "consistent embedding test"

        emb1 = embedding.embed_text(text)
        emb2 = embedding.embed_text(text)

        np.testing.assert_array_equal(emb1, emb2)


class TestRAGSystemIntegration:
    """Integration tests for RAGSystem."""

    def test_full_rag_pipeline(self, mock_llm, mock_embedding):
        """Test complete RAG pipeline from documents to answer."""
        rag = RAGSystem(mock_llm, mock_embedding)

        # Add knowledge base
        documents = [
            "The capital of France is Paris. It is known for the Eiffel Tower.",
            "Python is a programming language. It is widely used for data science.",
            "The Earth orbits around the Sun. It takes approximately 365 days.",
        ]

        rag.add_documents(documents)

        # Query
        result = rag.query_with_rag("What is the capital of France?", top_k=2)

        # Should succeed
        assert result["success"]
        assert len(result["response"]) > 0

        # Should retrieve documents
        assert result["num_docs_retrieved"] > 0

    def test_rag_with_metadata(self, mock_llm, mock_embedding):
        """Test RAG with document metadata."""
        rag = RAGSystem(mock_llm, mock_embedding)

        documents = ["Doc 1 content", "Doc 2 content"]
        metadatas = [
            {"source": "file1.txt", "date": "2024-01-01"},
            {"source": "file2.txt", "date": "2024-01-02"},
        ]

        rag.add_documents(documents, metadatas=metadatas)

        result = rag.query_with_rag("test query")

        # Retrieved docs should have metadata
        for doc in result["retrieved_docs"]:
            assert "metadata" in doc

    def test_rag_multiple_queries(self, mock_llm, mock_embedding):
        """Test multiple queries on same RAG system."""
        rag = RAGSystem(mock_llm, mock_embedding)

        rag.add_documents([
            "Information about topic A",
            "Information about topic B",
            "Information about topic C",
        ])

        queries = [
            "Tell me about topic A",
            "Tell me about topic B",
            "Tell me about topic C",
        ]

        results = [rag.query_with_rag(q) for q in queries]

        # All should succeed
        assert all(r["success"] for r in results)


# Helper functions

def assert_valid_response(response: dict):
    """Assert that LLM response has valid structure."""
    required_keys = ["response", "latency", "model", "success", "error"]
    for key in required_keys:
        assert key in response, f"Missing key: {key}"

    assert isinstance(response["response"], str)
    assert isinstance(response["latency"], (int, float))
    assert response["latency"] >= 0
    assert isinstance(response["success"], bool)
