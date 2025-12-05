"""
LLM Interface for Ollama and LangChain integration.

This module provides a unified interface for querying LLMs,
managing embeddings, and implementing RAG functionality.
"""

import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path

# LangChain imports
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# Embeddings
from sentence_transformers import SentenceTransformer

# Import configuration
from config import (
    OLLAMA_BASE_URL,
    PRIMARY_MODEL,
    FALLBACK_MODEL,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    get_model_name,
)


class LLMInterface:
    """Interface for interacting with LLMs via Ollama."""

    def __init__(self, model_name: Optional[str] = None,
                 temperature: float = LLM_TEMPERATURE):
        """
        Initialize LLM interface.

        Args:
            model_name: Name of the model to use (auto-detect if None)
            temperature: Temperature for generation
        """
        self.model_name = model_name or get_model_name()
        self.temperature = temperature

        # Initialize Ollama LLM
        self.llm = OllamaLLM(
            model=self.model_name,
            base_url=OLLAMA_BASE_URL,
            temperature=self.temperature,
        )

        print(f"Initialized LLM: {self.model_name}")

    def query(self, prompt: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Query the LLM with optional context.

        Args:
            prompt: Question or instruction
            context: Optional context to include

        Returns:
            Dictionary with response, latency, and metadata
        """
        if context:
            full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}\n\nAnswer:"
        else:
            full_prompt = prompt

        start_time = time.time()

        try:
            response = self.llm.invoke(full_prompt)
            latency = time.time() - start_time

            return {
                "response": response,
                "latency": latency,
                "model": self.model_name,
                "success": True,
                "error": None,
            }

        except Exception as e:
            latency = time.time() - start_time

            return {
                "response": "",
                "latency": latency,
                "model": self.model_name,
                "success": False,
                "error": str(e),
            }

    def query_with_template(self, template: str, **kwargs) -> Dict[str, Any]:
        """
        Query using a prompt template.

        Args:
            template: Prompt template string
            **kwargs: Variables to fill in template

        Returns:
            Dictionary with response and metadata
        """
        prompt = PromptTemplate(
            template=template,
            input_variables=list(kwargs.keys())
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)

        start_time = time.time()

        try:
            response = chain.run(**kwargs)
            latency = time.time() - start_time

            return {
                "response": response,
                "latency": latency,
                "model": self.model_name,
                "success": True,
                "error": None,
            }

        except Exception as e:
            latency = time.time() - start_time

            return {
                "response": "",
                "latency": latency,
                "model": self.model_name,
                "success": False,
                "error": str(e),
            }

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        # Rough approximation: ~4 characters per token
        return len(text) // 4


class EmbeddingInterface:
    """Interface for generating and managing embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding interface.

        Args:
            model_name: Name of the sentence-transformers model
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        print(f"Initialized embeddings: {model_name}")

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        return self.model.encode(text, convert_to_numpy=True)

    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple documents.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        return self.model.encode(texts, convert_to_numpy=True)


class RAGSystem:
    """Retrieval-Augmented Generation system using ChromaDB."""

    def __init__(self,
                 llm_interface: LLMInterface,
                 embedding_interface: EmbeddingInterface,
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 collection_name: str = "documents"):
        """
        Initialize RAG system.

        Args:
            llm_interface: LLM interface instance
            embedding_interface: Embedding interface instance
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            collection_name: Name for ChromaDB collection
        """
        self.llm = llm_interface
        self.embedding = embedding_interface
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        # In-memory ChromaDB (no persistence)
        self.vectorstore = None
        self.collection_name = collection_name

        print(f"Initialized RAG system with chunk_size={chunk_size}")

    def add_documents(self, documents: List[str],
                     metadatas: Optional[List[Dict[str, Any]]] = None):
        """
        Add documents to the vector store.

        Args:
            documents: List of document texts
            metadatas: Optional metadata for each document
        """
        # Split documents into chunks
        all_chunks = []
        all_metadatas = []

        for i, doc in enumerate(documents):
            chunks = self.text_splitter.split_text(doc)

            for j, chunk in enumerate(chunks):
                all_chunks.append(chunk)

                # Add metadata
                metadata = {"doc_id": i, "chunk_id": j}
                if metadatas and i < len(metadatas):
                    metadata.update(metadatas[i])

                all_metadatas.append(metadata)

        # Generate embeddings
        embeddings = self.embedding.embed_documents(all_chunks)

        # Create Document objects for ChromaDB
        docs = [
            Document(page_content=chunk, metadata=meta)
            for chunk, meta in zip(all_chunks, all_metadatas)
        ]

        # Create custom embedding function wrapper
        class EmbeddingFunction:
            def __init__(self, embedding_interface):
                self.embedding_interface = embedding_interface

            def embed_documents(self, texts):
                return self.embedding_interface.embed_documents(texts).tolist()

            def embed_query(self, text):
                return self.embedding_interface.embed_text(text).tolist()

        embedding_function = EmbeddingFunction(self.embedding)

        # Initialize ChromaDB with documents
        self.vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding_function,
            collection_name=self.collection_name,
        )

        print(f"Added {len(all_chunks)} chunks from {len(documents)} documents")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Query text
            top_k: Number of documents to retrieve

        Returns:
            List of retrieved document dictionaries
        """
        if self.vectorstore is None:
            raise ValueError("No documents in vector store. Call add_documents first.")

        # Perform similarity search
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
        """
        Query using RAG (retrieve then generate).

        Args:
            query: Question to answer
            top_k: Number of documents to retrieve

        Returns:
            Dictionary with response and retrieval information
        """
        # Retrieve relevant documents
        start_retrieve = time.time()
        retrieved_docs = self.retrieve(query, top_k=top_k)
        retrieve_time = time.time() - start_retrieve

        # Concatenate retrieved content as context
        context = "\n\n".join([doc["content"] for doc in retrieved_docs])

        # Query LLM with context
        llm_result = self.llm.query(prompt=query, context=context)

        # Combine results
        return {
            "response": llm_result["response"],
            "latency": llm_result["latency"],
            "retrieve_time": retrieve_time,
            "total_time": retrieve_time + llm_result["latency"],
            "retrieved_docs": retrieved_docs,
            "num_docs_retrieved": len(retrieved_docs),
            "success": llm_result["success"],
            "error": llm_result["error"],
        }


def create_llm_interface() -> LLMInterface:
    """Factory function to create LLM interface."""
    return LLMInterface()


def create_rag_system(chunk_size: int = 500,
                      chunk_overlap: int = 50) -> RAGSystem:
    """
    Factory function to create RAG system.

    Args:
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks

    Returns:
        Initialized RAG system
    """
    llm = create_llm_interface()
    embedding = EmbeddingInterface()

    return RAGSystem(
        llm_interface=llm,
        embedding_interface=embedding,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


if __name__ == "__main__":
    # Test the interface
    print("Testing LLM Interface...")

    llm = create_llm_interface()

    # Simple query
    result = llm.query("What is 2+2?")
    print(f"Response: {result['response']}")
    print(f"Latency: {result['latency']:.2f}s")

    # Test RAG
    print("\nTesting RAG System...")

    rag = create_rag_system()

    # Add test documents
    docs = [
        "The capital of France is Paris. It is known for the Eiffel Tower.",
        "Python is a programming language. It is widely used for data science.",
        "The Earth orbits around the Sun. It takes approximately 365 days.",
    ]

    rag.add_documents(docs)

    # Query
    result = rag.query_with_rag("What is the capital of France?", top_k=2)
    print(f"Response: {result['response']}")
    print(f"Total time: {result['total_time']:.2f}s")
