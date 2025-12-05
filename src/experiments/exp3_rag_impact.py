"""
Experiment 3: RAG Impact

This experiment compares Retrieval-Augmented Generation (RAG)
with full-context approaches to demonstrate RAG's effectiveness.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from typing import List, Dict, Any
import json
import numpy as np
from tqdm import tqdm

from config import (
    EXP3_CONFIG,
    EXP3_RESULTS_DIR,
    HEBREW_CORPUS_DIR,
)
from data_generator import DataGenerator
from llm_interface import create_llm_interface, create_rag_system
from evaluator import create_evaluator
from utils.visualization import plot_rag_comparison


class RAGImpactExperiment:
    """Experiment to measure RAG effectiveness."""

    def __init__(self):
        """Initialize experiment."""
        self.config = EXP3_CONFIG
        self.results_dir = EXP3_RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.data_generator = DataGenerator()
        self.llm = create_llm_interface()
        self.evaluator = create_evaluator(use_embeddings=True)

        # RAG system will be initialized after loading documents
        self.rag_system = None

        print("="*60)
        print("Experiment 3: RAG Impact")
        print("="*60)

    def generate_data(self) -> List[Dict[str, str]]:
        """
        Generate Hebrew corpus.

        Returns:
            List of document dictionaries
        """
        print("\n[1/4] Generating Hebrew corpus...")

        corpus = self.data_generator.generate_hebrew_corpus(
            num_docs=self.config["num_documents"],
            save_dir=HEBREW_CORPUS_DIR,
        )

        print(f"  ✓ Generated {len(corpus)} Hebrew documents")

        # Count medical documents
        med_docs = [d for d in corpus if d.get("topic") == "medicine"]
        print(f"  ✓ {len(med_docs)} medical documents with drug information")

        return corpus

    def initialize_rag(self, corpus: List[Dict[str, str]]):
        """
        Initialize RAG system with corpus.

        Args:
            corpus: List of documents
        """
        print("\n[2/4] Initializing RAG system...")

        # Create RAG system
        self.rag_system = create_rag_system(
            chunk_size=self.config["chunk_size"],
            chunk_overlap=self.config["chunk_overlap"],
        )

        # Extract document texts
        documents = [doc["content"] for doc in corpus]

        # Add metadata
        metadatas = [
            {
                "doc_id": doc["doc_id"],
                "topic": doc.get("topic", "unknown"),
            }
            for doc in corpus
        ]

        # Add documents to RAG system
        self.rag_system.add_documents(documents, metadatas)

        print(f"  ✓ RAG system initialized with {len(documents)} documents")

    def run_full_context_trial(self, corpus: List[Dict[str, str]],
                               query: str) -> Dict[str, Any]:
        """
        Run trial with full context (all documents).

        Args:
            corpus: Complete corpus
            query: Query to answer

        Returns:
            Trial results
        """
        # Concatenate all documents
        all_documents = "\n\n---\n\n".join([doc["content"] for doc in corpus])

        # Count tokens
        tokens_used = self.llm.count_tokens(all_documents)

        # Query LLM
        llm_result = self.llm.query(
            prompt=query,
            context=all_documents
        )

        return {
            "method": "full_context",
            "response": llm_result["response"],
            "latency": llm_result["latency"],
            "tokens_used": tokens_used,
            "num_docs_used": len(corpus),
            "success": llm_result["success"],
        }

    def run_rag_trial(self, query: str) -> Dict[str, Any]:
        """
        Run trial with RAG.

        Args:
            query: Query to answer

        Returns:
            Trial results
        """
        # Query with RAG
        rag_result = self.rag_system.query_with_rag(
            query=query,
            top_k=self.config["top_k_retrieval"]
        )

        # Calculate tokens used (only retrieved chunks)
        retrieved_text = "\n\n".join([
            doc["content"] for doc in rag_result["retrieved_docs"]
        ])
        tokens_used = self.llm.count_tokens(retrieved_text)

        return {
            "method": "rag",
            "response": rag_result["response"],
            "latency": rag_result["total_time"],
            "retrieve_time": rag_result["retrieve_time"],
            "tokens_used": tokens_used,
            "num_docs_used": rag_result["num_docs_retrieved"],
            "retrieved_docs": rag_result["retrieved_docs"],
            "success": rag_result["success"],
        }

    def run_experiment(self, corpus: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Run the complete experiment.

        Args:
            corpus: Document corpus

        Returns:
            Experiment results
        """
        print("\n[3/4] Running experiment trials...")

        # Find a medical document to query about
        medical_docs = [d for d in corpus if d.get("drug_name")]

        if not medical_docs:
            raise ValueError("No medical documents with drug information found")

        # Select first medical document
        target_doc = medical_docs[0]
        drug_name = target_doc["drug_name"]
        expected_side_effects = target_doc["side_effects"]

        # Create query
        query_template = self.config["question_templates"]["medicine"]
        query = query_template.format(drug_name=drug_name)

        print(f"\n  Query: {query}")
        print(f"  Expected answer includes: {', '.join(expected_side_effects)}")

        # Expected answer (for evaluation)
        expected_answer = ", ".join(expected_side_effects)

        results = {}

        # Run full context trial
        print("\n  Running full context trial...")
        full_result = self.run_full_context_trial(corpus, query)

        # Evaluate
        full_eval = self.evaluator.evaluate_rag_response(
            response=full_result["response"],
            expected_answer=expected_answer,
            latency=full_result["latency"],
            method="full_context",
        )

        results["full_context"] = {
            **full_result,
            **full_eval,
        }

        print(f"    Accuracy: {full_eval['overall_score']:.3f}, "
              f"Latency: {full_result['latency']:.2f}s, "
              f"Tokens: {full_result['tokens_used']}")

        # Run RAG trial
        print("\n  Running RAG trial...")
        rag_result = self.run_rag_trial(query)

        # Evaluate
        rag_eval = self.evaluator.evaluate_rag_response(
            response=rag_result["response"],
            expected_answer=expected_answer,
            latency=rag_result["latency"],
            method="rag",
        )

        results["rag"] = {
            **rag_result,
            **rag_eval,
        }

        print(f"    Accuracy: {rag_eval['overall_score']:.3f}, "
              f"Latency: {rag_result['latency']:.2f}s, "
              f"Tokens: {rag_result['tokens_used']}")

        # Add query and expected answer for reference
        results["query"] = query
        results["expected_answer"] = expected_answer
        results["drug_name"] = drug_name

        return results

    def visualize_results(self, results: Dict[str, Any]):
        """
        Create visualizations.

        Args:
            results: Experiment results
        """
        print("\n[4/4] Creating visualizations...")

        # Prepare data for plotting
        full_context_data = {
            "accuracy": results["full_context"]["overall_score"],
            "latency": results["full_context"]["latency"],
        }

        rag_data = {
            "accuracy": results["rag"]["overall_score"],
            "latency": results["rag"]["latency"],
        }

        # Create plot
        plot_path = self.results_dir / "rag_comparison.png"

        plot_rag_comparison(
            full_context_results=full_context_data,
            rag_results=rag_data,
            save_path=plot_path,
            title="Experiment 3: RAG vs Full Context Comparison"
        )

        print(f"  ✓ Saved plot to {plot_path}")

    def save_results(self, results: Dict[str, Any]):
        """
        Save results to JSON.

        Args:
            results: Experiment results
        """
        # Prepare summary
        summary = {
            "experiment": "RAG Impact",
            "config": self.config,
            "query": results["query"],
            "expected_answer": results["expected_answer"],
            "comparison": {
                "full_context": {
                    "accuracy": results["full_context"]["overall_score"],
                    "latency": results["full_context"]["latency"],
                    "tokens_used": results["full_context"]["tokens_used"],
                    "num_docs_used": results["full_context"]["num_docs_used"],
                },
                "rag": {
                    "accuracy": results["rag"]["overall_score"],
                    "latency": results["rag"]["latency"],
                    "tokens_used": results["rag"]["tokens_used"],
                    "num_docs_used": results["rag"]["num_docs_used"],
                },
            },
            "detailed_results": results,
        }

        # Save to JSON
        save_path = self.results_dir / "results.json"

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Saved results to {save_path}")

    def run(self):
        """Run the complete experiment pipeline."""
        # Generate data
        corpus = self.generate_data()

        # Initialize RAG
        self.initialize_rag(corpus)

        # Run experiment
        results = self.run_experiment(corpus)

        # Visualize
        self.visualize_results(results)

        # Save results
        self.save_results(results)

        # Print summary
        print("\n" + "="*60)
        print("Experiment 3 Complete!")
        print("="*60)

        print("\nKey Findings:")
        print(f"  Full Context: "
              f"acc={results['full_context']['overall_score']:.3f}, "
              f"latency={results['full_context']['latency']:.2f}s, "
              f"tokens={results['full_context']['tokens_used']}")

        print(f"  RAG: "
              f"acc={results['rag']['overall_score']:.3f}, "
              f"latency={results['rag']['latency']:.2f}s, "
              f"tokens={results['rag']['tokens_used']}")

        # Calculate improvements
        acc_improvement = (
            (results['rag']['overall_score'] -
             results['full_context']['overall_score']) /
            results['full_context']['overall_score'] * 100
        )

        speedup = (
            results['full_context']['latency'] /
            results['rag']['latency']
        )

        print(f"\n  RAG Improvements:")
        print(f"    Accuracy: {acc_improvement:+.1f}%")
        print(f"    Speedup: {speedup:.2f}x faster")

        print("\n✓ All results saved to:", self.results_dir)


def main():
    """Main entry point."""
    experiment = RAGImpactExperiment()
    experiment.run()


if __name__ == "__main__":
    main()
