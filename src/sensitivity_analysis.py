"""
Sensitivity Analysis for RAG Parameters

This script performs a sensitivity analysis on the RAG experiment by varying:
- chunk_size: 250, 500, 750
- top_k: 1, 3, 5

The analysis helps understand how these parameters affect retrieval quality.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import itertools

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import EXP3_RESULTS_DIR
from data_generator import DataGenerator
from evaluator import create_evaluator
from llm_interface import LLMInterface, RAGSystem, EmbeddingInterface


def run_sensitivity_analysis():
    """
    Run sensitivity analysis by varying chunk_size and top_k parameters.
    """
    print("=" * 60)
    print("RAG Parameter Sensitivity Analysis")
    print("=" * 60)

    # Parameter ranges
    chunk_sizes = [250, 500, 750]
    top_k_values = [1, 3, 5]

    # Initialize components
    print("\n[1/4] Initializing components...")
    llm = LLMInterface()
    embedding = EmbeddingInterface()
    data_gen = DataGenerator()
    evaluator = create_evaluator(use_embeddings=True)

    # Generate test corpus (small for quick testing)
    print("\n[2/4] Generating test corpus...")
    num_docs = 10
    facts = [
        "The capital of France is Paris",
        "Python was created by Guido van Rossum",
        "The speed of light is 299,792,458 meters per second",
        "DNA stands for deoxyribonucleic acid",
        "Mount Everest is the highest mountain in the world",
        "The Great Wall of China is over 13,000 miles long",
        "Shakespeare wrote 37 plays",
        "The human body has 206 bones",
        "Water boils at 100 degrees Celsius",
        "The Moon orbits Earth every 27.3 days"
    ]

    corpus = []
    for i in range(num_docs):
        # Generate filler text
        filler = data_gen.generate_filler_text(num_words=200, language="en")
        # Embed a fact in the middle
        doc = data_gen.embed_fact_in_text(filler, facts[i], position="middle")
        corpus.append(doc)

    print(f"  ✓ Generated {len(corpus)} documents")

    # Test questions
    questions = [
        "What is the capital of France?",
        "Who created Python?",
        "What is the speed of light?",
    ]

    # Run experiments
    print("\n[3/4] Running sensitivity analysis...")
    results = []
    total_runs = len(chunk_sizes) * len(top_k_values) * len(questions)
    current_run = 0

    for chunk_size, top_k in itertools.product(chunk_sizes, top_k_values):
        print(f"\n  Testing chunk_size={chunk_size}, top_k={top_k}")

        # Create RAG system with current parameters
        rag = RAGSystem(
            llm_interface=llm,
            embedding_interface=embedding,
            chunk_size=chunk_size,
            chunk_overlap=50
        )

        # Add documents to RAG
        rag.add_documents(corpus)

        # Run questions
        for question in questions:
            current_run += 1
            print(f"    [{current_run}/{total_runs}] Running question...")

            # Query RAG
            try:
                rag_result = rag.query_with_rag(question, top_k=top_k)
                response = rag_result["response"]

                # Evaluate (simple length and presence checks)
                result = {
                    "chunk_size": chunk_size,
                    "top_k": top_k,
                    "question": question,
                    "response": response,
                    "response_length": len(response),
                    "has_content": len(response.strip()) > 10
                }

                results.append(result)
            except Exception as e:
                print(f"      Error: {e}")
                results.append({
                    "chunk_size": chunk_size,
                    "top_k": top_k,
                    "question": question,
                    "error": str(e)
                })

    # Aggregate results
    print("\n[4/4] Aggregating results...")
    aggregated = {}

    for chunk_size, top_k in itertools.product(chunk_sizes, top_k_values):
        key = f"chunk{chunk_size}_k{top_k}"

        # Filter results for this configuration
        config_results = [
            r for r in results
            if r.get("chunk_size") == chunk_size and r.get("top_k") == top_k
        ]

        # Calculate metrics
        successful = [r for r in config_results if "error" not in r]

        aggregated[key] = {
            "chunk_size": chunk_size,
            "top_k": top_k,
            "num_trials": len(config_results),
            "num_successful": len(successful),
            "avg_response_length": (
                sum(r["response_length"] for r in successful) / len(successful)
                if successful else 0
            ),
            "success_rate": len(successful) / len(config_results) if config_results else 0
        }

    # Save results
    output_file = EXP3_RESULTS_DIR / "sensitivity_analysis.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "parameters": {
                "chunk_sizes": chunk_sizes,
                "top_k_values": top_k_values,
                "num_documents": num_docs,
                "num_questions": len(questions)
            },
            "detailed_results": results,
            "aggregated_results": aggregated
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results saved to {output_file}")

    # Print summary
    print("\nSummary:")
    print("-" * 60)
    for key, agg in aggregated.items():
        print(f"  {key}: avg_length={agg['avg_response_length']:.1f}, "
              f"success={agg['success_rate']:.1%}")

    return aggregated


if __name__ == "__main__":
    run_sensitivity_analysis()
