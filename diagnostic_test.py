"""
Diagnostic script to test Experiment 1 behavior
"""
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from llm_interface import create_llm_interface
from evaluator import create_evaluator

# Load dataset
dataset_path = Path(__file__).parent / 'data' / 'synthetic' / 'needle_haystack' / 'dataset.json'
with open(dataset_path, 'r') as f:
    dataset = json.load(f)

# Initialize interfaces
llm = create_llm_interface()
evaluator = create_evaluator(use_embeddings=True)

# Test one document from each position
positions_to_test = ['start', 'middle', 'end']

for position in positions_to_test:
    print(f"\n{'='*60}")
    print(f"Testing position: {position}")
    print(f"{'='*60}")

    # Get first document for this position
    doc = next(d for d in dataset if d['position'] == position)

    print(f"\nDocument preview (first 200 chars):")
    print(doc['document'][:200] + "...")
    print(f"\nExpected secret: {doc['secret_value']}")
    print(f"\nFact: {doc['fact']}")

    # Query the LLM
    query = "What is the secret password mentioned in the documents?"

    print(f"\n--- Querying LLM ---")
    llm_result = llm.query(prompt=query, context=doc['document'])

    print(f"LLM Response: {llm_result['response']}")
    print(f"Latency: {llm_result['latency']:.2f}s")

    # Evaluate
    print(f"\n--- Evaluating Response ---")
    eval_result = evaluator.evaluate_needle_haystack(
        response=llm_result['response'],
        secret_value=doc['secret_value'],
        position=position
    )

    print(f"Exact match: {eval_result['exact_match']:.3f}")
    print(f"Partial match: {eval_result['partial_match']:.3f}")
    print(f"Keyword match: {eval_result.get('keyword_match', 0):.3f}")
    print(f"Semantic similarity: {eval_result.get('semantic_similarity', 0):.3f}")
    print(f"Overall score: {eval_result['overall_score']:.3f}")
    print(f"Correct: {eval_result['correct']}")
    print(f"Extracted answer: {eval_result.get('extracted_answer', 'N/A')}")

print(f"\n{'='*60}")
print("Diagnostic complete")
print(f"{'='*60}")
