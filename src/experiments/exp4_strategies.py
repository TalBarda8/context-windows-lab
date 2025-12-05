"""
Experiment 4: Context Engineering Strategies

This experiment compares different context management strategies:
- SELECT: Use RAG to retrieve only relevant history
- COMPRESS: Automatically summarize history when it grows too large
- WRITE: External memory (scratchpad) for key facts
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from typing import List, Dict, Any, Optional
import json
import numpy as np
from tqdm import tqdm

from config import (
    EXP4_CONFIG,
    EXP4_RESULTS_DIR,
)
from llm_interface import create_llm_interface, create_rag_system
from evaluator import create_evaluator
from utils.visualization import plot_strategy_comparison


class ContextStrategy:
    """Base class for context management strategies."""

    def __init__(self, llm):
        """Initialize strategy."""
        self.llm = llm
        self.history = []

    def add_to_history(self, output: str, metadata: Dict[str, Any] = None):
        """Add item to history."""
        self.history.append({
            "output": output,
            "metadata": metadata or {},
        })

    def get_context(self, query: str) -> str:
        """Get context for query (to be implemented by subclasses)."""
        raise NotImplementedError

    def get_context_size(self) -> int:
        """Get current context size in tokens."""
        context = self.get_context("")
        return self.llm.count_tokens(context)


class SelectStrategy(ContextStrategy):
    """SELECT strategy: Use RAG to retrieve relevant history."""

    def __init__(self, llm, top_k: int = 5):
        """Initialize SELECT strategy."""
        super().__init__(llm)
        self.top_k = top_k
        self.rag_system = None

    def add_to_history(self, output: str, metadata: Dict[str, Any] = None):
        """Add to history and update RAG system."""
        super().add_to_history(output, metadata)

        # Update RAG system
        if self.rag_system is None:
            self.rag_system = create_rag_system(chunk_size=200, chunk_overlap=20)

        # Add to RAG
        all_outputs = [h["output"] for h in self.history]
        all_metadata = [h["metadata"] for h in self.history]

        # Re-initialize RAG with all history
        self.rag_system = create_rag_system(chunk_size=200, chunk_overlap=20)
        self.rag_system.add_documents(all_outputs, all_metadata)

    def get_context(self, query: str) -> str:
        """Retrieve relevant history using RAG."""
        if self.rag_system is None or not self.history:
            return ""

        try:
            # Retrieve relevant items
            retrieved = self.rag_system.retrieve(query, top_k=self.top_k)

            # Concatenate
            context = "\n\n".join([doc["content"] for doc in retrieved])

            return context
        except:
            # Fallback to most recent items
            recent = self.history[-self.top_k:]
            return "\n\n".join([h["output"] for h in recent])


class CompressStrategy(ContextStrategy):
    """COMPRESS strategy: Summarize history when it grows too large."""

    def __init__(self, llm, max_tokens: int = 2048):
        """Initialize COMPRESS strategy."""
        super().__init__(llm)
        self.max_tokens = max_tokens
        self.compressed_history = []

    def get_context(self, query: str = "") -> str:
        """Get context, compressing if necessary."""
        # Combine all history
        full_context = "\n\n".join([h["output"] for h in self.history])

        # Check size
        context_size = self.llm.count_tokens(full_context)

        if context_size <= self.max_tokens:
            return full_context

        # Need to compress
        print(f"    Compressing history ({context_size} tokens -> {self.max_tokens})")

        # Simple compression: keep only most recent items
        compressed = ""
        tokens_used = 0

        for item in reversed(self.history):
            item_text = item["output"]
            item_tokens = self.llm.count_tokens(item_text)

            if tokens_used + item_tokens <= self.max_tokens:
                compressed = item_text + "\n\n" + compressed
                tokens_used += item_tokens
            else:
                break

        return compressed.strip()


class WriteStrategy(ContextStrategy):
    """WRITE strategy: External memory (scratchpad) for key facts."""

    def __init__(self, llm, capacity: int = 20):
        """Initialize WRITE strategy."""
        super().__init__(llm)
        self.scratchpad = {}  # Key-value store
        self.capacity = capacity

    def extract_key_facts(self, text: str) -> Dict[str, str]:
        """
        Extract key facts from text using LLM.

        Args:
            text: Text to extract facts from

        Returns:
            Dictionary of key facts
        """
        # Simple extraction: look for patterns like "X is Y", "X: Y"
        facts = {}

        # Use LLM to extract facts
        extraction_prompt = f"""Extract key facts from the following text.
Return them as simple statements, one per line.

Text: {text}

Facts:"""

        try:
            result = self.llm.query(extraction_prompt)
            fact_lines = result["response"].strip().split("\n")

            for i, line in enumerate(fact_lines[:self.capacity]):
                if line.strip():
                    facts[f"fact_{len(self.scratchpad) + i}"] = line.strip()

        except Exception as e:
            print(f"    Warning: Could not extract facts: {e}")

        return facts

    def add_to_history(self, output: str, metadata: Dict[str, Any] = None):
        """Add to history and extract key facts to scratchpad."""
        super().add_to_history(output, metadata)

        # Extract and store key facts
        new_facts = self.extract_key_facts(output)

        # Add to scratchpad (with capacity limit)
        for key, value in new_facts.items():
            if len(self.scratchpad) < self.capacity:
                self.scratchpad[key] = value

    def get_context(self, query: str) -> str:
        """Get context from scratchpad."""
        if not self.scratchpad:
            return ""

        # Return all facts from scratchpad
        facts_text = "\n".join([
            f"- {value}" for value in self.scratchpad.values()
        ])

        return f"Key facts:\n{facts_text}"


class ContextEngineeringExperiment:
    """Experiment to compare context engineering strategies."""

    def __init__(self):
        """Initialize experiment."""
        self.config = EXP4_CONFIG
        self.results_dir = EXP4_RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.llm = create_llm_interface()
        self.evaluator = create_evaluator(use_embeddings=True)

        print("="*60)
        print("Experiment 4: Context Engineering Strategies")
        print("="*60)

    def simulate_agent_actions(self) -> List[Dict[str, str]]:
        """
        Simulate a sequence of agent actions.

        Returns:
            List of action dictionaries
        """
        print("\n[1/3] Generating agent action sequence...")

        actions = []

        # Create a sequence of related tasks
        tasks = [
            {"id": 1, "description": "Initialize database connection",
             "output": "Database connection established successfully. Connection ID: DB_12345"},

            {"id": 2, "description": "Create user table",
             "output": "User table created with columns: id, name, email, created_at"},

            {"id": 3, "description": "Insert first user",
             "output": "User inserted: ID=1, Name=Alice, Email=alice@example.com"},

            {"id": 4, "description": "Insert second user",
             "output": "User inserted: ID=2, Name=Bob, Email=bob@example.com"},

            {"id": 5, "description": "Create products table",
             "output": "Products table created with columns: id, name, price, stock"},

            {"id": 6, "description": "Insert products",
             "output": "Products inserted: Laptop ($999), Mouse ($29), Keyboard ($79)"},

            {"id": 7, "description": "Create orders table",
             "output": "Orders table created with columns: id, user_id, product_id, quantity"},

            {"id": 8, "description": "Process order for Alice",
             "output": "Order created: User Alice ordered 1 Laptop for $999"},

            {"id": 9, "description": "Process order for Bob",
             "output": "Order created: User Bob ordered 2 Mice for $58"},

            {"id": 10, "description": "Generate sales report",
             "query": "What is the total revenue from all orders?",
             "expected_answer": "$1057",
             "output": "Sales report: Total revenue is $1057 from 2 orders"},
        ]

        actions = tasks[:self.config["num_actions"]]

        print(f"  ✓ Generated {len(actions)} sequential actions")

        return actions

    def run_strategy(self, strategy_name: str,
                    actions: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Run experiment with specified strategy.

        Args:
            strategy_name: Name of strategy ('select', 'compress', 'write')
            actions: List of actions to execute

        Returns:
            List of step results
        """
        # Initialize strategy
        if strategy_name == "select":
            strategy = SelectStrategy(self.llm, top_k=self.config["select_top_k"])
        elif strategy_name == "compress":
            strategy = CompressStrategy(self.llm,
                                       max_tokens=self.config["max_tokens_threshold"])
        elif strategy_name == "write":
            strategy = WriteStrategy(self.llm,
                                    capacity=self.config["scratchpad_capacity"])
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        results = []

        # Execute actions
        for step, action in enumerate(actions, start=1):
            # Add previous output to history (except first step)
            if step > 1:
                prev_output = actions[step - 2]["output"]
                strategy.add_to_history(prev_output, {"step": step - 1})

            # Get context for current step
            query = action.get("query", action.get("description", ""))
            context = strategy.get_context(query)
            context_size = self.llm.count_tokens(context)

            # If this action has a query, test accuracy
            if "query" in action:
                # Query with context
                llm_result = self.llm.query(prompt=query, context=context)

                # Evaluate
                eval_result = self.evaluator.evaluate_strategy_step(
                    response=llm_result["response"],
                    expected_output=action["expected_answer"],
                    strategy=strategy_name,
                    step=step,
                    context_size=context_size,
                )

                results.append({
                    "step": step,
                    "strategy": strategy_name,
                    "context_size": context_size,
                    "accuracy": eval_result["overall_score"],
                    "correct": eval_result["correct"],
                    "query": query,
                    "response": llm_result["response"],
                    "expected": action["expected_answer"],
                })

        return results

    def run_experiment(self, actions: List[Dict[str, str]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run experiment with all strategies.

        Args:
            actions: List of actions

        Returns:
            Results for each strategy
        """
        print("\n[2/3] Running experiment with all strategies...")

        results = {}

        for strategy_name in self.config["strategies"]:
            print(f"\n  Testing {strategy_name.upper()} strategy...")

            strategy_results = self.run_strategy(strategy_name, actions)

            results[strategy_name] = strategy_results

            if strategy_results:
                avg_accuracy = sum(r["accuracy"] for r in strategy_results) / len(strategy_results)
                print(f"    Average accuracy: {avg_accuracy:.3f}")

        return results

    def visualize_results(self, results: Dict[str, List[Dict[str, Any]]]):
        """
        Create visualizations.

        Args:
            results: Experiment results
        """
        print("\n[3/3] Creating visualizations...")

        # Create plot
        plot_path = self.results_dir / "strategy_comparison.png"

        plot_strategy_comparison(
            results=results,
            save_path=plot_path,
            title="Experiment 4: Context Management Strategies Comparison"
        )

        print(f"  ✓ Saved plot to {plot_path}")

    def save_results(self, results: Dict[str, List[Dict[str, Any]]]):
        """
        Save results to JSON.

        Args:
            results: Experiment results
        """
        # Prepare summary
        summary = {
            "experiment": "Context Engineering Strategies",
            "config": self.config,
            "summary": {
                strategy: {
                    "mean_accuracy": sum(r["accuracy"] for r in res) / len(res) if res else 0,
                    "correct_count": sum(1 for r in res if r["correct"]),
                    "total_steps": len(res),
                }
                for strategy, res in results.items()
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
        # Generate action sequence
        actions = self.simulate_agent_actions()

        # Run experiment
        results = self.run_experiment(actions)

        # Visualize
        self.visualize_results(results)

        # Save results
        self.save_results(results)

        # Print summary
        print("\n" + "="*60)
        print("Experiment 4 Complete!")
        print("="*60)

        print("\nKey Findings:")
        for strategy, res in results.items():
            if res:
                avg_acc = sum(r["accuracy"] for r in res) / len(res)
                correct = sum(1 for r in res if r["correct"])
                print(f"  {strategy.upper()}: "
                      f"avg_accuracy={avg_acc:.3f}, "
                      f"correct={correct}/{len(res)}")

        print("\n✓ All results saved to:", self.results_dir)


def main():
    """Main entry point."""
    experiment = ContextEngineeringExperiment()
    experiment.run()


if __name__ == "__main__":
    main()
