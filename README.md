# Context Windows Lab

**Lab Assignment 5 - Optional Assignment 1 for Targil 5**
**Course**: LLMs in Multi-Agent Environments
**Instructor**: Dr. Yoram Segal
**Date**: December 2025

## Overview

This lab investigates how context window characteristics and management strategies affect the accuracy and efficiency of Large Language Models (LLMs). Through four modular experiments, we explore:

1. **Lost in the Middle**: Position-based information retrieval accuracy
2. **Context Size Impact**: Performance degradation with growing context
3. **RAG Effectiveness**: Retrieval-Augmented Generation vs. full-context approaches
4. **Context Engineering**: Strategies for managing context in multi-turn conversations

## Project Structure

```
context-windows-lab/
├── docs/               # Documentation (RPD, results)
├── data/               # Synthetic and corpus data
├── src/                # Source code
│   ├── experiments/    # Four experiment modules
│   └── utils/          # Utilities (visualization, metrics)
├── notebooks/          # Jupyter analysis notebooks
├── results/            # Experiment outputs
└── scripts/            # Setup and run scripts
```

## Prerequisites

- **Python**: 3.10 or higher
- **Ollama**: Installed and running locally
  - Install from: https://ollama.ai
  - Pull required model: `ollama pull llama2` (or llama3.2, mistral)
- **Git**: For version control

## Setup Instructions

### 1. Clone Repository

```bash
git clone <repository-url>
cd context-windows-lab
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Ollama Installation

```bash
# Check Ollama is running
ollama list

# If llama3.2 is not listed, pull it
ollama pull llama3.2
```

### 5. Run Setup Script (Optional)

```bash
bash scripts/setup_environment.sh
```

## Running Experiments

### Run All Experiments

```bash
bash scripts/run_all_experiments.sh
```

### Run Individual Experiments

```bash
# Experiment 1: Needle in Haystack
python -m src.experiments.exp1_needle_haystack

# Experiment 2: Context Window Size Impact
python -m src.experiments.exp2_context_size

# Experiment 3: RAG Impact
python -m src.experiments.exp3_rag_impact

# Experiment 4: Context Engineering Strategies
python -m src.experiments.exp4_strategies
```

## Analyzing Results

Launch Jupyter notebooks for interactive analysis:

```bash
jupyter notebook notebooks/
```

Available notebooks:
- `exp1_analysis.ipynb` - Lost in the Middle analysis
- `exp2_analysis.ipynb` - Context size impact analysis
- `exp3_analysis.ipynb` - RAG comparison analysis
- `exp4_analysis.ipynb` - Strategy comparison analysis

## Results

All experiment outputs are saved to the `results/` directory:
- **JSON**: Raw data and metrics
- **PNG**: Visualizations (graphs, charts)
- **CSV**: Statistical summaries

Final analysis and conclusions are available in `docs/RESULTS.md`.

## Configuration

Experiment parameters can be adjusted in `src/config.py`:
- Model selection (llama3.2, mistral, etc.)
- Token limits and chunk sizes
- Iteration counts for statistical validation
- Temperature and other LLM parameters

## Technical Stack

- **LLM Framework**: Ollama, LangChain
- **Vector DB**: ChromaDB
- **Embeddings**: sentence-transformers
- **Analysis**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn

## Troubleshooting

### Ollama Connection Error
```bash
# Ensure Ollama is running
ollama serve
```

### ChromaDB Issues
- The project uses in-memory ChromaDB by default
- No external database setup required

### Hebrew Text Display
- Ensure your terminal/IDE supports UTF-8 encoding
- Install Hebrew fonts if visualization has rendering issues

## Documentation

- **RPD**: Research, Plan & Development document → `docs/RPD.md`
- **Results**: Final analysis and findings → `docs/RESULTS.md`
- **Assignment**: Original PDF → `context-windows-lab.pdf`

## License

All rights reserved - Dr. Yoram Segal
For academic use only.

## Contact

For questions or issues, please contact the course instructor.

---

**Last Updated**: 2025-12-05
**Version**: 1.0
