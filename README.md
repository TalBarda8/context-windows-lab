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

### Core Documentation

- **PRD (Product Requirements)**: Complete project specification → `docs/PRD.md`
- **Architecture**: System design, C4 diagrams, ADRs → `docs/ARCHITECTURE.md`
- **API Reference**: Complete API documentation with Building Block pattern → `docs/API.md`
- **Results**: Comprehensive analysis with cost/token breakdown → `docs/RESULTS.md`
- **Prompts**: LLM-assisted development log → `docs/PROMPTS.md`
- **Compliance**: Guideline verification report → `COMPLIANCE_VERIFICATION.md`

### Quick Links

| Document | Purpose |
|----------|---------|
| `README.md` | Project overview and setup |
| `HOW_TO_RUN.md` | Detailed execution instructions |
| `docs/PRD.md` | Requirements and objectives |
| `docs/ARCHITECTURE.md` | System design and decisions |
| `docs/API.md` | Public interface reference |
| `docs/RESULTS.md` | Experimental findings |
| `docs/PROMPTS.md` | Development process log |

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

Key configuration options:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `PRIMARY_MODEL` | `llama2` | LLM model to use |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `RANDOM_SEED` | `42` | Random seed for reproducibility |
| `LLM_TEMPERATURE` | `0.0` | Temperature (0=deterministic) |

See `.env.example` for all available options.

### Advanced Configuration

Edit `src/config.py` to modify:
- Experiment parameters (iterations, document counts)
- Context window limits
- Evaluation thresholds
- Visualization settings
- File paths

## Testing

### Unit Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests with coverage
pytest

# Generate HTML coverage report
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

### Code Quality

```bash
# Format code with Black
black src/ tests/

# Check style with flake8
flake8 src/ tests/

# Type checking (optional)
mypy src/
```

## Performance

### Benchmarks

| Experiment | LLM Calls | Runtime | Tokens |
|------------|-----------|---------|--------|
| Exp 1 | 30 | ~25s | 36,600 |
| Exp 2 | 25 | ~98s | 144,650 |
| Exp 3 | 2 | ~35s | 6,039 |
| Exp 4 | 30 | ~66s | 26,000 |
| **Total** | **87** | **~3m 44s** | **~215K** |

### Optimization Tips

1. **Use RAG for large corpora** (95% token reduction)
2. **Enable GPU acceleration** for embeddings
3. **Increase Ollama workers** in `ollama serve`
4. **Use llama-3** for 128K context windows
5. **Cache embeddings** for repeated queries

## Contribution Guidelines

This is an academic project for M.Sc. coursework. External contributions are not accepted. However, you may:

- Fork the repository for educational purposes
- Report issues via GitHub Issues
- Suggest improvements via Pull Requests (for review only)

### Code Standards

- Follow PEP 8 style guide
- Use type hints for all functions
- Write docstrings in Building Block pattern (Input/Output/Setup Data)
- Add tests for new functionality
- Update documentation for API changes

## Reproducibility

This project ensures full reproducibility:

1. **Fixed Random Seeds**: All experiments use `RANDOM_SEED=42`
2. **Deterministic LLM**: Temperature set to 0.0
3. **Pinned Dependencies**: Exact versions in `requirements.txt`
4. **Version Control**: All code committed to Git
5. **Documentation**: Complete setup instructions
6. **Configuration**: All parameters in `config.py` and `.env`

### Reproducing Results

```bash
# 1. Clone repository
git clone https://github.com/TalBarda8/context-windows-lab.git
cd context-windows-lab

# 2. Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Install and start Ollama
ollama pull llama2

# 4. Run all experiments
bash scripts/run_all_experiments.sh

# 5. Verify results match
diff results/exp1/results.json expected_results/exp1/results.json
```

## Security Considerations

### Data Privacy

- All processing is **local** (no external API calls)
- No data leaves your machine
- Ollama runs entirely offline
- No telemetry or analytics

### Safe Practices

- `.env` file excluded from Git (contains configuration, not secrets)
- No hardcoded credentials
- Input validation on all user-provided data
- Sanitized error messages (no sensitive data in logs)

### Potential Risks

- Large language models may generate unexpected content
- Hebrew corpus generation uses Faker (synthetic data only)
- Local resource consumption (CPU/RAM/disk)

## Extensions and Future Work

### Planned Enhancements

1. **Additional Models**: Test with GPT-4, Claude 3, Gemma
2. **Larger Contexts**: Use llama-3 (128K tokens)
3. **Multilingual**: Extend to French, Spanish, Arabic
4. **Real Datasets**: Replace synthetic data with actual corpora
5. **Interactive Demo**: Streamlit or Gradio web interface
6. **Parallelization**: Multiprocessing for faster experiments

### Research Directions

- Position bias with different model architectures
- Optimal chunk size for RAG (vs fixed 500 tokens)
- Hybrid strategies combining SELECT + COMPRESS
- Attention visualization for context analysis

## Acknowledgments

- **Instructor**: Dr. Yoram Segal
- **Course**: LLMs in Multi-Agent Environments
- **Institution**: [University Name]
- **Tools**: Ollama, LangChain, ChromaDB, sentence-transformers

## License

**All Rights Reserved** - Dr. Yoram Segal

This project is for academic use only. Redistribution, commercial use, or publication requires explicit permission from the copyright holder.

### Academic Use

Students may:
- Study the code for educational purposes
- Fork for personal learning
- Reference in academic work (with proper citation)

Students may NOT:
- Submit as their own work
- Redistribute without attribution
- Use for commercial purposes

### Citation

If referencing this work:

```bibtex
@software{barda2025contextwindows,
  author = {Barda, Tal},
  title = {Context Windows Lab: Investigating LLM Context Management},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/TalBarda8/context-windows-lab}
}
```

## Contact

**Student**: Tal Barda
**Email**: tal.barda@example.com
**Instructor**: Dr. Yoram Segal
**Course**: LLMs in Multi-Agent Environments

For questions or issues:
1. Check documentation in `docs/` directory
2. Review existing GitHub Issues
3. Open new issue with detailed description
4. Contact course instructor for academic matters

## Project Status

**Status**: ✅ **Complete and Compliant**

- [x] All 4 experiments implemented
- [x] Full PDF specification compliance
- [x] Comprehensive documentation (PRD, Architecture, API, Results)
- [x] Code pushed to GitHub
- [x] Reproducible setup instructions
- [x] Cost/token analysis included
- [x] Prompt engineering log complete

**Compliance**: 100% (see `COMPLIANCE_VERIFICATION.md`)

---

**Version**: 1.0.0
**Last Updated**: 2025-12-06
**Build Status**: Passing ✅
**Documentation**: Complete ✅
**Test Coverage**: N/A (research project)

---

**Quick Start**: `ollama pull llama2 && pip install -r requirements.txt && python -m src.experiments.exp1_needle_haystack`
