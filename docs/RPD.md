# RPD: Context Windows Lab - Research, Plan & Development Document

## 1. Research Questions

### Primary Research Question:
**How do context window characteristics and management strategies affect the accuracy and efficiency of Large Language Models in practical scenarios?**

### Sub-Questions:
1. **Lost in the Middle**: Does the position of critical information within a context window significantly impact retrieval accuracy?
2. **Context Size Impact**: How does increasing context window size affect model accuracy, latency, and token consumption?
3. **RAG Effectiveness**: Does Retrieval-Augmented Generation (RAG) provide measurable improvements over full-context approaches in terms of accuracy and speed?
4. **Context Engineering**: Which context management strategies (Select/Compress/Write) perform best for maintaining accuracy in sequential multi-step scenarios?

---

## 2. Experiments We Will Run

### Experiment 1: Needle in Haystack (Lost in the Middle)
**Objective**: Demonstrate that LLMs struggle to retrieve information from the middle of long contexts

**Methodology**:
- Generate 5 synthetic documents (200 words each)
- Embed a critical fact in each document at different positions (start/middle/end)
- Query the LLM to retrieve the critical fact
- Measure accuracy by fact position
- Run 10 iterations per position for statistical significance

**Metrics**:
- Accuracy rate by position (start/middle/end)
- Confidence scores (if available)

### Experiment 2: Context Window Size Impact
**Objective**: Quantify how context size degrades performance

**Methodology**:
- Test with varying document counts: 2, 5, 10, 20, 50 documents
- For each size, measure:
  - Response latency (time to generate answer)
  - Token count (actual context size)
  - Accuracy of responses
- Run 5 iterations per size

**Metrics**:
- Accuracy vs. context size
- Latency vs. context size
- Tokens used vs. number of documents

### Experiment 3: RAG Impact
**Objective**: Compare full-context vs. RAG-based retrieval

**Methodology**:
- Create a corpus of 20 Hebrew documents (topics: technology, law, medicine)
- Test question: "What are the side effects of drug X?"
- Mode A: Full Context - pass all 20 documents to LLM
- Mode B: RAG - use similarity search to retrieve top-k=3 relevant chunks
- Compare accuracy, latency, and response quality

**Metrics**:
- Accuracy (full vs. RAG)
- Latency (full vs. RAG)
- Precision/recall of information retrieval

### Experiment 4: Context Engineering Strategies
**Objective**: Evaluate different context management strategies

**Methodology**:
- Simulate a multi-agent system with 10 sequential actions
- Each action produces output that adds to context
- Test 3 strategies:
  - **SELECT**: Use RAG to retrieve only relevant history (k=5)
  - **COMPRESS**: Auto-summarize history when exceeding MAX_TOKENS
  - **WRITE**: External memory (scratchpad) for key facts
- Measure accuracy degradation over time

**Metrics**:
- Accuracy per action step (1-10) for each strategy
- Context size growth rate
- Latency per strategy

---

## 3. Expected Results

### Experiment 1 - Expected Pattern:
- **Start position**: 85-95% accuracy
- **Middle position**: 40-60% accuracy (significant drop)
- **End position**: 80-90% accuracy
- **Conclusion**: U-shaped accuracy curve (high at edges, low in middle)

### Experiment 2 - Expected Pattern:
- Small contexts (2-5 docs): High accuracy (~90%)
- Medium contexts (10-20 docs): Moderate accuracy (~70%)
- Large contexts (50 docs): Lower accuracy (~50-60%)
- Latency increases linearly with context size
- **Conclusion**: Accuracy inversely correlates with context size

### Experiment 3 - Expected Pattern:
- **RAG**: Higher accuracy (85-95%), faster response (<2s)
- **Full Context**: Lower accuracy (60-70%), slower response (>5s)
- **Conclusion**: RAG provides both accuracy and efficiency gains

### Experiment 4 - Expected Pattern:
- **SELECT**: Best accuracy, maintains performance across all 10 steps
- **COMPRESS**: Good accuracy initially, slight degradation after step 5
- **WRITE**: Consistent performance, depends on key fact extraction quality
- **Conclusion**: SELECT (RAG-based) is most effective for long conversations

---

## 4. Architecture and Code Components

### 4.1 Core Components

```
┌─────────────────────────────────────────────────────┐
│                  Main Orchestrator                   │
│         (experiment_runner.py)                       │
└─────────────────┬───────────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
   ┌────▼─────┐      ┌─────▼──────┐
   │   Data   │      │    LLM     │
   │Generator │      │  Interface │
   └────┬─────┘      └─────┬──────┘
        │                   │
   ┌────▼─────────────┬────▼──────┬──────────┐
   │                  │           │          │
┌──▼──────┐  ┌───────▼────┐ ┌───▼─────┐ ┌──▼──────┐
│Exp 1    │  │  Exp 2     │ │ Exp 3   │ │ Exp 4   │
│Needle   │  │  Size      │ │ RAG     │ │ Strategies│
└──┬──────┘  └───────┬────┘ └───┬─────┘ └──┬──────┘
   │                  │          │          │
   └──────────────────┴──────────┴──────────┘
                      │
              ┌───────▼────────┐
              │   Visualizer   │
              │ (plots/tables) │
              └────────────────┘
```

### 4.2 Technology Stack

**LLM Framework**:
- **Ollama**: Local LLM inference (primary model: llama3.2 or mistral)
- **LangChain**: For chaining, memory management, and RAG
- **ChromaDB**: Vector database for embeddings and similarity search
- **Nomic Embed Text**: Embedding model for RAG

**Data Processing**:
- **Python 3.10+**: Core language
- **NumPy/Pandas**: Data manipulation and statistics
- **Matplotlib/Seaborn**: Visualization

**Utilities**:
- **tiktoken**: Token counting
- **faker**: Synthetic data generation
- **tqdm**: Progress bars

---

## 5. File Structure

```
context-windows-lab/
│
├── README.md                          # Project overview and setup
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore file
│
├── docs/
│   ├── RPD.md                         # This document
│   └── RESULTS.md                     # Final results and analysis
│
├── data/
│   ├── synthetic/                     # Generated synthetic documents
│   │   ├── needle_haystack/          # Exp 1 data
│   │   └── context_size/             # Exp 2 data
│   └── hebrew_corpus/                 # Exp 3 Hebrew documents
│
├── src/
│   ├── __init__.py
│   ├── config.py                      # Configuration and constants
│   ├── llm_interface.py               # Ollama/LangChain interface
│   ├── data_generator.py              # Synthetic data creation
│   ├── evaluator.py                   # Accuracy evaluation logic
│   │
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── exp1_needle_haystack.py    # Experiment 1
│   │   ├── exp2_context_size.py       # Experiment 2
│   │   ├── exp3_rag_impact.py         # Experiment 3
│   │   └── exp4_strategies.py         # Experiment 4
│   │
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py           # Plotting utilities
│       └── metrics.py                 # Statistical calculations
│
├── notebooks/
│   ├── exp1_analysis.ipynb            # Experiment 1 analysis
│   ├── exp2_analysis.ipynb            # Experiment 2 analysis
│   ├── exp3_analysis.ipynb            # Experiment 3 analysis
│   └── exp4_analysis.ipynb            # Experiment 4 analysis
│
├── results/
│   ├── exp1/                          # Experiment 1 outputs
│   ├── exp2/                          # Experiment 2 outputs
│   ├── exp3/                          # Experiment 3 outputs
│   └── exp4/                          # Experiment 4 outputs
│
└── scripts/
    ├── run_all_experiments.sh         # Run all experiments
    └── setup_environment.sh           # Environment setup
```

---

## 6. Evaluation Methodology

### 6.1 Accuracy Metrics

**For Experiments 1, 2, 3**:
- **Exact Match**: Binary score (1 if correct fact retrieved, 0 otherwise)
- **Partial Match**: Fuzzy string matching (e.g., Levenshtein distance)
- **Semantic Similarity**: Embedding-based similarity between expected and actual answer

**For Experiment 4**:
- **Task Success Rate**: Percentage of correct responses per strategy
- **Context Efficiency**: Ratio of tokens used to accuracy achieved

### 6.2 Statistical Validation
- Run each experiment configuration **10 times**
- Calculate: Mean, Standard Deviation, 95% Confidence Interval
- Use statistical tests (t-test, ANOVA) where appropriate

### 6.3 Visualization
- **Experiment 1**: Bar chart showing accuracy by position (start/middle/end)
- **Experiment 2**: Line graph showing accuracy vs. context size
- **Experiment 3**: Comparison bar chart (RAG vs. Full Context)
- **Experiment 4**: Multi-line graph showing accuracy degradation over 10 steps

### 6.4 Reporting
- Summary tables with all metrics
- Statistical significance indicators
- Conclusions and recommendations

---

## 7. Assumptions and Ambiguities

### 7.1 Assumptions

1. **LLM Model**: We will use **Ollama with llama3.2** (or mistral if unavailable) as the primary model. Assumption: Model is available locally.

2. **Language**:
   - Experiments 1, 2, 4: English (easier synthetic data generation)
   - Experiment 3: Hebrew (as specified in assignment)

3. **Token Limits**: Assuming model context window of ~4K-8K tokens (standard for local models)

4. **Synthetic Data Quality**: Generated filler text will be coherent enough to simulate realistic documents

5. **Ground Truth**: For accuracy evaluation, we will use exact fact matching and manual verification of a sample

### 7.2 Ambiguities and Resolutions

| Ambiguity | Resolution |
|-----------|------------|
| **What specific LLM to use?** | Use Ollama with llama3.2 (3B or 8B variant). If unavailable, fallback to mistral:7b |
| **How to generate "realistic" synthetic documents?** | Use a combination of faker library + template-based generation + LLM-generated filler text |
| **What defines "accuracy" for open-ended questions?** | Use multi-criteria: (1) Exact match, (2) Keyword presence, (3) Semantic similarity >0.8 |
| **Hebrew corpus for Exp 3 - where to get it?** | Generate synthetic Hebrew documents using LLM with domain-specific prompts (medical, legal, tech) |
| **MAX_TOKENS threshold for COMPRESS strategy?** | Set to 2048 tokens (half of typical 4K context window) |
| **How to implement "scratchpad" in WRITE strategy?** | Use a simple key-value store with LLM-extracted facts as keys |
| **k value for RAG retrieval?** | Use k=3 for Exp 3, k=5 for Exp 4 SELECT strategy (based on typical best practices) |
| **Chunk size for RAG?** | 500 tokens per chunk (as specified in pseudocode) |
| **What embedding model for RAG?** | nomic-embed-text (specified in assignment, good for both English and Hebrew) |

### 7.3 Technical Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| **Ollama not installed** | Include setup instructions in README; provide Docker alternative |
| **Model too slow on local machine** | Use smaller model variants (3B instead of 7B); reduce iteration counts |
| **Hebrew text handling issues** | Ensure UTF-8 encoding throughout; test Hebrew rendering early |
| **ChromaDB installation issues** | Use in-memory mode as fallback; provide alternative with FAISS |
| **Inconsistent LLM outputs** | Set temperature=0 for deterministic results; use seed where possible |

---

## 8. Success Criteria

This lab will be considered successful when:

1. ✅ All 4 experiments run end-to-end without errors
2. ✅ Results statistically validate expected patterns (with 95% confidence)
3. ✅ Visualizations clearly demonstrate the phenomena being studied
4. ✅ Code is well-documented, modular, and reproducible
5. ✅ Final report (RESULTS.md) contains:
   - Clear explanations of findings
   - Graphs and tables
   - Statistical analysis
   - Conclusions and insights
6. ✅ All code is committed to git with meaningful commit messages

---

## 9. Timeline and Milestones

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| **Phase 1: Setup** | 30 min | Environment setup, dependencies installed, basic structure |
| **Phase 2: Data Generation** | 45 min | Synthetic data generators for all experiments |
| **Phase 3: LLM Interface** | 30 min | Ollama/LangChain integration, basic query functions |
| **Phase 4: Experiment 1** | 1 hour | Exp 1 complete with results and visualization |
| **Phase 5: Experiment 2** | 1 hour | Exp 2 complete with results and visualization |
| **Phase 6: Experiment 3** | 1.5 hours | RAG implementation, Exp 3 complete |
| **Phase 7: Experiment 4** | 1.5 hours | Context strategies, Exp 4 complete |
| **Phase 8: Analysis** | 1 hour | Notebooks with statistical analysis |
| **Phase 9: Documentation** | 30 min | Final RESULTS.md and README updates |

**Total Estimated Time**: ~8 hours

---

## 10. Dependencies

```
python>=3.10
ollama
langchain>=0.1.0
langchain-community
chromadb
nomic
numpy
pandas
matplotlib
seaborn
tiktoken
faker
python-dotenv
tqdm
jupyter
notebook
scikit-learn
```

---

## RPD Summary

This RPD provides a comprehensive blueprint for executing the Context Windows Lab. The plan is:
- **Modular**: Each experiment is independent
- **Reproducible**: All parameters documented, deterministic where possible
- **Statistically rigorous**: Multiple iterations, confidence intervals
- **Practical**: Uses real tools (Ollama, LangChain, ChromaDB)
- **Well-structured**: Clear file organization and separation of concerns

The experiments will provide empirical evidence for:
1. Lost in the Middle phenomenon
2. Context size vs. accuracy tradeoff
3. Benefits of RAG over full-context approaches
4. Effectiveness of different context engineering strategies

---

**Document Version**: 1.0
**Date**: 2025-12-05
**Status**: Approved ✅
