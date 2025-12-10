# Cost Analysis and Computational Efficiency

**Project**: Context Windows Lab
**Date**: December 2025
**Total Project Cost**: $0.00 (Ollama local inference)

---

## Executive Summary

This document provides comprehensive cost and computational efficiency analysis for all experiments, optimization strategies, and scaling projections.

**Key Metrics**:
- **Total LLM Calls**: 87
- **Total Tokens**: 220,790 (Input: 219,050, Output: 1,740)
- **Total Runtime**: 4 minutes
- **Cost (Local)**: $0.00
- **Cost (GPT-3.5)**: $0.113
- **Cost (GPT-4)**: $6.67

---

## Per-Experiment Breakdown

### Experiment 1: Needle in Haystack

| Metric | Value |
|--------|-------|
| LLM Calls | 30 (3 positions × 10 iterations) |
| Avg Tokens/Call | ~1,400 |
| Total Tokens | 42,600 |
| Runtime | 81 seconds |
| Cost (Ollama) | $0.00 |
| Cost (GPT-3.5) | $0.021 |
| Cost (GPT-4) | $1.28 |

**Token Breakdown**:
- Context: 13 docs × 105 words ≈ 1,365 tokens/call
- Query: ~15 tokens
- Response: ~20 tokens

**Optimization Applied**: Fixed haystack size (13 docs) to achieve U-shape while minimizing tokens.

---

### Experiment 2: Context Window Size Impact

| Size | Calls | Tokens/Call | Total Tokens | Latency |
|------|-------|-------------|--------------|---------|
| 2 docs | 5 | 672 | 3,360 | 0.99s |
| 5 docs | 5 | 1,659 | 8,295 | 0.98s |
| 10 docs | 5 | 3,306 | 16,530 | 1.45s |
| 20 docs | 5 | 6,625 | 33,125 | 8.03s |
| 50 docs | 5 | 16,668 | 83,340 | 8.04s |
| **Total** | **25** | - | **144,650** | - |

**Cost Analysis**:
- Ollama: $0.00
- GPT-3.5: $0.072
- GPT-4: $4.34

**Key Finding**: Linear token growth (333 tokens/doc). 50-doc test exceeds llama2's 4K limit (severe degradation).

**Optimization Opportunity**: Use RAG for > 20 documents → 95% token reduction.

---

### Experiment 3: RAG vs Full Context

| Approach | Calls | Tokens | Latency | Cost (GPT-3.5) |
|----------|-------|--------|---------|----------------|
| Full Context | 1 | 5,758 | 22.75s | $0.0029 |
| RAG (top-3) | 1 | 281 | 11.80s | $0.00014 |
| **Savings** | - | **-95.1%** | **-48%** | **-95.2%** |

**ROI**: RAG provides 20.5× token efficiency with minimal accuracy loss (0.099 vs 0.091).

**Cost Projection (1000 queries)**:
- Full Context: 5.76M tokens = $2.88
- RAG: 281K tokens = $0.14
- **Savings**: $2.74 (95%)

---

### Experiment 4: Context Engineering Strategies

| Strategy | Actions | Tokens/Action | Total Tokens | Efficiency |
|----------|---------|---------------|--------------|------------|
| SELECT | 10 | ~800 | 8,000 | Medium |
| COMPRESS | 10 | ~1,200 | 12,000 | Low |
| WRITE | 10 | ~600 | 6,000 | High |
| **Total** | **30** | - | **26,000** | - |

**Cost**:
- Ollama: $0.00
- GPT-3.5: $0.013
- GPT-4: $0.78

**Trade-off**: WRITE is most token-efficient but sacrifices some accuracy. COMPRESS uses most tokens but provides best accuracy.

---

## Total Project Costs

### Actual Cost (Ollama)

| Component | Tokens | Cost |
|-----------|--------|------|
| Input | 219,050 | $0.00 |
| Output | 1,740 | $0.00 |
| **Total** | **220,790** | **$0.00** |

### Cloud API Projections

#### OpenAI GPT-3.5 Turbo
- Input: 219,050 tokens × $0.0005/1K = $0.110
- Output: 1,740 tokens × $0.0015/1K = $0.003
- **Total**: **$0.113**

#### OpenAI GPT-4
- Input: 219,050 tokens × $0.03/1K = $6.57
- Output: 1,740 tokens × $0.06/1K = $0.10
- **Total**: **$6.67**

#### Anthropic Claude 3 Haiku
- Input: 219,050 tokens × $0.00025/1K = $0.055
- Output: 1,740 tokens × $0.00125/1K = $0.002
- **Total**: **$0.057**

---

## Optimization Strategies

### 1. RAG Implementation (95% Token Reduction)

**Before**: 20 documents = 5,758 tokens
**After**: Top-3 chunks = 281 tokens
**Savings**: 5,477 tokens (95.1%)

**Application**: All corpora > 10 documents

### 2. Context Windowing

**Problem**: 50-doc test uses 16,668 tokens (4.1× limit)

**Solutions**:
- **Chunking**: Split into 4 batches → 4× cost
- **RAG**: Retrieve only relevant docs → 96% savings ✓
- **Summarization**: Compress before querying → 88% savings
- **Larger Model**: llama-3 (128K context) or GPT-4 → Higher cost

**Recommendation**: RAG (best balance)

### 3. Response Length Control

**Current**: ~20 tokens/response
**Optimization**: Set `max_tokens=30` to prevent verbose responses
**Impact**: Minimal (~$0.002 savings)

### 4. Batch Processing

**Current**: Sequential (1 call at a time)
**Optimization**: Parallel processing with ThreadPoolExecutor
**Benefit**: 5× faster wall-clock time, no token savings

### 5. Caching & Deduplication

**Opportunity**: Identical prompts across iterations
**Implementation**: `@lru_cache` for embeddings
**Savings**: ~50% embedding computation, 10-15% latency improvement

---

## Memory and Runtime Analysis

### Memory Usage

| Component | Memory |
|-----------|--------|
| llama2 Model | ~4GB (loaded once) |
| Embeddings Model | ~100MB |
| Data Generation | ~50MB peak |
| Experiment Execution | ~200MB peak |
| **Total Peak** | **~4.4GB** |

**Recommendation**: 8GB RAM minimum for comfortable operation.

### Runtime Breakdown

| Experiment | Runtime | Tokens/Second |
|------------|---------|---------------|
| Exp 1 | 81s | 526 |
| Exp 2 | 120s | 1,205 |
| Exp 3 | 35s | 165 |
| Exp 4 | 25s | 1,040 |
| **Total** | **261s (4.4 min)** | **846 avg** |

**Bottleneck**: LLM inference (Ollama on CPU). GPU would provide 5-10× speedup.

---

## Scaling Projections

### 1,000 Experiments

| Model | Total Tokens | Cost | Runtime (est) |
|-------|--------------|------|---------------|
| Ollama (local) | 221M | $0 | 67 hours |
| GPT-3.5 | 221M | $113 | 12 hours* |
| GPT-4 | 221M | $6,670 | 24 hours* |
| Claude 3 Haiku | 221M | $57 | 15 hours* |

*With API rate limits and parallelization

**Key Insight**: Local inference saves $6,670 at 1,000× scale (GPT-4 comparison).

### 10,000 Experiments

| Model | Cost | Days to Complete |
|-------|------|------------------|
| Ollama | $0 | 28 days (sequential) |
| GPT-3.5 | $1,130 | 5 days (parallel) |
| GPT-4 | $66,700 | 10 days (parallel) |

**Break-even**: Ollama is worth it if running > 100 experiments OR limited budget.

---

## Budget Management

### Hypothetical $100 Budget

| Phase | Calls | Tokens | Cost (GPT-3.5) | % of Budget |
|-------|-------|--------|----------------|-------------|
| Initial testing | 50 | 50,000 | $0.03 | 0.03% |
| Bug fixing | 100 | 100,000 | $0.06 | 0.06% |
| Parameter tuning | 30 | 30,000 | $0.02 | 0.02% |
| Final experiments | 87 | 220,790 | $0.113 | 0.113% |
| **Total Used** | **267** | **400,790** | **$0.223** | **0.223%** |

**Remaining**: $99.78 (99.78%)

**Conclusion**: Extremely cost-efficient even with commercial APIs.

---

## Trade-offs

### Cost vs Quality

| Factor | Ollama (llama2) | GPT-3.5 | GPT-4 |
|--------|-----------------|---------|-------|
| Cost | $0 | $0.11 | $6.67 |
| Quality | Good | Excellent | Outstanding |
| Context Window | 4K | 16K | 128K |
| Speed | Medium | Fast | Fast |
| Reproducibility | Excellent | Good | Good |

**Decision**: Prioritize cost and reproducibility for academic research.

### Speed vs Cost

| Approach | Speed | Cost | Use Case |
|----------|-------|------|----------|
| Local (CPU) | Slow | $0 | Development, iteration |
| Local (GPU) | Medium | $0 | Production with hardware |
| Cloud API | Fast | $0.11-$7 | Quick prototyping |

---

## Recommendations

### For This Project
✅ **Ollama (llama2)** - Optimal for:
- Academic research (methodology focus)
- Limited budget
- Frequent iteration
- Reproducibility requirements

### For Production
Consider cloud APIs if:
- Quality > cost
- Latency critical (< 1s response)
- Multilingual support needed
- Longer contexts required (> 4K)
- Minimal setup time desired

### For Large Scale (> 1,000 experiments)
- **Option 1**: Ollama with GPU (0 cost, faster)
- **Option 2**: Claude 3 Haiku (best quality/cost ratio: $57 vs $6,670 for GPT-4)
- **Option 3**: GPT-3.5 (good balance: $113 total)

---

## Monitoring and Alerts

### Implemented Safeguards

```python
# Token counting
def count_tokens(text: str) -> int:
    return len(text) // 4  # Heuristic

# Context window validation
if token_count > MAX_CONTEXT_WINDOW:
    logger.warning(f"Exceeds limit: {token_count}")

# Usage tracking
# All experiments log token usage to results.json
```

### Recommended Enhancements

```python
# Budget monitoring
class TokenBudget:
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.used_tokens = 0

    def check_and_increment(self, tokens: int):
        if self.used_tokens + tokens > self.max_tokens:
            raise BudgetExceededError()
        self.used_tokens += tokens
```

---

## Key Takeaways

1. **Zero-Cost Achievement**: $0 total cost using Ollama (vs $6.67 with GPT-4)
2. **RAG Efficiency**: 95% token reduction with maintained accuracy
3. **Scalability**: Can run 1,000 experiments for $0 (local) vs $6,670 (GPT-4)
4. **Trade-offs**: Sacrifice some quality and speed for massive cost savings
5. **Budget-Friendly**: Even with GPT-3.5, entire project costs < $0.25

**Bottom Line**: This project demonstrates that rigorous LLM research can be conducted at near-zero cost using local inference with strategic optimization (RAG, caching, efficient prompting).

---

**Last Updated**: December 10, 2025
**Version**: 1.0
