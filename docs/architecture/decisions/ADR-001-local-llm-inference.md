# ADR-001: Local LLM Inference with Ollama

**Status**: Accepted
**Date**: November 2025
**Decision Makers**: Development Team
**Impact**: High - Affects cost, reproducibility, and deployment

---

## Context

We need to run 4 context window experiments with ~87 LLM calls totaling ~220K tokens. We must decide between:
1. Cloud API services (OpenAI, Anthropic)
2. Local inference with open-source models

---

## Decision

**We will use Ollama for local LLM inference with llama2 (7B).**

---

## Rationale

### Advantages of Local Inference

1. **Zero Cost**
   - No per-token charges
   - $0 total vs $0.11 (GPT-3.5) or $6.67 (GPT-4)
   - Enables unlimited experimentation during development

2. **No Rate Limits**
   - Can run experiments sequentially without delays
   - No quota concerns for iterative testing

3. **Data Privacy**
   - All data stays local
   - No external API calls
   - Important for research reproducibility

4. **Reproducibility**
   - Controlled environment
   - Specific model version
   - Deterministic with seed=42

5. **Offline Operation**
   - No internet dependency
   - Reliable for demonstrations

### Why Ollama Specifically

1. **Easy Setup**
   - Single command installation
   - Simple model management
   - Cross-platform support

2. **Python Integration**
   - Native Python client library
   - Clean API similar to cloud providers
   - Easy to abstract for future migration

3. **Model Selection**
   - llama2 (7B) is well-documented
   - Appropriate size for academic study
   - Shows position effects at shorter contexts

---

## Alternatives Considered

### Alternative 1: OpenAI GPT-3.5

**Pros**:
- Higher quality responses
- Better multilingual support
- Faster inference
- Industry standard

**Cons**:
- $0.11 for this project (cheap but not free)
- Requires API key management
- Network dependency
- Rate limits
- Less reproducible (model updates)

**Why Rejected**: Cost adds up during development (dozens of iterations). For ~$100 budget, prefer to spend on experimentation rather than API calls.

### Alternative 2: OpenAI GPT-4

**Pros**:
- Highest quality
- Best for complex reasoning
- Would show "Lost in Middle" at longer contexts

**Cons**:
- $6.67 for this project
- Very expensive at scale
- Overkill for simple retrieval tasks
- Would need 10K-30K token contexts (expensive)

**Why Rejected**: Cost prohibitive. llama2 is sufficient to demonstrate phenomena.

### Alternative 3: Anthropic Claude

**Pros**:
- Excellent quality
- Good pricing ($0.06)
- Strong at longer contexts

**Cons**:
- Requires API key
- Not free
- Limited availability

**Why Rejected**: Same cost concerns as OpenAI, though cheaper.

### Alternative 4: Hugging Face Transformers (Direct)

**Pros**:
- Free
- More control
- Can use GPU directly

**Cons**:
- Complex setup
- Requires CUDA/PyTorch expertise
- Memory management challenges
- Slower than Ollama
- No built-in chat interface

**Why Rejected**: Ollama provides same benefits with easier setup.

---

## Consequences

### Positive

- ✅ **Cost**: $0 for entire project
- ✅ **Flexibility**: Can iterate freely
- ✅ **Reproducibility**: Exact model version controlled
- ✅ **Privacy**: No data leaves local machine
- ✅ **Educational**: Students can replicate easily

### Negative

- ⚠️ **Quality**: llama2 (7B) less capable than GPT-4
- ⚠️ **Speed**: Slower inference (2-3s per call vs < 1s for cloud)
- ⚠️ **Context Window**: 4K tokens vs 128K+ for newer models
- ⚠️ **Multilingual**: Weaker on Hebrew (Exp 3)
- ⚠️ **Setup**: Requires local installation

### Mitigation Strategies

1. **Quality**: Acceptable for demonstrating phenomena; focus on methodology not absolute performance
2. **Speed**: Pre-generate data; run experiments overnight if needed
3. **Context Window**: Design experiments within 4K limit; use shorter contexts
4. **Multilingual**: Acknowledge limitation; focus on method demonstration
5. **Setup**: Provide clear installation instructions; Docker alternative

---

## Trade-offs

| Factor | Local (Ollama) | Cloud (GPT-3.5) | Cloud (GPT-4) |
|--------|----------------|-----------------|---------------|
| **Cost** | $0 | $0.11 | $6.67 |
| **Quality** | Good | Excellent | Outstanding |
| **Speed** | 2-3s/call | <1s/call | <1s/call |
| **Context** | 4K | 16K | 128K |
| **Setup** | Medium | Easy | Easy |
| **Reproducibility** | Excellent | Good | Good |

**Decision**: Prioritize cost and reproducibility over absolute quality.

---

## Implementation Notes

### Interface Abstraction

Created `LLMInterface` class to abstract Ollama details:

```python
class LLMInterface:
    def query(self, prompt: str, context: str = "", **kwargs) -> Dict:
        """Abstract query method - can swap providers"""
        pass
```

This enables future migration to cloud APIs if needed.

### Configuration

All Ollama settings in `config.py`:
```python
OLLAMA_BASE_URL = "http://localhost:11434"
PRIMARY_MODEL = "llama2"
FALLBACK_MODEL = "mistral"
LLM_TEMPERATURE = 0.0  # Deterministic
LLM_SEED = 42  # Reproducible
```

### Fallback Strategy

If llama2 unavailable, falls back to mistral. If neither available, provides clear error message with installation instructions.

---

## Validation

### Success Criteria

- [x] Ollama installation < 5 minutes
- [x] All 4 experiments complete successfully
- [x] Results reproducible with seed=42
- [x] Total cost = $0
- [x] Can run offline

### Validation Results

- Installation: ~2 minutes (download model ~10 min)
- All experiments: ✅ Complete
- Reproducibility: ✅ Zero variance across 5 replications
- Cost: ✅ $0
- Offline: ✅ Works without internet after model download

**Decision validated successfully.**

---

## Future Considerations

### When to Reconsider

Consider switching to cloud API if:
1. Budget increases (>$100 available)
2. Multilingual quality becomes critical
3. Longer contexts required (>4K tokens)
4. Publication requires state-of-the-art model
5. Scale increases significantly (>1000 experiments)

### Migration Path

If switching to cloud API:
1. Update `LLMInterface` implementation
2. Add API key management
3. Update `config.py` with new URLs/models
4. Re-run experiments and compare results
5. Update cost analysis in documentation

Interface abstraction makes this straightforward.

---

## References

- [Ollama Documentation](https://ollama.ai/docs)
- [llama2 Model Card](https://huggingface.co/meta-llama/Llama-2-7b)
- [OpenAI Pricing](https://openai.com/pricing)
- [Anthropic Pricing](https://www.anthropic.com/pricing)

---

**Last Updated**: December 10, 2025
**Review Date**: Before next major experiment iteration
