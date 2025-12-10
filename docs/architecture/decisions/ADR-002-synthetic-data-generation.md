# ADR-002: Synthetic Data Generation

**Status**: Accepted
**Date**: November 2025
**Impact**: High - Affects experiment validity and reproducibility

---

## Decision

**Generate all experimental data synthetically using Faker library with deterministic seeding (seed=42).**

---

## Context

Experiments require:
- Needle-in-haystack documents (Exp 1)
- Business documents with revenue facts (Exp 2)
- Hebrew medical corpus (Exp 3)
- Agent action histories (Exp 4)

---

## Alternatives

### Alternative 1: Real-world data collection
**Rejected**: Time-consuming, privacy concerns, hard to control variables

### Alternative 2: Manual creation
**Rejected**: Not scalable, introduces bias, poor reproducibility

### Alternative 3: GPT-generated content
**Rejected**: Costs money, not reproducible, potential circular reasoning

---

## Rationale

1. **Reproducibility**: Fixed seed=42 ensures identical data every run
2. **Control**: Can precisely control document length, fact placement, distractors
3. **Cost**: $0 - no API calls or data acquisition costs
4. **Speed**: Instant generation vs hours of collection
5. **Privacy**: No PII or sensitive data concerns
6. **Scalability**: Easy to generate 10 or 10,000 documents

---

## Implementation

- Use `Faker` library for realistic text generation
- Separate `Random` instances per generator for isolation
- All parameters in `config.py` for configurability
- Character-based positioning for precise fact embedding

---

## Consequences

✅ Perfect reproducibility
✅ Zero cost
✅ Controlled experiments
⚠️ Less realistic than real data (acceptable for methodology demonstration)

---

**Last Updated**: December 10, 2025
