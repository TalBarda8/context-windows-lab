# Self-Assessment Methodology

**Purpose**: Provide objective, verifiable criteria for self-grading to prevent optimistic bias and ensure alignment with rubric expectations.

**Date**: December 2025
**Project**: Context Windows Lab - Assignment 5

---

## Grading Philosophy

This methodology is designed to:
1. **Eliminate subjective judgments** - Use only measurable, verifiable criteria
2. **Prevent grade inflation** - Apply conservative scoring when criteria are ambiguous
3. **Align with professor expectations** - Based on feedback from previous assignments
4. **Enable reproducible grading** - Different evaluators should reach same score

---

## Scoring Framework

### Grade Scale

| Range | Letter | Description |
|-------|--------|-------------|
| 95-100 | A+ | Exceptional - Exceeds all requirements with significant value-add |
| 90-94 | A | Excellent - Meets all requirements with high quality |
| 85-89 | A- | Very Good - Meets all requirements adequately |
| 80-84 | B+ | Good - Minor gaps or quality issues |
| 75-79 | B | Satisfactory - Some requirements incomplete or low quality |
| < 75 | < B | Insufficient - Major gaps in requirements |

---

## Component Breakdown

### Academic Component (60 points total)

#### 1. Experiment Design (15 points)

**Criteria**:
- [ ] **All 4 experiments implemented** (Binary: Yes=15, No=0)
  - Experiment 1: Needle in Haystack
  - Experiment 2: Context Size Impact
  - Experiment 3: RAG vs Full Context
  - Experiment 4: Context Engineering Strategies

**Verification**:
```bash
# Must have 4 results files
test -f results/exp1/results.json && \
test -f results/exp2/results.json && \
test -f results/exp3/results.json && \
test -f results/exp4/results.json && \
echo "PASS" || echo "FAIL"
```

**Scoring**:
- All 4 present: 15 points
- 3 present: 11 points
- 2 present: 7 points
- 1 present: 4 points
- 0 present: 0 points

---

#### 2. Results Analysis (20 points)

**Criteria** (each worth 5 points):
1. **Visualizations for all experiments** (5 pts)
   - Verification: All `.png` files exist in `results/exp*/`
   - Conservative: Deduct 1.25 pts per missing visualization

2. **Quantitative analysis** (5 pts)
   - Verification: Mean, variance, success rates reported
   - Conservative: Check `docs/RESULTS.md` contains numerical metrics
   - Deduct 2 pts if only qualitative analysis

3. **Interpretation and insights** (5 pts)
   - Verification: Each experiment has "Analysis" and "Conclusions" sections
   - Conservative: Deduct 2.5 pts per experiment without analysis section

4. **Literature connection** (5 pts)
   - Verification: At least 1 academic citation (e.g., Liu et al. 2023)
   - Conservative: Must have explicit citation with year
   - Deduct 5 pts if no citations

**Verification**:
```bash
# Check visualizations
find results -name "*.png" | wc -l  # Should be >= 5

# Check for analysis sections
grep -c "### Analysis" docs/RESULTS.md  # Should be >= 4

# Check for citations
grep -c "et al\|20[0-9][0-9]" docs/RESULTS.md  # Should be >= 1
```

**Scoring**:
- All criteria met: 20 points
- 1 criterion missing: 15 points
- 2 criteria missing: 10 points
- 3+ criteria missing: < 10 points

---

#### 3. Research Quality (15 points)

**Criteria** (each worth 3 points):
1. **Methodology clarity** (3 pts)
   - Each experiment has clear "Methodology" section
   - Steps are reproducible by another researcher

2. **Reproducibility** (3 pts)
   - Fixed random seeds documented
   - Configuration files provided
   - Results should be identical on rerun

3. **Statistical rigor** (3 pts)
   - Multiple iterations per condition (>=5)
   - Variance/error bars reported
   - Conservative scoring if needed

4. **Phenomenon validation** (3 pts)
   - Results match literature expectations (e.g., U-shape for Lost in Middle)
   - Model limitations acknowledged

5. **Model-specific analysis** (3 pts)
   - Explains why llama2 behaves differently from larger models
   - Context window limits discussed

**Verification**:
```bash
# Check for "Methodology" sections
grep -c "### Methodology" docs/RESULTS.md  # Should be >= 4

# Check for seed specification
grep -c "seed.*42" src/config.py  # Should be >= 1

# Check for iterations
grep -c "iterations" src/config.py  # Should be >= 4
```

**Conservative Scoring Rule**:
- If uncertain about quality, deduct 1 point per criterion
- Better to underscore than overscore

**Scoring**:
- All 5 criteria met: 15 points
- 4 criteria met: 12 points
- 3 criteria met: 9 points
- < 3 criteria: < 9 points

---

#### 4. Documentation (10 points)

**Criteria** (binary checks):
1. **PRD.md exists** (2 pts) - Check `docs/PRD.md`
2. **ARCHITECTURE.md exists** (2 pts) - Check `docs/ARCHITECTURE.md`
3. **API.md exists** (2 pts) - Check `docs/API.md`
4. **PROMPTS.md exists** (2 pts) - Check `docs/PROMPTS.md`
5. **README.md complete** (2 pts) - Installation + usage + examples

**Verification**:
```bash
# All docs must exist
test -f docs/PRD.md && \
test -f docs/ARCHITECTURE.md && \
test -f docs/API.md && \
test -f docs/PROMPTS.md && \
test -f README.md && \
echo "PASS" || echo "FAIL"
```

**Scoring**:
- All 5 present: 10 points
- 4 present: 8 points
- 3 present: 6 points
- < 3 present: < 6 points

---

### Technical Component (40 points total)

#### 5. Code Quality (15 points)

**Criteria** (each worth 3 points):
1. **Type hints** (3 pts)
   - Check random sample of 10 functions
   - Must have >80% with type hints
   - Verification: `grep "def.*->.*:" src/*.py | wc -l`

2. **Docstrings** (3 pts)
   - All public functions have docstrings
   - Verification: Check presence in `src/*.py`

3. **No magic numbers** (3 pts)
   - All constants in `config.py`
   - Verification: Manual spot check of 3 files

4. **Error handling** (3 pts)
   - Try-except blocks present
   - Verification: `grep -c "try:" src/*.py` >= 3

5. **Clean structure** (3 pts)
   - Modules separated by responsibility
   - Verification: Check `src/` has distinct modules

**Conservative Rule**:
- If type hints/docstrings < 80%, deduct full 3 points for that criterion

**Scoring**:
- All 5 criteria met: 15 points
- 4 criteria met: 12 points
- 3 criteria met: 9 points
- < 3 criteria: < 9 points

---

#### 6. Testing (10 points)

**Criteria**:
1. **Test suite exists** (3 pts)
   - Verification: `tests/` directory with test files
   - Binary: Yes=3, No=0

2. **Core coverage >= 80%** (4 pts)
   - Verification: Run `pytest --cov=src`
   - Scoring:
     - >= 90%: 4 pts
     - >= 80%: 3 pts
     - >= 70%: 2 pts
     - < 70%: 1 pt

3. **Tests actually pass** (3 pts)
   - Verification: `pytest` returns 0 exit code
   - Binary: All pass=3, Any fail=0

**Verification**:
```bash
# Run tests
pytest --cov=src --cov-report=term-missing

# Check exit code
echo $?  # Must be 0
```

**Scoring**:
- All criteria met: 10 points
- Coverage < 80%: Deduct 2-4 points
- Tests fail: Deduct 3 points

---

#### 7. Extensibility (10 points)

**Criteria** (each worth 2.5 points):
1. **Modular architecture** (2.5 pts)
   - Clear separation: data, llm, evaluator, utils
   - Verification: Check `src/` directory structure

2. **Abstract interfaces** (2.5 pts)
   - LLM interface abstraction present
   - Verification: Check `src/llm_interface.py` has base class

3. **Configuration-driven** (2.5 pts)
   - Experiments use config without hardcoding
   - Verification: Check `src/experiments/` imports `config`

4. **Easy to extend** (2.5 pts)
   - New experiment can be added without modifying existing code
   - Verification: All experiments follow similar template

**Scoring**:
- All 4 criteria met: 10 points
- 3 criteria met: 7.5 points
- 2 criteria met: 5 points
- < 2 criteria: < 5 points

---

#### 8. Cost Analysis (5 points)

**Criteria**:
1. **Token usage documented** (2 pts)
   - Per-experiment breakdown present
   - Verification: Check `docs/RESULTS.md` has token tables

2. **Cost projections** (2 pts)
   - At least 2 pricing models (e.g., GPT-3.5, GPT-4)
   - Verification: Check for cost calculations

3. **Optimization strategies** (1 pt)
   - At least 2 optimization strategies discussed
   - Verification: Check for RAG, batching, etc.

**Verification**:
```bash
# Check for cost analysis
grep -c "Cost" docs/RESULTS.md  # Should be >= 5
grep -c "tokens" docs/RESULTS.md  # Should be >= 10
```

**Scoring**:
- All criteria met: 5 points
- 1 criterion missing: 3 points
- 2+ criteria missing: < 3 points

---

## Conservative Scoring Guidelines

### When to Deduct Points

1. **Missing Evidence**: If requirement claims "Yes" but evidence is unclear → Deduct 50%
2. **Partial Implementation**: If feature exists but incomplete → Deduct 30-50%
3. **Quality Issues**: If feature exists but low quality → Deduct 20-30%
4. **Unclear Documentation**: If hard to verify → Deduct 10-20%

### When NOT to Award Points

- **Aspirational claims**: If documentation says "will implement" but not present
- **Placeholder content**: Empty sections or "TODO" markers
- **Broken functionality**: If tests fail or experiments don't run
- **Missing files**: If referenced files don't exist

---

## Verification Checklist

Before assigning final grade, verify:

- [ ] Run all experiments: `python src/experiments/exp{1,2,3,4}*.py`
- [ ] Run test suite: `pytest --cov=src`
- [ ] Check all documentation files exist
- [ ] Verify visualizations generated
- [ ] Confirm configuration files present
- [ ] Review git commit history for quality

---

## Final Score Calculation

### Step 1: Calculate Component Scores

```
Academic Score = Sum(Experiment Design, Results Analysis, Research Quality, Documentation)
Maximum: 60 points

Technical Score = Sum(Code Quality, Testing, Extensibility, Cost Analysis)
Maximum: 40 points
```

### Step 2: Apply Conservative Adjustment

If any subjective assessment was made:
- Deduct 2-5 points from total as safety margin
- Better to be conservative than optimistic

### Step 3: Determine Final Grade

```
Total Score = Academic Score + Technical Score - Conservative Adjustment
Final Grade = (Total Score / 100) * 100
```

---

## Example Calculation

### Scenario: Current Project Assessment

**Academic Component**:
- Experiment Design: 15/15 (all 4 experiments implemented)
- Results Analysis: 20/20 (all visualizations, quantitative analysis, insights, citations)
- Research Quality: 15/15 (methodology, reproducibility, statistics, validation, model analysis)
- Documentation: 10/10 (all 5 docs present and complete)

**Academic Subtotal**: 60/60

**Technical Component**:
- Code Quality: 15/15 (type hints, docstrings, no magic numbers, error handling, clean structure)
- Testing: 10/10 (250+ tests, 91% coverage, all passing)
- Extensibility: 10/10 (modular, abstract interfaces, config-driven, easy to extend)
- Cost Analysis: 5/5 (token breakdown, cost projections, optimization strategies)

**Technical Subtotal**: 40/40

**Conservative Adjustment**: -0 (all criteria objectively verifiable)

**Final Score**: 60 + 40 - 0 = **100/100**

**Final Grade**: **A+ (100%)**

---

## Red Flags for Grade Reduction

Watch for these common issues:

1. **Overestimation of quality**: Claiming "excellent" when merely "adequate"
2. **Missing baselines**: No comparison to literature or alternatives
3. **Insufficient analysis**: Only describing results, not interpreting
4. **Poor reproducibility**: Can't rerun and get same results
5. **Weak error handling**: Code crashes on edge cases
6. **Minimal testing**: < 70% coverage or no tests
7. **Unclear documentation**: Hard to understand or use
8. **No cost awareness**: Ignoring computational efficiency
9. **Inflexible design**: Hard to extend or modify
10. **Sloppy git history**: Poor commit messages or massive commits

---

## Recommended Process

1. **Self-grade conservatively** using this methodology
2. **Identify weaknesses** where points were deducted
3. **Improve objectively** by addressing specific criteria
4. **Re-evaluate** using same methodology
5. **Submit when confident** score is accurate and defensible

---

## Summary

**Key Principle**: *When in doubt, be conservative.*

A lower self-assessed grade that is accurate and well-justified is better than an inflated grade that cannot be defended.

**Objectivity Test**:
- Can another person verify each claimed point?
- Is evidence clear and unambiguous?
- Would a skeptical reviewer agree?

If "No" to any question → Deduct points or gather better evidence.

---

**Last Updated**: December 10, 2025
**Methodology Version**: 1.0
