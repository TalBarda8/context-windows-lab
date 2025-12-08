# Product Requirements Document (PRD)
# Context Windows Lab

**Version**: 1.1.0
**Date**: December 8, 2025
**Owner**: Tal Barda
**Course**: LLMs in Multi-Agent Environments
**Instructor**: Dr. Yoram Segal

---

## 1. Executive Summary

### 1.1 Overview

The Context Windows Lab is a research platform designed to systematically investigate how context window characteristics and management strategies affect Large Language Model (LLM) performance. Through four comprehensive experiments, this project aims to quantify the impact of context size, position effects, RAG effectiveness, and context engineering strategies on LLM accuracy and efficiency.

### 1.2 Strategic Background

**Problem Statement:**
Large Language Models have limited context windows, and their ability to process information varies significantly based on context size, information position, and retrieval strategies. Current research lacks systematic comparative analysis of these factors under controlled conditions.

**Market Gap:**
While academic papers discuss "lost in the middle" phenomena and RAG benefits theoretically, there is limited open-source tooling for practitioners to reproduce and extend these experiments with local, controllable infrastructure.

**Opportunity:**
Create an educational research platform that demonstrates empirically how context window engineering affects LLM performance, enabling students and researchers to explore these concepts hands-on.

### 1.3 Target Audience

**Primary Stakeholders:**
- **Graduate Students**: Learning about LLM limitations and optimization
- **Researchers**: Investigating context window behaviors
- **Course Instructors**: Teaching advanced NLP concepts

**Secondary Stakeholders:**
- **LLM Application Developers**: Understanding production deployment considerations
- **Academic Community**: Reproducing and validating findings

---

## 2. Goals and Objectives

### 2.1 Business Goals

| Goal | Description | Success Metric |
|------|-------------|----------------|
| **G1: Educational Value** | Provide hands-on learning about LLM context limitations | Student comprehension > 80% |
| **G2: Research Contribution** | Generate reproducible empirical data | All 4 experiments complete |
| **G3: Open Source Impact** | Enable community extensions | GitHub stars/forks |
| **G4: Academic Excellence** | Demonstrate M.Sc.-level technical competency | Grade >= 90 |

### 2.2 Product Objectives

| ID | Objective | KPI | Target |
|----|-----------|-----|--------|
| **O1** | Measure position-based accuracy degradation | Accuracy delta | Document 30+ trials |
| **O2** | Quantify context size impact on performance | Latency vs size | Test 5 sizes |
| **O3** | Compare RAG vs full-context approaches | Accuracy/efficiency ratio | 2 methods compared |
| **O4** | Evaluate context management strategies | Strategy effectiveness | 3 strategies tested |
| **O5** | Produce publication-quality visualizations | Chart clarity | 4 charts generated |
| **O6** | Ensure reproducibility | Setup success rate | 100% on clean machine |

---

## 3. User Needs and Requirements

### 3.1 User Stories

#### Story 1: As a Graduate Student
```
As a graduate student studying LLMs,
I want to see empirical evidence of "lost in the middle" phenomena,
So that I understand why position matters in prompt engineering.

Acceptance Criteria:
✅ Clear visualization showing accuracy by position
✅ Statistical significance documentation
✅ Explainer text in RESULTS.md
```

#### Story 2: As a Researcher
```
As an NLP researcher,
I want to reproduce context window experiments locally,
So that I can validate published claims without cloud API costs.

Acceptance Criteria:
✅ All experiments run on local Ollama
✅ Complete parameter documentation
✅ JSON output for further analysis
✅ < 5 minute total runtime
```

#### Story 3: As a Course Instructor
```
As a course instructor,
I want students to understand RAG trade-offs,
So that they make informed architecture decisions in their projects.

Acceptance Criteria:
✅ RAG vs Full Context comparison with metrics
✅ Visual representation of efficiency gains
✅ Documentation of when to use each approach
```

#### Story 4: As a Developer
```
As an LLM application developer,
I want to understand context engineering strategies,
So that I can optimize my production prompt engineering.

Acceptance Criteria:
✅ 3+ strategies demonstrated
✅ Performance comparison over time
✅ Best practices documented
```

### 3.2 Use Cases

#### UC1: Run Single Experiment
**Actor**: Student
**Preconditions**: Python + Ollama installed
**Main Flow**:
1. Student clones repository
2. Student runs `python -m src.experiments.exp1_needle_haystack`
3. System generates data, runs trials, produces results
4. Student views `results/exp1/accuracy_by_position.png`

**Postconditions**: Results saved, student understands position effects

#### UC2: Analyze All Experiments
**Actor**: Researcher
**Preconditions**: All experiments complete
**Main Flow**:
1. Researcher opens Jupyter notebook
2. Researcher loads results JSON files
3. Researcher performs custom statistical analysis
4. Researcher exports findings

**Postconditions**: Custom analysis complete, findings documented

#### UC3: Reproduce for Different Model
**Actor**: Advanced User
**Preconditions**: Alternative Ollama model installed
**Main Flow**:
1. User modifies `config.py` (MODEL_NAME = "llama3.2")
2. User runs all experiments
3. System produces comparison data
4. User analyzes model differences

**Postconditions**: Cross-model comparison available

---

## 4. Functional Requirements

### 4.1 Core Features

#### F1: Experiment 1 - Needle in Haystack

**Priority**: P0 (Critical)
**Description**: Test position-based information retrieval accuracy

**Requirements:**
- **F1.1**: Generate 5 documents with hidden information
- **F1.2**: Test 3 positions (start, middle, end)
- **F1.3**: Run 10 iterations per position for statistical validity
- **F1.4**: Measure exact, partial, and semantic accuracy
- **F1.5**: Visualize accuracy by position as bar chart
- **F1.6**: Save results as `results/exp1/results.json`

**Acceptance Criteria:**
- ✅ Total 30 LLM calls (3 positions × 10 iterations)
- ✅ Accuracy range 0.0-1.0 for each metric
- ✅ Chart shows clear trend
- ✅ JSON contains all trial data

#### F2: Experiment 2 - Context Size Impact

**Priority**: P0 (Critical)
**Description**: Measure performance degradation as context grows

**Requirements:**
- **F2.1**: Test 5 document counts: [2, 5, 10, 20, 50]
- **F2.2**: Keep document length constant (200 words)
- **F2.3**: Run 5 iterations per size
- **F2.4**: Measure accuracy, latency, token usage
- **F2.5**: Generate 3-panel visualization (accuracy/latency/tokens vs size)
- **F2.6**: Calculate degradation rate

**Acceptance Criteria:**
- ✅ Total 25 LLM calls (5 sizes × 5 iterations)
- ✅ All metrics recorded
- ✅ 50-document test completes (max context stress test)
- ✅ Chart shows degradation trends

#### F3: Experiment 3 - RAG Effectiveness

**Priority**: P0 (Critical)
**Description**: Compare RAG vs full-context retrieval

**Requirements:**
- **F3.1**: Generate 20-document Hebrew corpus
- **F3.2**: Index corpus in ChromaDB with embeddings
- **F3.3**: Execute same query with RAG (top-K=3) and full context
- **F3.4**: Measure accuracy and token efficiency
- **F3.5**: Visualize comparison as grouped bar chart
- **F3.6**: Calculate efficiency ratio (tokens saved vs accuracy lost)

**Acceptance Criteria:**
- ✅ 2 approaches tested
- ✅ RAG retrieves correct documents
- ✅ Token usage < 20% for RAG vs full
- ✅ Accuracy delta documented

#### F4: Experiment 4 - Context Engineering Strategies

**Priority**: P0 (Critical)
**Description**: Evaluate different context management approaches

**Requirements:**
- **F4.1**: Implement 3 strategies: SELECT, COMPRESS, WRITE
- **F4.2**: Simulate 10 sequential actions per strategy
- **F4.3**: Track accuracy degradation over time
- **F4.4**: Visualize strategy comparison as line chart
- **F4.5**: Identify best strategy for different scenarios

**Acceptance Criteria:**
- ✅ 3 strategies implemented
- ✅ 30 trials total (3 strategies × 10 actions)
- ✅ Clear winner identified or trade-offs documented
- ✅ Best practices recommended

### 4.2 Supporting Features

#### F5: Data Generation

**Priority**: P0 (Critical)
**Requirements:**
- **F5.1**: Generate synthetic English documents
- **F5.2**: Generate Hebrew corpus for RAG
- **F5.3**: Support reproducible random seeds
- **F5.4**: Validate document lengths
- **F5.5**: Insert "needles" at specified positions

#### F6: LLM Integration

**Priority**: P0 (Critical)
**Requirements:**
- **F6.1**: Abstract Ollama API calls
- **F6.2**: Handle retries on failure
- **F6.3**: Track token usage accurately
- **F6.4**: Measure latency per call
- **F6.5**: Support multiple models

#### F7: Evaluation

**Priority**: P0 (Critical)
**Requirements:**
- **F7.1**: Exact match scoring
- **F7.2**: Partial overlap scoring
- **F7.3**: Semantic similarity via embeddings
- **F7.4**: Keyword matching
- **F7.5**: Configurable thresholds

#### F8: Visualization

**Priority**: P1 (High)
**Requirements:**
- **F8.1**: Generate high-resolution PNG charts
- **F8.2**: Support bar, line, and grouped bar charts
- **F8.3**: Include proper labels, titles, legends
- **F8.4**: Use color-blind friendly palettes
- **F8.5**: Export publication-quality figures

---

## 5. Non-Functional Requirements

### 5.1 Performance Requirements

| ID | Requirement | Target | Measurement Method |
|----|-------------|--------|-------------------|
| **NFR-P1** | Total experiment runtime | < 5 minutes | Wall-clock time |
| **NFR-P2** | Single LLM call latency | < 3 seconds | Per-request timing |
| **NFR-P3** | Memory usage | < 6 GB | Peak RAM during Exp2 |
| **NFR-P4** | Disk space | < 100 MB | Results folder size |

### 5.2 Scalability Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| **NFR-S1** | Maximum context size | 16,000 tokens (50 docs) |
| **NFR-S2** | Maximum corpus size | 100 documents (RAG) |
| **NFR-S3** | Concurrent experiments | 1 (sequential by default) |
| **NFR-S4** | Future multiprocessing | Ready for parallel runs |

### 5.3 Reliability Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| **NFR-R1** | Ollama connection retry | 3 attempts with backoff |
| **NFR-R2** | Graceful failure handling | No crashes, log errors |
| **NFR-R3** | Data integrity | JSON validation on save |
| **NFR-R4** | Reproducibility | Same seed = same results |

### 5.4 Usability Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| **NFR-U1** | Setup time (experienced user) | < 10 minutes |
| **NFR-U2** | Setup time (novice user) | < 30 minutes |
| **NFR-U3** | README comprehension | 100% clarity |
| **NFR-U4** | Error messages | Actionable and clear |

### 5.5 Maintainability Requirements

| ID | Requirement | Compliance |
|----|-------------|------------|
| **NFR-M1** | Code documentation | Docstrings on all public functions |
| **NFR-M2** | Architecture docs | C4 diagrams + ADRs |
| **NFR-M3** | Modular design | Separate experiments, shared utils |
| **NFR-M4** | Dependency management | pyproject.toml + requirements.txt |

### 5.6 Portability Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| **NFR-PO1** | Operating Systems | macOS, Linux, Windows |
| **NFR-PO2** | Python versions | 3.10, 3.11, 3.12 |
| **NFR-PO3** | No cloud dependencies | 100% local execution |
| **NFR-PO4** | Offline operation | After initial model download |

### 5.7 Security Requirements

| ID | Requirement | Compliance |
|----|-------------|------------|
| **NFR-SE1** | No API keys required | Ollama is local |
| **NFR-SE2** | No PII in data | Synthetic data only |
| **NFR-SE3** | Input validation | All building blocks validate |
| **NFR-SE4** | Dependency scanning | No known vulnerabilities |

---

## 6. System Constraints

### 6.1 Technical Constraints

| Constraint | Description | Mitigation |
|------------|-------------|------------|
| **C1: Ollama Required** | Cannot run without Ollama installed | Clear install instructions |
| **C2: Model Size** | llama2:7B requires ~4 GB RAM | Document minimum specs |
| **C3: Context Window** | llama2 limited to 4096 tokens | Document in limitations |
| **C4: Hebrew Support** | llama2 has weak Hebrew capability | Note in results analysis |

### 6.2 Organizational Constraints

| Constraint | Impact |
|------------|--------|
| **C5: Timeline** | Must complete by course deadline |
| **C6: Solo Developer** | Limited bandwidth for features |
| **C7: Academic Context** | Prioritize documentation over optimization |
| **C8: No Budget** | Free tools only (Ollama, open models) |

### 6.3 Regulatory Constraints

| Constraint | Compliance |
|------------|------------|
| **C9: Academic Integrity** | All code self-written, LLM-assisted documented |
| **C10: Licensing** | All dependencies MIT/Apache compatible |
| **C11: Data Privacy** | No external data collection |

---

## 7. Dependencies and Assumptions

### 7.1 External Dependencies

| Dependency | Version | Purpose | Fallback |
|------------|---------|---------|----------|
| Ollama | Latest | LLM inference | None (required) |
| ChromaDB | 0.4.22+ | Vector storage | In-memory (no server) |
| sentence-transformers | 2.2.2+ | Embeddings | Required |
| Matplotlib | 3.7+ | Visualization | Manual charting |
| LangChain | 0.3.15+ | Orchestration | Direct API calls |

### 7.2 Assumptions

**Technical Assumptions:**
1. User has admin rights to install software
2. Internet available for initial setup
3. Minimum 8 GB RAM available
4. Disk has 10+ GB free space

**User Assumptions:**
1. Basic Python knowledge
2. Familiarity with command line
3. Understanding of LLM concepts
4. Can read technical documentation

**Experimental Assumptions:**
1. Ollama provides deterministic results (temp=0)
2. Embeddings are consistent across runs
3. Synthetic data is representative enough
4. Hebrew corpus quality is adequate

---

## 8. Timeline and Milestones

### 8.1 Development Timeline

| Phase | Duration | Deliverables | Status |
|-------|----------|--------------|--------|
| **Phase 1: Setup** | Week 1 | Environment, dependencies | ✅ Complete |
| **Phase 2: Core** | Week 2 | LLM interface, data gen, evaluator | ✅ Complete |
| **Phase 3: Experiments** | Week 3 | Exp 1-4 implementation | ✅ Complete |
| **Phase 4: Analysis** | Week 4 | Notebooks, visualizations | ✅ Complete |
| **Phase 5: Documentation** | Week 5 | README, RPD, RESULTS | ✅ Complete |
| **Phase 6: Compliance** | Week 6 | PRD, Architecture, API docs, Tests | ✅ Complete |

### 8.2 Milestones

| Milestone | Criteria | Date Achieved |
|-----------|----------|---------------|
| **M1: Ollama Working** | Can call llama2 successfully | Dec 1, 2025 |
| **M2: First Experiment** | Exp1 produces results | Dec 2, 2025 |
| **M3: All Experiments** | Exp1-4 all working | Dec 3, 2025 |
| **M4: Bug Fixes** | 50-doc test works, JSON saves | Dec 4, 2025 |
| **M5: Documentation** | README, RESULTS complete | Dec 5, 2025 |
| **M6: Full Compliance** | Meets all guideline requirements | Dec 8, 2025 |

---

## 9. Success Criteria

### 9.1 Launch Readiness Criteria

**Must-Have (P0):**
- ✅ All 4 experiments execute without errors
- ✅ All visualizations generate correctly
- ✅ README provides complete setup instructions
- ✅ Results analysis is comprehensive
- ✅ Code is on GitHub with all commits

**Should-Have (P1):**
- ✅ Jupyter notebooks for deeper analysis
- ✅ Statistical significance testing
- ✅ Cross-experiment comparisons
- ✅ Full guideline compliance documentation

**Nice-to-Have (P2):**
- ✅ Unit tests with 70%+ coverage
- ✅ RAG parameter sensitivity analysis
- ⬜ CI/CD pipeline
- ⬜ Docker containerization
- ⬜ Multi-model comparison

### 9.2 Quality Metrics

| Metric | Target | Actual |
|--------|--------|--------|
| **Code Quality** | No crashes | ✅ 0 crashes |
| **Documentation** | All sections complete | ✅ 100% |
| **Reproducibility** | Same results on re-run | ✅ Yes |
| **Performance** | < 5 min runtime | ✅ 3m 44s |
| **Accuracy** | Metrics make sense | ✅ Validated |

---

## 10. Out of Scope

### 10.1 Explicitly Not Included

**Features:**
- ❌ Cloud LLM API support (OpenAI, Anthropic)
- ❌ Real-time streaming responses
- ❌ Web-based UI/dashboard
- ❌ Multi-user concurrent access
- ❌ Production deployment tooling
- ❌ Automated hyperparameter tuning
- ❌ Cross-model benchmarking suite
- ❌ Fine-tuning capabilities

**Experiments:**
- ❌ Experiment 5+: Not in assignment
- ❌ Multi-modal contexts (images, audio)
- ❌ Code generation tasks
- ❌ Adversarial prompt testing

**Infrastructure:**
- ❌ Cloud deployment
- ❌ Kubernetes orchestration
- ❌ Monitoring/alerting
- ❌ Auto-scaling

### 10.2 Future Considerations

**Potential Extensions:**
- Support for other local LLM frameworks (llama.cpp, vLLM)
- Batch processing for large-scale experiments
- A/B testing framework
- Cost tracking for cloud deployment scenarios
- Integration with W&B or MLflow

---

## 11. Risks and Mitigation

### 11.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **R1: Ollama fails** | Low | High | Retry logic, clear error messages |
| **R2: Model unavailable** | Low | High | Document which models are tested |
| **R3: Memory overflow** | Medium | Medium | Document minimum specs, optimize data |
| **R4: Slow performance** | Low | Low | Already optimized, acceptable runtime |

### 11.2 Schedule Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **R5: Scope creep** | Medium | Medium | Strict adherence to PRD |
| **R6: Bugs discovered late** | Low | Low | Already tested thoroughly |
| **R7: Guideline changes** | Very Low | Medium | Flexible architecture |

### 11.3 Quality Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **R8: Low accuracy** | High | Low | Expected (small model), document clearly |
| **R9: Hebrew poor** | High | Low | Expected limitation, RAG still demo'd |
| **R10: Reproducibility** | Low | High | Fixed seeds, version pinning |

---

## 12. Appendices

### Appendix A: Glossary

- **Context Window**: Maximum input tokens an LLM can process
- **Needle in Haystack**: Test where target info is hidden in large context
- **RAG**: Retrieval-Augmented Generation - retrieve relevant docs before generating
- **Ollama**: Local LLM inference platform
- **ChromaDB**: Vector database for embeddings
- **Token**: Unit of text (word/subword) processed by LLM

### Appendix B: References

1. "Lost in the Middle" paper (Liu et al., 2023)
2. LangChain documentation
3. Ollama documentation
4. Assignment PDF: `context-windows-lab.pdf`

### Appendix C: Change Log

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0.0 | Dec 6, 2025 | Initial PRD creation | Tal Barda |
| 1.1.0 | Dec 8, 2025 | Updated to reflect project completion: Phase 6 complete, M6 achieved, test coverage 70%+, sensitivity analysis added | Tal Barda |

---

**Document Approval:**

| Role | Name | Date | Signature |
|------|------|------|-----------|
| **Product Owner** | Tal Barda | Dec 6, 2025 | ✅ Approved |
| **Technical Lead** | Tal Barda | Dec 6, 2025 | ✅ Approved |
| **Instructor** | Dr. Yoram Segal | Pending | Pending |

---

**End of PRD**
