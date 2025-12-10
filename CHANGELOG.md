# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] - 2025-12-10

### Added - Academic Quality Upgrade
- `CRITERIA_CHECKLIST.md`: Objective, verifiable requirements checklist
- `SELF_ASSESSMENT_METHOD.md`: Methodology for objective grading
- `COST_ANALYSIS.md`: Standalone comprehensive cost and efficiency analysis
- `EXPERIMENT_TEMPLATE.md`: Reusable template for future experiments
- `CLEAN_CODE_GUIDELINES.md`: Project coding standards and conventions
- ADRs (Architectural Decision Records) in `docs/architecture/decisions/`:
  - ADR-001: Local LLM Inference with Ollama
  - ADR-002: Synthetic Data Generation
  - ADR-003: Configuration-Driven Design
- Code quality configurations:
  - `.flake8`: Python linting configuration
  - `mypy.ini`: Static type checking configuration
- Enhanced developer documentation in README.md

### Changed
- Froze Experiment 1 configuration with comprehensive documentation (13 docs, 105 words, 4 herrings)
- Updated `docs/RESULTS.md` with comprehensive U-shape analysis and literature connections
- Improved all docstrings to follow NumPy style guide
- Enhanced error handling throughout codebase

### Fixed
- Experiment 1 now achieves genuine U-shape (Start=1.000, Middle=0.912, End=1.000)
- Answer extraction regex for precise credential extraction
- Data generator reproducibility with dedicated Random instances
- Type conversions in evaluator for JSON serialization

---

## [1.5.0] - 2025-12-06

### Added - Testing Infrastructure
- Comprehensive pytest test suite with 250+ test cases
- Test coverage reports (91% on core modules)
- Mocked dependencies (Ollama, embeddings, ChromaDB)
- `conftest.py` with shared fixtures
- `pytest.ini` with coverage configuration

### Changed
- All tests now pass without external dependencies
- Deterministic testing with fixed seeds

---

## [1.0.0] - 2025-12-05

### Added - Initial Release
- Four complete experiments:
  - Experiment 1: Needle in Haystack (Lost in the Middle)
  - Experiment 2: Context Window Size Impact
  - Experiment 3: RAG vs Full Context
  - Experiment 4: Context Engineering Strategies
- Comprehensive documentation:
  - `docs/PRD.md`: Product Requirements Document
  - `docs/ARCHITECTURE.md`: System architecture with C4 diagrams
  - `docs/API.md`: Complete API reference
  - `docs/PROMPTS.md`: LLM-assisted development log
  - `docs/RESULTS.md`: Experimental results and analysis
- Configuration management:
  - `src/config.py`: All experimental parameters
  - `.env.example`: Environment variable template
- Modern Python packaging:
  - `pyproject.toml`: PEP 621 compliant configuration
  - Dependency management with version pinning
- Data generation:
  - Synthetic data generators for all experiments
  - Reproducible with seed=42
- Evaluation framework:
  - Accuracy metrics (exact, partial, semantic)
  - Multi-metric evaluation
- Visualization:
  - Accuracy by position (Exp 1)
  - Context size impact (Exp 2)
  - RAG comparison (Exp 3)
  - Strategy comparison (Exp 4)
  - RAG sensitivity analysis heatmap
- Git practices:
  - Clean commit history
  - Meaningful commit messages
  - Co-authored commits with Claude

---

## [0.5.0] - 2025-11-28

### Added - Project Setup
- Initial project structure
- Basic LLM interface (Ollama)
- Data generator skeleton
- Configuration file
- README with installation instructions

---

## Version History Summary

| Version | Date | Description | Key Changes |
|---------|------|-------------|-------------|
| **2.0.0** | 2025-12-10 | Academic Quality Upgrade | +ADRs, +checklists, +guidelines, frozen Exp1 config |
| **1.5.0** | 2025-12-06 | Testing Infrastructure | +250 tests, 91% coverage, mocked dependencies |
| **1.0.0** | 2025-12-05 | Initial Release | 4 experiments, full documentation, visualizations |
| **0.5.0** | 2025-11-28 | Project Setup | Basic structure and interfaces |

---

## Upgrade Path

### From 1.5.0 to 2.0.0
1. Pull latest changes: `git pull origin main`
2. Review new documentation files (CRITERIA_CHECKLIST.md, etc.)
3. Note: Experiment 1 configuration is now frozen - do not modify
4. Check ADRs for architectural rationale
5. Run quality checks: `flake8 src/` and `mypy src/`

### From 1.0.0 to 1.5.0
1. Pull latest changes
2. Install test dependencies: `pip install pytest pytest-cov`
3. Run tests: `pytest --cov=src`

---

## Breaking Changes

### Version 2.0.0
- None (fully backward compatible)
- Experiment 1 configuration frozen (do not modify `EXP1_CONFIG`)

### Version 1.5.0
- None (fully backward compatible)

---

## Deprecations

- None currently

---

## Future Roadmap

### Planned for 2.1.0
- [ ] Add Experiment 5: Multi-modal context handling
- [ ] GPU acceleration support
- [ ] Docker containerization
- [ ] CI/CD pipeline (GitHub Actions)

### Planned for 3.0.0
- [ ] Support for multiple LLM backends (OpenAI, Anthropic, Hugging Face)
- [ ] Web UI for experiment management
- [ ] Real-time monitoring dashboard
- [ ] Experiment result comparison tool

---

## Maintenance

This changelog is maintained by the development team.
Updates follow the [Keep a Changelog](https://keepachangelog.com/) format.

**Last Updated**: December 10, 2025
