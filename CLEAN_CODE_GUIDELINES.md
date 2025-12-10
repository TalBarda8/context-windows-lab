# Clean Code Guidelines

**Project**: Context Windows Lab
**Purpose**: Maintain consistent, high-quality code across the project

---

## Code Style

### Python Version
- **Target**: Python 3.9+
- **Compatibility**: Ensure code works on 3.9-3.12

### Formatter
- **Tool**: Black (default settings)
- **Line Length**: 88 characters (Black default)
- **Command**: `black src/ tests/`

### Linter
- **Tool**: flake8
- **Config**: `.flake8`
- **Command**: `flake8 src/`

### Type Checker
- **Tool**: mypy
- **Config**: `mypy.ini`
- **Command**: `mypy src/`

---

## Naming Conventions

### Files and Modules
- Use `snake_case` for all Python files
- Example: `data_generator.py`, `llm_interface.py`

### Classes
- Use `PascalCase`
- Clear, descriptive names
- Example: `DataGenerator`, `LLMInterface`, `NeedleHaystackExperiment`

### Functions and Methods
- Use `snake_case`
- Verb-based names
- Example: `generate_data()`, `run_experiment()`, `evaluate_response()`

### Variables
- Use `snake_case`
- Descriptive names (no abbreviations unless standard)
- Example: `num_documents`, `context_window`, `llm_response`

### Constants
- Use `UPPER_SNAKE_CASE`
- Define in `config.py` or module top
- Example: `MAX_CONTEXT_WINDOW`, `DEFAULT_SEED`

### Private Methods/Variables
- Prefix with single underscore
- Example: `_reset_seeds()`, `_aggregate_results()`

---

## Documentation

### Docstrings
- **Style**: NumPy format
- **Required**: All public functions, classes, methods
- **Components**: Description, Args, Returns, Raises (if applicable)

**Example**:
```python
def evaluate_accuracy(response: str, expected: str, threshold: float = 0.8) -> Dict[str, float]:
    """
    Evaluate response accuracy against expected answer.

    Combines exact match, partial match, and semantic similarity to compute
    an overall accuracy score.

    Parameters
    ----------
    response : str
        The LLM's response text
    expected : str
        The expected correct answer
    threshold : float, optional
        Minimum similarity threshold for semantic match, by default 0.8

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - 'exact_match': 1.0 if exact match, 0.0 otherwise
        - 'partial_match': Score between 0.0-1.0
        - 'semantic_similarity': Cosine similarity score
        - 'overall_score': Weighted average of all metrics

    Examples
    --------
    >>> evaluate_accuracy("Paris", "Paris", threshold=0.8)
    {'exact_match': 1.0, 'partial_match': 1.0, 'semantic_similarity': 1.0, 'overall_score': 1.0}
    """
    # Implementation
```

### Comments
- Use for complex logic explanation
- Don't state the obvious
- Good: `# Skip processing if cache hit to save time`
- Bad: `# Set x to 5`

### Type Hints
- **Required**: All function signatures
- Use `from typing import ...` for complex types
- Example:
```python
from typing import List, Dict, Optional, Tuple

def process_documents(docs: List[str], max_length: Optional[int] = None) -> Dict[str, Any]:
    ...
```

---

## Code Organization

### Module Structure
```python
"""Module docstring."""

# Standard library imports
import os
import sys
from pathlib import Path

# Third-party imports
import numpy as np
from tqdm import tqdm

# Local imports
from config import CONFIG_NAME
from utils import helper_function

# Constants
MAX_RETRIES = 3

# Classes
class MyClass:
    ...

# Functions
def my_function():
    ...

# Main execution
if __name__ == "__main__":
    main()
```

### Class Structure
```python
class ExperimentRunner:
    """Class docstring."""

    def __init__(self, ...):
        """Initialize."""
        # Public attributes
        self.config = config

        # Private attributes
        self._cache = {}

    def public_method(self):
        """Public method."""
        pass

    def _private_method(self):
        """Private helper method."""
        pass

    @property
    def computed_property(self):
        """Property accessor."""
        return self._compute()
```

---

## Best Practices

### DRY (Don't Repeat Yourself)
- Extract repeated code into functions
- Use configuration for repeated values
- Leverage inheritance/composition

### Error Handling
```python
# Good: Specific exceptions with context
try:
    result = llm.query(prompt)
except ConnectionError as e:
    logger.error(f"Failed to connect to LLM service: {e}")
    raise RuntimeError("LLM service unavailable") from e
except TimeoutError:
    logger.warning("LLM query timed out, retrying...")
    result = llm.query(prompt, timeout=60)

# Bad: Bare except
try:
    result = llm.query(prompt)
except:
    pass
```

### Configuration Over Hardcoding
```python
# Good
MAX_TOKENS = config.get("max_tokens", 4096)
if token_count > MAX_TOKENS:
    truncate()

# Bad
if token_count > 4096:
    truncate()
```

### Explicit Over Implicit
```python
# Good
is_valid = len(response) > 0 and response.strip() != ""

# Bad
is_valid = response  # Implicit boolean conversion
```

---

## Testing

### Test Naming
```python
def test_{what}_when_{condition}_then_{expected}():
    """Test description."""
    pass

# Examples
def test_evaluate_accuracy_when_exact_match_then_returns_one():
    ...

def test_llm_query_when_timeout_then_raises_error():
    ...
```

### Test Structure (Arrange-Act-Assert)
```python
def test_feature():
    """Test feature behavior."""
    # Arrange
    generator = DataGenerator(seed=42)
    expected = "test_value"

    # Act
    result = generator.generate(...)

    # Assert
    assert result == expected
```

### Fixtures
- Use `conftest.py` for shared fixtures
- Keep fixtures focused and reusable
- Name clearly: `mock_llm`, `sample_data`, `test_config`

---

## Git Practices

### Commit Messages
```
Type: Brief summary (50 chars or less)

Detailed description of what changed and why, not how.
Wrap at 72 characters.

- Bullet points for multiple changes
- Reference issues if applicable

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>
```

**Types**: feat, fix, docs, style, refactor, test, chore

### Branching
- `main`: Production-ready code
- `feature/name`: New features
- `fix/name`: Bug fixes
- `docs/name`: Documentation updates

---

## Performance

### Efficiency
- Profile before optimizing
- Use generators for large datasets
- Cache expensive computations
- Batch API calls when possible

### Memory
- Use generators for large data: `yield` instead of `return list`
- Clear large objects when done: `del large_object`
- Use `@lru_cache` for repeated computations

---

## Security

### Sensitive Data
- Never commit API keys, passwords, secrets
- Use `.env` files (add to `.gitignore`)
- Use environment variables for secrets

### Input Validation
- Validate all external inputs
- Sanitize user-provided data
- Check file paths before operations

---

## Code Review Checklist

Before committing:
- [ ] All tests pass: `pytest`
- [ ] Code formatted: `black src/`
- [ ] Linting clean: `flake8 src/`
- [ ] Type checking passes: `mypy src/`
- [ ] Documentation updated
- [ ] No commented-out code
- [ ] No debug print statements
- [ ] No TODOs without context

---

## Quality Metrics

### Target Standards
- **Test Coverage**: > 85% on core modules
- **Cyclomatic Complexity**: < 10 per function
- **Function Length**: < 50 lines
- **File Length**: < 500 lines
- **Type Hint Coverage**: 100% on public APIs

---

## Example: Good vs Bad

### Variable Naming
```python
# Good
user_authentication_token = get_token()
max_retry_attempts = 3

# Bad
t = get_token()
x = 3
```

### Function Design
```python
# Good: Single responsibility, clear purpose
def calculate_accuracy(predicted: str, expected: str) -> float:
    """Calculate accuracy score."""
    return 1.0 if predicted == expected else 0.0

# Bad: Multiple responsibilities
def process_and_save_and_log(data, file, logger):
    # Does too many things
    pass
```

### Magic Numbers
```python
# Good
SECONDS_PER_DAY = 86400
timeout = 2 * SECONDS_PER_DAY

# Bad
timeout = 172800  # What does this number mean?
```

---

## Tools

### Recommended IDE Setup
- **VSCode**: Python extension, Black formatter, Pylance
- **PyCharm**: Professional or Community edition
- **Vim**: with ALE or coc.nvim

### Pre-commit Hooks (Optional)
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black
  - repo: https://github.com/PyCQA/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
```

---

## Learning Resources

- [PEP 8](https://pep8.org/): Python style guide
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Clean Code](https://www.amazon.com/Clean-Code-Handbook-Software-Craftsmanship/dp/0132350882): Book by Robert Martin
- [The Zen of Python](https://www.python.org/dev/peps/pep-0020/): `import this`

---

**Last Updated**: December 10, 2025
**Version**: 1.0
