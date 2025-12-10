# ADR-003: Configuration-Driven Design

**Status**: Accepted
**Date**: November 2025
**Impact**: Medium - Affects maintainability and extensibility

---

## Decision

**All experimental parameters in `src/config.py`. No magic numbers in code.**

---

## Rationale

1. **Single Source of Truth**: All parameters in one file
2. **Easy Experimentation**: Change config without code modification  
3. **Reproducibility**: Config file documents exact parameters used
4. **Extensibility**: New experiments add config sections
5. **Testing**: Can override config for different test scenarios

---

## Implementation

```python
# config.py structure
EXP1_CONFIG = {...}
EXP2_CONFIG = {...}
EXP3_CONFIG = {...}
EXP4_CONFIG = {...}
```

All experiments import and use these configs exclusively.

---

## Consequences

✅ No hardcoded values in experiments
✅ Easy to modify parameters
✅ Clear documentation of settings
⚠️ Must maintain config as single source

---

**Last Updated**: December 10, 2025
