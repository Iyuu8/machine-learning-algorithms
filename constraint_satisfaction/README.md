# Constraint Satisfaction

## Intuition
Assign values to variables while respecting pairwise constraints, using backtracking search with MRV and degree heuristic.

## Core rule
For each variable assignment `(var=value)`, all neighbor variables must avoid the same value.

## Files
- `constraints.py`: CSP definition and solver.

## From-scratch vs libraries
- From-scratch search/heuristics implementation.

## Dependencies
- Python standard library only.

## Usage
```bash
python constraint_satisfaction/constraints.py
```

## Dataset
- None.

## Limitations / caveats
- Current example creates a dense 10x10 constraint graph and can be expensive.
- Intended for educational demonstration rather than large-scale CSP benchmarks.
