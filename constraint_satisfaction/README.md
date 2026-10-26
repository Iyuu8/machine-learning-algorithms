# Constraint Satisfaction

## Intuition
Assign values to variables while respecting pairwise constraints, using backtracking search with MRV and degree heuristic.

## Core rule
For each variable assignment `(var=value)`, all neighbor variables must avoid the same value.

## Theory from First Principles
### 1) Goal
Assign values to variables so all pairwise constraints are satisfied (a feasibility problem, not optimization).

### 2) Approach
The solver uses depth-first backtracking. Variable choice applies MRV (fewest legal values first), with degree heuristic as tie-break (most constraints on remaining variables). Candidate values are filtered by consistency against current partial assignment.

### 3) Result / Output
Returns a complete assignment (set of `(variable, value)` pairs) if solvable; otherwise `None`.

### 4) Formulas
- Constraint form: for edge `(u,v)`, enforce `x_u \neq x_v`
- MRV choice: `u^*=\arg\min_{u\in U}|D_u^{valid}|`
- Degree tie-break: choose `u` with largest number of unassigned neighbors

Variables: `U` unassigned variables, `D_u^{valid}` currently consistent domain values for variable `u`.

### 5) Intuition (small example)
In exam scheduling, assigning hardest or most-constrained exams first reduces dead ends later. Backtracking undoes only conflicting choices, making search practical compared to brute-force enumeration.


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
