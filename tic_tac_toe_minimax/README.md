# Tic-Tac-Toe Minimax

## Intuition
Enumerate game outcomes recursively and choose actions maximizing guaranteed result.

## Core rule
Max player picks highest value branch; Min player picks lowest. Tie-breakers include depth and subtree-average score.

## Theory from First Principles
### 1) Goal
Choose moves that guarantee the best possible game outcome against an optimal opponent.

### 2) Approach
The game tree is explored recursively. Terminal boards get utility values (`+1` X win, `-1` O win, `0` draw). `maxValue` assumes X picks an action maximizing utility; `minValue` assumes O minimizes it. This implementation also breaks ties by preferring quicker wins, slower losses, and then subtree average.

### 3) Result / Output
For a given board and role, the algorithm returns or selects the move with optimal minimax value under perfect play assumptions.

### 4) Formulas
- Utility: `U(s)\in\{-1,0,1\}` for terminal state `s`
- Recurrence:
  - `V_{max}(s)=\max_{a\in A(s)} V_{min}(T(s,a))`
  - `V_{min}(s)=\min_{a\in A(s)} V_{max}(T(s,a))`
where `T(s,a)` is transition after action `a`.

### 5) Intuition (small example)
If one move guarantees at least a draw while another allows a forced loss, minimax picks the safe move because it assumes the opponent answers optimally at every step.


## Files
- `main.py`: agent logic (`winner`, `terminal`, minimax recursion) and CLI game loop.

## From-scratch vs libraries
- Fully from-scratch game and minimax logic.

## Dependencies
- Python standard library only.

## Usage
```bash
python tic_tac_toe_minimax/main.py
```

## Dataset
- None.

## Limitations / caveats
- Interactive script expects terminal input.
- Depth-first exhaustive search is fine for 3x3 but not scalable to larger boards.
