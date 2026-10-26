# Tic-Tac-Toe Minimax

## Intuition
Enumerate game outcomes recursively and choose actions maximizing guaranteed result.

## Core rule
Max player picks highest value branch; Min player picks lowest. Tie-breakers include depth and subtree-average score.

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
