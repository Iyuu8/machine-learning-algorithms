# Hill Climbing Hospital Configuration

## Intuition
Place hospitals on a grid to reduce summed Manhattan distance from houses; use local improvements and random restarts.

## Core objective
Minimize: `Σ_houses min_hospitals ManhattanDistance(house, hospital)`.

## Files
- `hospitals.py`: map generation, hill climbing, random restart, export helpers.

## From-scratch vs libraries
- From-scratch optimization/search logic.
- Pillow used for optional image export.

## Dependencies
- `Pillow`

## Usage
```bash
python hill_climbing_hospital_configuration/hospitals.py
```

## Dataset
- None.

## Limitations / caveats
- Stochastic behavior from random initialization.
- Local search may still converge to local optima despite restarts.
