# Hill Climbing Hospital Configuration

## Intuition
Place hospitals on a grid to reduce summed Manhattan distance from houses; use local improvements and random restarts.

## Core objective
Minimize: `Σ_houses min_hospitals ManhattanDistance(house, hospital)`.

## Theory from First Principles
### 1) Goal
Place a fixed number of hospitals on a grid so houses have low travel distance to their nearest hospital.

### 2) Approach
A state is a set of hospital coordinates. Neighborhood moves one hospital one grid step (up/down/left/right) into empty cells. Hill climbing repeatedly applies cost-improving moves until local optimum; random restarts sample new initial placements and keep the best final state.

### 3) Result / Output
Outputs a hospital placement and its total assignment cost. Optional JSON or image export records exact coordinates and cost.

### 4) Formulas
- Manhattan distance: `d((r_1,c_1),(r_2,c_2)) = |r_1-r_2| + |c_1-c_2|`
- Objective: `J(H)=\sum_{h\in Houses}\min_{p\in H} d(h,p)`

Variables: `H` is current set of hospital positions. Hill climbing seeks a local minimizer of `J`.

### 5) Intuition (small example)
If moving one hospital one cell left reduces distance for several nearby houses more than it harms others, total cost drops, so hill climbing accepts the move. Restarts help escape poor local placements.


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
