# DBSCAN

## Intuition
Cluster points by density: core points have enough neighbors in an epsilon-radius neighborhood, and clusters expand through density reachability.

## Core rule
Neighborhood: `Nε(p) = {q | distance(p, q) < ε}` (self excluded in this implementation).

## Files
- `model.py`: DBSCAN clustering logic.
- `generate_dataset.py`: deterministic toy 3D dataset generator.
- `main.py`: clustering run + 3D plots.
- `visualize_dataset.py`: standalone dataset visualization.

## From-scratch vs libraries
- From-scratch DBSCAN logic.
- Pandas/Matplotlib only for data IO and visualization.

## Dependencies
- `numpy`, `pandas`, `matplotlib`

## Usage
```bash
python dbscan/generate_dataset.py
python dbscan/main.py
```

## Dataset
- Generated at `dbscan/dataset/dbscan_toy_3d.csv`.

## Limitations / caveats
- Current convention excludes self from neighbor count.
- Seed check uses `count >= m`; expansion currently uses `count > m` (preserved intentionally).
