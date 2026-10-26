# Dijkstra Algorithm

## Intuition
Maintain the best-known distance to each node and repeatedly finalize the closest unvisited node.

## Core rule
Relaxation: if `dist[u] + w(u,v) < dist[v]`, update `dist[v]` and predecessor.

## Theory from First Principles
### 1) Goal
Find shortest path distances from a source node to all reachable nodes in a weighted graph with non-negative edge costs.

### 2) Approach
Maintain tentative best distances in a min-priority queue. Repeatedly pop the unvisited node with smallest distance; that distance is final. Relax each outgoing edge to possibly improve neighbor distances and predecessor links.

### 3) Result / Output
The function returns:
- `bestWeights`: final shortest-known distance per node
- `path`: reconstructed source-to-target path (empty if unreachable)

### 4) Formulas
- Initialization: `d(s)=0`, `d(v)=\infty` for `v\neq s`
- Relaxation: `d(v)\leftarrow\min(d(v), d(u)+w(u,v))`
- Predecessor update when relaxation improves distance: `\pi(v)\leftarrow u`

### 5) Intuition (small example)
If node A is currently closest unsettled node, any alternative route to A through farther unsettled nodes cannot beat its current cost when all edges are non-negative. So we can safely lock A and expand outward.


## Files
- `main.py`: reusable `shortest_path` function + example graph run.

## From-scratch vs libraries
- From-scratch graph algorithm.
- Uses standard-library `heapq` for priority queue.

## Dependencies
- Python standard library only.

## Usage
```bash
python dijkstra_algorithm/main.py
```

## Dataset
- None (embedded graph example).

## Limitations / caveats
- Designed for non-negative edge weights.
- Unreachable targets return an empty path.
