# Dijkstra Algorithm

## Intuition
Maintain the best-known distance to each node and repeatedly finalize the closest unvisited node.

## Core rule
Relaxation: if `dist[u] + w(u,v) < dist[v]`, update `dist[v]` and predecessor.

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
