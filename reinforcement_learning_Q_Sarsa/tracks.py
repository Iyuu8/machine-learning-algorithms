"""
Track grids for the racetrack RL environment.

Legend:
    '#'  -> off track (wall / crash zone)
    '.'  -> track (drivable path)
    'S'  -> start line cell
    'F'  -> finish line cell

Each track is a list of strings (rows). Convert to a 2D array however you like,
e.g.:
    grid = [list(row) for row in OVAL]
or
    import numpy as np
    grid = np.array([list(row) for row in OVAL])

Cell (row, col) = grid[row][col], with row 0 at the top.
"""

# --- Oval ring track (18 x 30) ------------------------------------------
# A closed loop. Start line on the inner-left straight, finish line just
# after it (so a full lap around the ring reaches the finish).
OVAL = [
    "##############################",
    "##############################",
    "##########..........##########",
    "#######................#######",
    "#####....................#####",
    "####......................####",
    "###F.......................###",
    "###F......##########........##",
    "###F...################.....##",
    "#######################.....##",
    "##S.......##########........##",
    "###S.......................###",
    "####S.....................####",
    "#####....................#####",
    "#######................#######",
    "##########..........##########",
    "##############################",
    "##############################",
]

# --- L-shaped track (20 x 20) --------------------------------------------
# One sharp 90-degree turn. Start line at the bottom of the vertical leg,
# finish line at the far right of the horizontal leg.
L_SHAPE = [
    "####################",
    "####################",
    "##...............F##",
    "##...............F##",
    "##...............F##",
    "##...............F##",
    "##...............F##",
    "##.....#############",
    "##.....#############",
    "##.....#############",
    "##.....#############",
    "##.....#############",
    "##.....#############",
    "##.....#############",
    "##.....#############",
    "##.....#############",
    "##.....#############",
    "##.....#############",
    "##SSSSS#############",
    "####################",
]

# --- S-curve track (22 x 26) ---------------------------------------------
# Two consecutive turns in opposite directions (a proper "S"), more demanding
# than a single corner -- good for showing cumulative divergence between
# Q-learning and SARSA paths across back-to-back turns.
S_CURVE = [
    "##########################",
    "##########################",
    "####..................F###",
    "####..................F###",
    "####..................F###",
    "####..................F###",
    "####......################",
    "####......################",
    "####............##########",
    "####............##########",
    "####............##########",
    "####............##########",
    "##########......##########",
    "##########......##########",
    "##########......##########",
    "##########......##########",
    "##S.............##########",
    "##S.............##########",
    "##S.............##########",
    "##S.............##########",
    "##########################",
    "##########################",
]

TRACKS = {
    "oval": OVAL,
    "l_shape": L_SHAPE,
    "s_curve": S_CURVE,
}


def load_track(name):
    """Return the given track as a list of lists of single characters."""
    rows = TRACKS[name]
    return [list(row) for row in rows]


def start_cells(grid):
    """Return list of (row, col) for all 'S' cells."""
    return [(r, c) for r, row in enumerate(grid)
            for c, ch in enumerate(row) if ch == "S"]


def finish_cells(grid):
    """Return list of (row, col) for all 'F' cells."""
    return [(r, c) for r, row in enumerate(grid)
            for c, ch in enumerate(row) if ch == "F"]


def is_on_track(grid, row, col):
    """True if (row, col) is in bounds and not a wall cell."""
    if row < 0 or row >= len(grid):
        return False
    if col < 0 or col >= len(grid[0]):
        return False
    return grid[row][col] != "#"


if __name__ == "__main__":
    for name, rows in TRACKS.items():
        print(f"--- {name} ({len(rows)}x{len(rows[0])}) ---")
        for row in rows:
            print(row)
        print()