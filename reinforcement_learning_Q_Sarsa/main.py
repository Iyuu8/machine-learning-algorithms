" main by claude : for the simulation "

import os
import matplotlib
matplotlib.use("Agg")  # no display needed, just save to file
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.collections import LineCollection

from tracks import load_track
from model import QLearning, Sarsa

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def print_learning_curve(name, episodeReturns, bucket_size):
    n = len(episodeReturns)
    print(f"\n{name} learning curve (avg return per {bucket_size}-episode bucket):")
    for start in range(0, n, bucket_size):
        chunk = episodeReturns[start:start + bucket_size]
        avg = sum(chunk) / len(chunk)
        print(f"  episodes {start:>6}-{start+len(chunk)-1:<6}: avg return = {avg:8.2f}")


def draw_track(ax, track):
    rows, cols = len(track), len(track[0])
    code = {"#": 0, ".": 1, "S": 2, "F": 3}
    grid = [[code[ch] for ch in row] for row in track]
    cmap = matplotlib.colors.ListedColormap(
        ["#2b2b2b", "#e8e8e8", "#7fd17f", "#e07b7b"]
    )
    ax.imshow(grid, cmap=cmap, origin="upper", extent=(-0.5, cols - 0.5, rows - 0.5, -0.5))
    return rows, cols


def draw_agent_path(ax, states, cmap_name):
    """
    Draws the path as a sequence of line segments colored by time step
    (dark/early -> bright/late), so re-visited cells are still readable.
    Segments that follow a crash-reset are NOT drawn (they'd just be a
    fake straight line from the crash site to a random start cell) -
    instead the reset point is marked with a red X.
    """
    n = len(states)
    segments = []
    seg_order = []
    crash_points = []

    for i in range(n - 1):
        cur, nxt = states[i], states[i + 1]
        if nxt.get("crashed", False):
            crash_points.append((cur["x"], cur["y"]))
            continue
        segments.append([(cur["x"], cur["y"]), (nxt["x"], nxt["y"])])
        seg_order.append(i)

    if segments:
        lc = LineCollection(segments, cmap=cmap_name, linewidths=2.5, zorder=3)
        lc.set_array(seg_order)
        ax.add_collection(lc)

    # start marker
    ax.scatter([states[0]["x"]], [states[0]["y"]], color="black", s=70,
               marker="o", zorder=5, label="start")
    # end marker (only meaningful if it actually finished)
    ax.scatter([states[-1]["x"]], [states[-1]["y"]], color="black", s=90,
               marker="*", zorder=5, label="end")
    # crash markers
    if crash_points:
        cx, cy = zip(*crash_points)
        ax.scatter(cx, cy, color="red", marker="x", s=60, zorder=6, label="crash (reset)")

    return len(crash_points)


def plot_comparison(track, q_states, s_states, title, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax, states, cmap_name, name in [
        (axes[0], q_states, "Blues", "Q-learning"),
        (axes[1], s_states, "Oranges", "SARSA"),
    ]:
        draw_track(ax, track)
        n_crashes = draw_agent_path(ax, states, cmap_name)
        ax.set_title(f"{name}  ({len(states)-1} steps, {n_crashes} crash(es))")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    track_name = "s_curve"   # try "oval" or "l_shape" too
    track = load_track(track_name)

    common_kwargs = dict(
        track=track,
        speedMax=4,
        epsilon=0.1,            
        throttleFailProb=0.1,   
        etha=0.25,
        alpha=0.1,
        gamma=0.95,
    )

    nbEpisodesQ = 50000
    nbEpisodesSarsa = 50000   # SARSA (on-policy) typically needs more episodes to
                              # match Q-learning's greedy-policy quality, since it
                              # converges to the value of its epsilon-soft behavior
                              # policy rather than the optimal policy directly.
    nbStepsPerEpisode = 600
    maxRunSteps = 600

    print(f"Training on track: {track_name}\n")

    print("Training Q-learning...")
    q_agent = QLearning(**common_kwargs)
    q_table, q_returns = q_agent.train(nbEpisodesQ, nbStepsPerEpisode)
    print_learning_curve("Q-learning", q_returns, bucket_size=max(1, nbEpisodesQ // 10))

    print("\nTraining SARSA...")
    s_agent = Sarsa(**common_kwargs)
    s_table, s_returns = s_agent.train(nbEpisodesSarsa, nbStepsPerEpisode)
    print_learning_curve("SARSA", s_returns, bucket_size=max(1, nbEpisodesSarsa // 10))

    print("\nRunning greedy policies...")
    q_states = q_agent.run(maxRunSteps)
    s_states = s_agent.run(maxRunSteps)

    q_reached = track[q_states[-1]["y"]][q_states[-1]["x"]] == "F"
    s_reached = track[s_states[-1]["y"]][s_states[-1]["x"]] == "F"

    print(f"Q-learning: {len(q_states)} steps, reached finish: {q_reached}")
    print(f"SARSA:      {len(s_states)} steps, reached finish: {s_reached}")

    out_path = os.path.join(SCRIPT_DIR, "path_comparison.png")
    plot_comparison(
        track, q_states, s_states,
        title=f"Q-learning vs SARSA — greedy path on '{track_name}' track",
        out_path=out_path,
    )
    print(f"\nSaved comparison plot to {out_path}")


if __name__ == "__main__":
    main()