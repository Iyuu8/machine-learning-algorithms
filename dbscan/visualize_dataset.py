from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATASET_PATH = Path(__file__).parent / "dataset" / "dbscan_toy_3d.csv"


def main() -> None:
    data = pd.read_csv(DATASET_PATH)

    required_columns = {"x", "y", "z"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")

    figure = plt.figure(figsize=(10, 8))
    axes = figure.add_subplot(projection="3d")
    axes.scatter(
        data["x"],
        data["y"],
        data["z"],
        c=data["z"],
        cmap="viridis",
        s=24,
        alpha=0.8,
        edgecolors="none",
    )
    axes.set_title("DBSCAN Toy Dataset")
    axes.set_xlabel("X")
    axes.set_ylabel("Y")
    axes.set_zlabel("Z")
    figure.colorbar(axes.collections[0], ax=axes, pad=0.1, label="Z value")
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()