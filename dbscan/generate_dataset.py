import numpy as np
import pandas as pd

np.random.seed(42)

# Generate 4 well-separated dense blobs
centers = np.array([
    [15, 15, 15],
    [-15, 15, -15],
    [15, -15, -15],
    [-15, -15, 15]
])

points = []
for c in centers:
    # Dense blobs: standard deviation = 1.0
    blob = np.random.normal(loc=c, scale=1.0, size=(150, 3))
    points.append(blob)

# Add scattered points far from the blobs
scattered = np.random.uniform(low=-30, high=30, size=(20, 3))
points.append(scattered)

all_points = np.vstack(points)

# Create a DataFrame and save as CSV
df = pd.DataFrame(all_points, columns=['x', 'y', 'z'])
df.to_csv('dataset/dbscan_toy_3d.csv', index=False)
print("Dataset generated successfully at dataset/dbscan_toy_3d.csv")
