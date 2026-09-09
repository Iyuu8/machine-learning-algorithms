import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import Model

df = pd.read_csv('./dataset/dbscan_toy_3d.csv')
features = ['x','y','z']
dataset = df[features].to_numpy()


m = 4
k = m
epsilon = 2.1

model = Model(epsilon, m, dataset)
clusters = model.createClusters()
print(clusters)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2], c=clusters, cmap='tab10', s=30)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D DBSCAN Clustering')
plt.colorbar(scatter)
plt.show()

# justifying the choice of epsilon
distances = model.distances
sortedDistances = np.sort(distances, axis=1)
KthNeaestDistances = np.sort(sortedDistances[:, k])

plt.figure(figsize=(10, 6))
plt.plot(
	np.arange(1, len(KthNeaestDistances) + 1),
	KthNeaestDistances,
	linewidth=2,
)
plt.title(f'Sorted {k}th Nearest Neighbor Distances')
plt.xlabel('Points sorted by distance')
plt.ylabel('Distance')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()




