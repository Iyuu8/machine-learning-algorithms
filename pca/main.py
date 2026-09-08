import numpy as np
import pandas as pd
from model import Model
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv("./dataset/Country-data.csv")
dfClean = df.dropna()

features = [ "child_mort", "exports", "health", "imports", "income", "inflation", "life_expec", "total_fer", "gdpp"]

dataset = dfClean[features].to_numpy()
dataset = (dataset - np.mean(dataset,axis=0)) / np.std(dataset, axis=0)

for k in range(1,dataset.shape[1]+1):
    model = Model(k, dataset)
    reducedData, W,  eigenvaluesAll, eigenvalues= model.compress()

    approxData, error = model.deCompress(reducedData)
    percentage = eigenvalues.sum() / eigenvaluesAll.sum()
    print(k, error, percentage)



# data visualization code by gemini
# the dataset used here is the same as the one used in k-means, the visualization bellow serves the check whether the value of k=3 the k-means algo study results in is actually valid

model_pca = Model(3, dataset)
reducedData, _, _, _ = model_pca.compress()


kmeans = KMeans(n_clusters=3, random_state=42)
assigned_classes = kmeans.fit_predict(reducedData)


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')


scatter = ax.scatter(reducedData[:, 0], reducedData[:, 1], reducedData[:, 2], 
           c=assigned_classes, cmap='viridis', alpha=0.8, edgecolor='w', s=40)

ax.set_title("Dataset Projected onto 3 Principal Components (Clustered)", pad=15)
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")


legend = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend)

plt.tight_layout()
plt.show()