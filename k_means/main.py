import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import random
from model import Model

df = pd.read_csv('./dataset/Country-data.csv')
dfClean = df.dropna()
features = [ "child_mort", "exports", "health", "imports", "income", "inflation", "life_expec", "total_fer", "gdpp"]

dataset = dfClean[features].to_numpy()
dataset = (dataset - np.mean(dataset, axis=0)) / np.std(dataset, axis=0)

silhouetteScores = []
costScores = []
kRange = range(1,15)
for k in kRange:
    model = Model(k,features=dataset)
    finalCost = model.cluster(20)
    s = model.silhouetteScore()
    silhouetteScores.append(s)
    costScores.append(finalCost)

# the visulization code beloow is by gemini
fig, ax = plt.subplots(1, 2, figsize=(14, 5))


ax[0].plot(kRange, costScores, marker='o', color='b', linewidth=2)
ax[0].set_title("Elbow Method (WCSS Cost)")
ax[0].set_xlabel("K")
ax[0].set_ylabel("Total Cost")
ax[0].grid(True)


ax[1].plot(kRange, silhouetteScores, marker='s', color='orange', linewidth=2)
ax[1].set_title("Silhouette Scores")
ax[1].set_xlabel("K")
ax[1].set_ylabel("Score")
ax[1].grid(True)

plt.tight_layout()
plt.show()



