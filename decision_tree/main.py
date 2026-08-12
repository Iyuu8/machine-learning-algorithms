import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from model import DecisionTreeClassifier

df = pd.read_csv("./dataset/Iris.csv")
dfClean = df.dropna()

dfTrain, dfTest = train_test_split(dfClean, test_size=0.2, random_state=42)

trainFeatures = dfTrain[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].to_numpy()
trainFeaturesScaled = (trainFeatures - np.mean(trainFeatures, axis=0)) / np.std(trainFeatures, axis=0)
trainTarget = dfTrain[["Species"]].to_numpy().ravel()

testFeatures = dfTest[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].to_numpy()
testFeaturesScaled = (testFeatures - np.mean(trainFeatures, axis=0)) / np.std(trainFeatures, axis=0)  
testTarget = dfTest[["Species"]].to_numpy().ravel()

model = DecisionTreeClassifier()
model.buildTree(trainFeaturesScaled, np.asarray(trainTarget).ravel())
print((model.predict(testFeaturesScaled) == np.asarray(testTarget).ravel()).sum() / testTarget.shape[0])

