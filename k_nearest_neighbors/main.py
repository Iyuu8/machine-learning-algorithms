import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from model import Model

df = pd.read_csv("./dataset/Iris.csv")
dfClean = df.dropna()




dfTrain, dfTest = train_test_split(dfClean, test_size=0.5, random_state=42)

trainFeatures = dfTrain[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]].to_numpy()
trainFeaturesScaled  = ( trainFeatures - np.mean(trainFeatures, axis=0)) / np.std(trainFeatures, axis=0)
classes = dfTrain["Species"].unique()

encodeClass = {name: i for i, name in enumerate(classes)}
trainTarget = dfTrain["Species"].map(encodeClass).to_numpy()

testFeatures = dfTest[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]].to_numpy()
testFeaturesScaled = ( testFeatures - np.mean(trainFeatures, axis=0)) / np.std(trainFeatures, axis=0)
testTarget = dfTest["Species"].map(encodeClass).to_numpy() # necessary to be a simple array

knnModel = Model(3,trainFeaturesScaled,trainTarget)
print(knnModel.accuracy(testFeaturesScaled,testTarget))




