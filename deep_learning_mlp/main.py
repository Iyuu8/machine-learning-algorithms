import pandas as pd
import numpy as np
from model import NeuralNet
from sklearn.model_selection import train_test_split


df = pd.read_csv("./digit-recognizer/train.csv")
dfClean = df.dropna()
dfTrain , dfTest = train_test_split(dfClean,test_size=0.1,random_state=42)


X = dfTrain.drop("label",axis=1).values
X = X / 255.0
Y = pd.get_dummies(dfTrain["label"],dtype=int).values


Xtest = dfTest.drop("label",axis=1).values
Xtest = Xtest / 255.0
Ytest = pd.get_dummies(dfTest["label"],dtype=int).values

layerSizes = [784, 128, 10]
learningRate = 0.1
nbEpochs = 60
batchSize = 128
myModel = NeuralNet(X, Y, layerSizes, learningRate)

print("Starting training...")
myModel.train(nbEpochs, batchSize)

predictions = myModel.predict(Xtest.T, Ytest.T)
acc = myModel.accuracy(predictions, Ytest.T)

print(f"Final test Accuracy: {acc * 100:.2f}%")