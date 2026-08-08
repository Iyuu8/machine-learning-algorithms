import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from model import Model

df = pd.read_csv("./dataset/credit_customers.csv")

dfClean = df.dropna()
dfClean["class"] = dfClean["class"].map({"good":1,"bad":0})
dfClean["foreign_worker"] = dfClean["foreign_worker"].map({"yes":1,"no":0})
dfClean = pd.get_dummies(dfClean, columns=["checking_status","credit_history","savings_status","property_magnitude"],drop_first=True,dtype=int)
dfClean["duration_x_amount"] = dfClean["duration"] * dfClean["credit_amount"]

featuresCols = [
    'duration', 
    'credit_amount', 
    'checking_status_<0', 
    'checking_status_>=200', 
    'checking_status_no checking', 
    'credit_history_critical/other existing credit', 
    'credit_history_delayed previously', 
    'credit_history_existing paid', 
    'credit_history_no credits/all paid', 
    'savings_status_500<=X<1000', 
    'savings_status_<100', 
    'savings_status_>=1000', 
    'savings_status_no known savings', 
    'property_magnitude_life insurance', 
    'property_magnitude_no known property', 
    'property_magnitude_real estate',
    "duration_x_amount"
] 

dfTrain, dfTest = train_test_split(dfClean, test_size=0.2, random_state=42)

trainFeatures = dfTrain[featuresCols].to_numpy()
trainFeaturesScaled = (trainFeatures - np.mean(trainFeatures,axis=0)) / np.std(trainFeatures, axis=0)
trainTarget = dfTrain[["class"]].to_numpy()

testFeatures = dfTest[featuresCols].to_numpy()
testFeaturesScaled = (testFeatures - np.mean(trainFeatures,axis=0)) / np.std(trainFeatures, axis=0)
testTarget = dfTest[["class"]].to_numpy()

logisticRegressionModel = Model(trainFeaturesScaled,trainTarget,0.1)
print(logisticRegressionModel.train(count=5000)) 

print(logisticRegressionModel.test(testFeaturesScaled,testTarget))

print(logisticRegressionModel.confusionMatrix(testFeaturesScaled, testTarget))

# Feature importance: since features are scaled, weight magnitude ~ importance ( code by claude )
weightNames = featuresCols + ["bias"]
weightValues = logisticRegressionModel.weights.flatten()

importance = sorted(zip(weightNames, weightValues), key=lambda x: -abs(x[1]))
print("\nFeature importance (sorted by |weight|):")
for name, w in importance:
    print(f"{name:55s} {w:+.4f}")

plt.figure(figsize=(10, 8))
names = [x[0] for x in importance]
vals = [x[1] for x in importance]
colors = ['green' if v > 0 else 'red' for v in vals]
plt.barh(names, vals, color=colors)
plt.xlabel('Weight (standardized)')
plt.title('Feature weights — green pushes toward "good", red toward "bad"')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()