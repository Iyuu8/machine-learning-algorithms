import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from model import Model

df = pd.read_csv("student_performance_dataset.csv")
dfClean = df.dropna()

dfClean["part_time_job"] = dfClean["part_time_job"].map({"Yes":1,"No":0})
dfClean["internet_access"] = dfClean["internet_access"].map({"Yes":1,"No":0})
dfClean = pd.get_dummies(dfClean, columns=["parental_education","gender"],drop_first=True,dtype=int)


featuresCols = [
    "study_time_hours",
    "previous_grade",
    "attendance_percent",
    "sleep_hours",
    "part_time_job",
    "internet_access",
    "parental_education_High School", 
    "parental_education_Masters", 
    "parental_education_PhD",
    "gender_Male"
]
print(dfClean.columns.to_list())




dfTrain, dfTest = train_test_split(dfClean,test_size=0.2,random_state=42) 
features = dfTrain[featuresCols].to_numpy()
target = dfTrain[["final_exam_score"]].to_numpy()

testFeatures = dfTest[featuresCols].to_numpy()
testFeaturesScaled = (testFeatures - np.mean(features,axis=0)) / np.std(features,axis=0)
testTarget = dfTest[["final_exam_score"]].to_numpy()


featuresScaled = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

linearRegressionModel = Model(featuresScaled,target,0.1)
baseCost = linearRegressionModel.cost()
resultTraining = linearRegressionModel.train()
print(f"accuracy: {(1-(resultTraining["cost"] / baseCost)) * 100:.2f}% | Weights: {linearRegressionModel.weights}")

for i in range(10):
    print(resultTraining["estimates"][i],target[i])

print(linearRegressionModel.meanAbsError())

testResult = linearRegressionModel.test(testFeaturesScaled,testTarget)

print(testResult["meanAbsError"])
for i in range(10):
    print(testResult["estimates"][i],testTarget[i])

















