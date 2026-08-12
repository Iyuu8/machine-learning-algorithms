import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from model import Model


from sklearn.svm import SVC # to bench mark against actual library, by claude

df = pd.read_csv("./dataset/Exam_Score_Prediction.csv")
cleanDf = df.dropna()
cleanDf["internet_access"] = cleanDf["internet_access"].map({"yes":1,"no":0})
cleanDf = pd.get_dummies(cleanDf,columns=["course","sleep_quality","study_method","exam_difficulty"],drop_first=True,dtype=int)



featuresCols = ['age', 'study_hours', 'class_attendance', 'internet_access', 'sleep_hours', 'course_b.sc', 'course_b.tech', 'course_ba', 'course_bba', 'course_bca', 'course_diploma', 'sleep_quality_good', 'sleep_quality_poor', 'study_method_group study', 'study_method_mixed', 'study_method_online videos', 'study_method_self-study', 'exam_difficulty_hard', 'exam_difficulty_moderate']

dfTrain , dfTest = train_test_split(cleanDf,test_size=0.2,random_state=42)

featuresTrain = dfTrain[featuresCols].to_numpy()
targetTrainPre = dfTrain[["exam_score"]].to_numpy()
targetTrain = np.where(targetTrainPre>=60, 1 , -1)

featuresTest = dfTest[featuresCols].to_numpy()
featuresTestScaled = ( featuresTest - np.mean(featuresTrain,axis=0)) / np.std(featuresTrain,axis=0)
targetTestPre = dfTest[["exam_score"]].to_numpy()
targetTest = np.where(targetTestPre>=60, 1 , -1)


featuresTrainScaled = (featuresTrain - np.mean(featuresTrain, axis=0)) / np.std(featuresTrain,axis=0)

svmModel = Model(featuresTrainScaled,targetTrain,0.1,3)
print(svmModel.train(count=1000,printInterval=100))

print(svmModel.predict(featuresTestScaled,targetTest))



# code by claude
sk_model = SVC(kernel='linear', C=3)
sk_model.fit(featuresTrainScaled, targetTrain.ravel())

print("\n--- Sklearn SVC Results ---")
print(f"Sklearn Train Accuracy: {sk_model.score(featuresTrainScaled, targetTrain.ravel()):.4f}")
print(f"Sklearn Test Accuracy:  {sk_model.score(featuresTestScaled, targetTest.ravel()):.4f}")


# after running : 

"""
-- my models accuracy ---

0.8300625
0.837

--- Sklearn SVC Results ---

Sklearn Train Accuracy: 0.8348
Sklearn Test Accuracy:  0.8347 


"""
