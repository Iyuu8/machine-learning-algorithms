import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from model import DecisionTreeRegression
from model import RandomForest

df = pd.read_csv("./dataset/Exam_Score_Prediction.csv")
dfClean = df.dropna()
dfClean["internet_access"] = dfClean["internet_access"].map({"yes":1,"no":0})
dfClean["sleep_quality"] = dfClean["sleep_quality"].map({"poor":0,"average":1,"good":2})
dfClean["facility_rating"] = dfClean["facility_rating"].map({"low":0,"medium":1,"high":2})
dfClean["exam_difficulty"] = dfClean["exam_difficulty"].map({"easy":0,"moderate":1,"hard":2})


dfClean = pd.get_dummies(dfClean, columns=["gender","course","study_method"], drop_first=True, dtype=int)
featuresCols = [
    'age', 'study_hours', 'class_attendance', 'internet_access', 'sleep_hours', 'sleep_quality', 'facility_rating', 'exam_difficulty','gender_male', 'gender_other', 'course_b.sc', 'course_b.tech', 'course_ba', 'course_bba', 'course_bca', 'course_diploma', 'study_method_group study', 'study_method_mixed', 'study_method_online videos', 'study_method_self-study'
]



dfTrain, dfTest = train_test_split(dfClean, test_size=0.2, random_state=42)


trainFeatures = dfTrain[featuresCols].to_numpy()

trainTarget = dfTrain[["exam_score"]].to_numpy().ravel()

testFeatures = dfTest[featuresCols].to_numpy()

testTarget = dfTest[["exam_score"]].to_numpy().ravel()

model = RandomForest(nbTrees=200,bagSize=0.8, depth=12, minLeaf=5)
model.buildForest(trainFeatures,trainTarget)

print(np.abs(np.asarray(model.predict(testFeatures)).ravel() - testTarget).mean())


# by gemini to compare with scikit learn
sklearn_model = RandomForestRegressor(
    n_estimators=200, 
    max_samples=0.8, 
    max_depth=12, 
    min_samples_leaf=5, 
    max_features=max(1, trainFeatures.shape[1] // 3),
    random_state=42
)

# Train and predict
sklearn_model.fit(trainFeatures, trainTarget)
sklearn_preds = sklearn_model.predict(testFeatures)

# Calculate MAE
sklearn_mae = mean_absolute_error(testTarget, sklearn_preds)
print(f"Scikit-Learn MAE: {sklearn_mae}")

