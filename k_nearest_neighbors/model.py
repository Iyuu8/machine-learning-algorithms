import numpy as np
class Model:
    def __init__(self, k, features, target):
        self.k = k
        self.features = features
        self.target = target

    def predict(self, testFeatures):
        distances = np.sum((testFeatures[:, np.newaxis, :] - self.features[np.newaxis, :, :])**2,axis=2) # make each row in the test features on a different layer and that's in order for the row to get duplicated as much as the number of rows in the trainFeatures ( and hence we get the distances from all the points in the training set from the test vector ) , put the training set in a single layer so that it gets duplicated as much as the number of vector ( each vector is on an independent layer )
        nearestIndices = np.argpartition(distances, self.k-1, axis=1)[:,:self.k] # get the a vector with the indices of the k first elements being smaller than the upcoming elements and cut it at k elements
        nearestLabels = self.target[nearestIndices] # transform the indices to labels: for each row in indices get the corresponding elements in labels
        predictions = np.array([np.bincount(row).argmax() for row in nearestLabels]) # return a vector where each row is the element with the highest frequency
        return predictions

    def accuracy(self, testFeatures, testTarget): # compare the predictions with the correct classes and return the accuracy 
        predictions = self.predict(testFeatures)
        mask = predictions == testTarget
        correctCount = np.sum(mask) 

        return correctCount / len(testTarget)
