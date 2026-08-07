import numpy as np
class Model: # this model accepts only features with a type of number

    # features and target are numpy matrices
    def __init__(self,features,target,learningRate):

        self.a = learningRate
        self.features = np.hstack([features,np.ones((features.shape[0],1))]) # matrix of features, ideal for matrix multiplication
        self.target = target 
        targetMean = target.mean()
        self.weights = np.append(np.zeros(len(features[0])),targetMean).reshape(-1,1)

    def cost(self): 
        estimates = self.features @ self.weights
        errors = self.target - estimates
        return np.mean(errors**2)
    
    def gradientCost(self):
        errors = (self.features @ self.weights) - self.target
        return (2/len(self.target)) * (np.transpose(self.features,(1,0)) @ errors)

    def updateWeights(self):
        self.weights = self.weights - self.a * self.gradientCost()

    def train(self):
        i=0
        while True: 
            i+=1
            costOld = self.cost()
            self.updateWeights()
            costNew = self.cost()
            # if ( abs(costOld-costNew)/costOld < 0.0001): break
            if i>1000: break

        return {
            "cost":costNew,
            "estimates":self.features @ self.weights
        }

    def test(self, testFeatures, testTarget):
        featuresMatrix = np.hstack([testFeatures,np.ones((testFeatures.shape[0],1))])
        estimates = featuresMatrix @ self.weights
        cost = np.mean((testTarget-estimates)**2)

        return {
            "cost":cost,
            "meanAbsError": np.mean(np.abs(testTarget-estimates)),
            "estimates":estimates
        }

    def meanAbsError(self):
        return np.mean(np.abs(self.target - self.features @ self.weights))


        
        