import numpy as np
class Model:
    def __init__(self, features, target, learningRate):
        self.alpha = learningRate
        self.features = np.hstack([features,np.ones((features.shape[0],1))])
        self.weights = np.zeros((features.shape[1]+1,1))
        self.weights[len(self.weights)-1][0] = target.mean() # the mean of the values
        self.target = target

    def sigmoid(self,z):
        return 1/(1+np.exp(z*-1))

    # the acual formula is J = 1/m sum(1,m,yi*ln(p(xi)) + (1-yi)*ln(1-p(xi))) where p(xi) = y^i= 1/(1+e^-z), z = W * X ( weights vector times features vector ), however the formula can be changed with another formula that is more efficient to use ( numpy ) using linear algebra: J = (1/m)(Y_T * Z^) + mean(ln(1-Y^)) 
    def cost(self): 
        epsilon = 1e-15
        preEstimated = self.features @ self.weights # Z^ 
        estimated = np.clip(self.sigmoid(preEstimated),epsilon,1-epsilon)# Y^

        return (-1/len(self.target))*(self.target.T @ preEstimated) - np.log(1-estimated).mean()

    def gradient(self):
        epsilon = 1e-15
        estimated = np.clip(self.sigmoid(self.features @ self.weights) , epsilon, 1-epsilon)
        errors = estimated - self.target
        return (1/len(self.target)) * (self.features.T @ errors)

    def updateWeights(self):
        self.weights = self.weights - self.alpha * self.gradient()

    def train(self,count = 10000,printInterval=1000):
        i=0
        while True: 
            i+=1
            self.updateWeights()
            if i > count: break
            if (i % printInterval == 0):
                current_cost = self.cost()
                # Ensure the cost is extracted if it's a 1x1 matrix/array
                if isinstance(current_cost, np.ndarray):
                    current_cost = current_cost.item()
                print(f"Iteration {i}: Cost = {current_cost:.6f}")

        print("end")
        return self.accuracy()

    def classify(self,y):
        return np.where(y >= 0.80, 1, 0)
    def accuracy(self):
        epsilon = 1e-15
        estimated = np.clip(self.sigmoid(self.features @ self.weights) , epsilon, 1-epsilon)
        mask = self.classify(estimated) == self.target
        correctCount = len(self.target[mask])
        return {"accuracy":correctCount/len(self.target), "correctCount":correctCount,"noneCorrect":len(self.target)-correctCount}

    def test(self, testFeatures,testTarget):
        epsilon = 1e-15
        testFeatures = np.hstack([testFeatures, np.ones((testFeatures.shape[0], 1))])
        estimated = np.clip(self.sigmoid(testFeatures @ self.weights), epsilon, 1-epsilon)
        

        mask = self.classify(estimated) == testTarget
        correctCount = len(testTarget[mask])
        return {"accuracy":correctCount/len(testTarget), "correctCount":correctCount,"noneCorrect":len(testTarget)-correctCount}

    # by claude

    def confusionMatrix(self, testFeatures, testTarget):
        epsilon = 1e-15
        testFeaturesB = np.hstack([testFeatures, np.ones((testFeatures.shape[0], 1))])
        estimated = np.clip(self.sigmoid(testFeaturesB @ self.weights), epsilon, 1-epsilon)
        predictions = self.classify(estimated)

        tp = int(np.sum((predictions == 1) & (testTarget == 1)))
        tn = int(np.sum((predictions == 0) & (testTarget == 0)))
        fp = int(np.sum((predictions == 1) & (testTarget == 0)))
        fn = int(np.sum((predictions == 0) & (testTarget == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0

        return {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1
        }
        
        

    



        