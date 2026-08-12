import numpy as np
class Model:
    def __init__(self, features, target, alpha, c):
        self.features = features
        self.target = target
        self.weights = np.ones((features.shape[1],1))
        self.alpha = alpha
        self.c = c
        self.bias = np.mean(target,axis=0)

    def updateWeights(self):
        margins = self.target * ( self.features @ self.weights + self.bias ) # get the predictions compared to the correct values
        targetMiss = np.where(margins < 1, self.target, 0) 

        dW = self.weights - self.c * self.features.T @ targetMiss / self.features.shape[0]
        db = - self.c * np.sum(targetMiss, axis=0) / self.features.shape[0]
        self.weights = self.weights - self.alpha * dW
        self.bias = self.bias - self.alpha * db

        return self.weights

    def cost(self):
        margins = self.target * ( self.features @ self.weights + self.bias)
        hingeLoss = np.maximum(0, 1 - margins)

        return 0.5 * np.sum(self.weights**2,axis=0) + self.c * np.sum(hingeLoss, axis=0) / self.features.shape[0]

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

    def accuracy(self):
        margins = self.target * ( self.features @ self.weights + self.bias)
        count = np.sum(np.where(margins>= 0,1,0))

        return count/self.features.shape[0]     

    def predict(self, testFeatures, testTarget):
        margins = testTarget * ( testFeatures @ self.weights + self.bias)
        count = np.sum(np.where(margins>=0, 1,0 ))
        return count/testFeatures.shape[0]   
