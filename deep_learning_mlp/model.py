import numpy as np

def relu(Z): 
    return np.maximum(0,Z)

def relu_deriv(Z):
    return (Z>0).astype(float)

# the column here is actually a 1d list ( a row ), the output is also the same
def softmax(Z):
   exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
   return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def crossEntropy(AL,Y):
    AL = np.clip(AL, 1e-15, 1.0 - 1e-15)
    return -1.0 * np.mean(np.sum(Y * np.log(AL), axis=0))
    
class NeuralNet:

    # layers is an array containing the number of neurons in each layer including the input and output layers
    # X and Y are directly loaded from pandas, hence the rows are the observations, and the columns are the features, the convention this program uses though is rows for the features and columns for the observations so we take their transpose instead, ( we must make U one-hot encoded too)
    def __init__(self, X, Y, layers, alpha):
        self.layers = [None]*len(layers) 
        self.nbNeurons = layers
        self.Y = Y.T
        self.X = X.T
        self.ALPHA = alpha

        # the weights here is an array of matrices 
        # the weights that link layer l and l+1 are in index l

        # the bias is proadcasted in the operations
        # the bias applied to layer l to get layer l+1 is in index l
        self.weights = [None]
        self.bias = [None]

        for i in range(len(layers)):
            if i==0: continue
            self.weights.append(np.random.randn(layers[i],layers[i-1]) * np.sqrt(2/layers[i-1]))
            self.bias.append(np.zeros((layers[i],1)))

    # X and Y are in batches, we do not use the ones stored in the class itself directly, no need to transpose them here, because they are direclty drawn from class itself where they were transponsed
    def _forward(self,X,Y):
        self.layers[0]={"Z":X,"A":X}
        for l in range(len(self.nbNeurons)):
            if l==0: continue
            Zl = self.weights[l] @ self.layers[l-1]["A"] + self.bias[l] 
            Al = relu(Zl) if l<len(self.nbNeurons)-1 else softmax(Zl)

            # works as cache in the backward pass
            self.layers[l] = {"Z": Zl, "A": Al}
        return crossEntropy(self.layers[len(self.layers)-1]["A"],Y)

    def _backward(self,Y):
        dZ = [None] * len(self.nbNeurons)
        dZ[len(dZ)-1] = self.layers[len(self.nbNeurons)-1]["A"]-Y

        for l in range(len(self.nbNeurons)-1,0,-1):
            if l < len(self.nbNeurons)-1:
                dZl = self.weights[l+1].T @ dZ[l+1] * (relu_deriv(self.layers[l]["Z"]))
                dZ[l]=dZl
            dWl = (1/Y.shape[1]) * (dZ[l] @ self.layers[l-1]["A"].T)
            dBl = (1/Y.shape[1]) * np.sum(dZ[l],axis=1,keepdims=True)
            self.weights[l]-= self.ALPHA * dWl
            self.bias[l] -= self.ALPHA * dBl

    def _getBatches(self,batchSize):
        m = self.X.shape[1]                       # your convention: examples are columns
        perm = np.random.permutation(m)      # shuffle indices
        X = self.X[:, perm]
        Y = self.Y[:, perm]
        
        batches = []
        for i in range(0, m, batchSize):
            XBatch = X[:, i:i+batchSize]
            YBatch = Y[:, i:i+batchSize]
            batches.append((XBatch, YBatch))
        return batches

    
    def train(self, nbEpochs, batchSize):
        for i in range(nbEpochs):
            lossEpoch = 0
            batches = self._getBatches(batchSize) # we split the data to shuffled batches ( to escape local minima )
            for (X,Y) in batches:
                loss = self._forward(X,Y)
                lossEpoch += loss
                self._backward(Y)
            print(f"epoch {i}, avg loss {lossEpoch/len(batches):.4f}")

    def predict(self, X , Y):
        self._forward(X,Y)
        predictedClasses = np.argmax(self.layers[len(self.layers)-1]["A"], axis=0)
        nbClasses = self.layers[len(self.layers)-1]["A"].shape[0]
        
        return np.eye(nbClasses)[predictedClasses].T

    def accuracy(self, predictions, Y):
        return np.mean(np.argmax(predictions, axis=0) == np.argmax(Y, axis=0))




        
            



            
