import numpy as np

class Model:
    def __init__(self, k, dataset):
        self.k = k
        self.dataset = dataset
        self.W = None

    def compress(self):

        cov = (1/self.dataset.shape[0]) * ( self.dataset.T @ self.dataset )
        eigenvalues, eigenvectors = np.linalg.eigh(cov) # eigenvalues is a flat list, is a d*d tensor eigenvectors, eigh is dedicated for sym matrices and returns orthogonal eigenvectors

        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:,::-1]
        W = eigenvectors[:, :self.k]

        self.W = W
        reducedData = self.dataset @ W
        return reducedData, W, eigenvalues, eigenvalues[:self.k]

    def deCompress(self, compressedData): # approximate the original data through the compressed data
        approxData = compressedData @ self.W.T
        error = np.sum(np.abs(self.dataset - approxData))

        return approxData , error
