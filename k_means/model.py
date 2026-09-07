import numpy as np
import random
class Model:
    def __init__(self, k, features):
        self.k = k
        self.features = features
        firstCentroid = features[random.randint(0,features.shape[0]-1)] # choose the first centroid based uniformly

        centroids = np.array([firstCentroid]) # (k,d) matrix to be constructed
        for i in range(1,k):
            distances = self.distancesFromCentroids(centroids)
            
            probas = (distances / np.sum(distances,axis=0)).squeeze()
            nextCentroid = features[np.random.choice(features.shape[0],p=probas)]
            centroids = np.vstack([centroids, nextCentroid])

        self.centroids = centroids
        
    def distancesFromCentroids(self, centroids):
        features = self.features[:, np.newaxis , :]
        centroids = centroids[np.newaxis,: , :]

        distances = np.min(np.sum((features - centroids)**2,axis=2), keepdims=True, axis=1) # (n,1) matrix of distances of each point to the nearest cetroid to it
        return distances

    def assignClasses(self):
        features = self.features[:, np.newaxis , :]
        centroids = self.centroids[np.newaxis,: , :]

        distances = np.sum((features - centroids)**2,axis=2) 
        assignedClasses = np.argmin(distances, axis=1, keepdims=True)

        return assignedClasses


    def calcCentroinds(self, assignedClasses):
        for i in range(self.k):
            mask = (assignedClasses == i).squeeze()
            if np.sum(mask) > 0:
                centroid = np.mean(self.features[mask],axis=0)
                self.centroids[i] = centroid


    def cluster(self, nbEpochs):

        for i in range(nbEpochs):
            assignedClasses = self.assignClasses()
            self.calcCentroinds(assignedClasses)

        J = self.distancesFromCentroids(self.centroids).sum()
        print(f"Final Cost = {J:.6f}")
        return J
            


    def silhouetteScore(self):
        assignedClasses = self.assignClasses()
        labels = np.array(assignedClasses).squeeze()
        nbSamples = self.features.shape[0]
        uniqueLabels = np.unique(labels)
        k = len(uniqueLabels)
    
        if k <= 1 or k == nbSamples:
            return 0.0

        
        differences = self.features[:, np.newaxis, :] - self.features[np.newaxis, :, :]
        distances = np.sqrt(np.sum(differences**2, axis=2))
        
        silhouetteCoefficients = np.zeros(nbSamples)
        for i in range(nbSamples):
            currentLabel = labels[i]
            sameClusterMask = (labels == currentLabel)
            sameClusterMask[i] = False 
            
            if np.sum(sameClusterMask) > 0: Ai = np.mean(distances[i, sameClusterMask])
            else: Ai = 0.0 
            Bi = np.inf
            for otherLabel in uniqueLabels:
                if otherLabel == currentLabel: continue
                
                otherClusterMask = (labels == otherLabel)
                avgDistToOtherCluster = np.mean(distances[i, otherClusterMask])
                if avgDistToOtherCluster < Bi: Bi = avgDistToOtherCluster

            if max(Ai, Bi) > 0:silhouetteCoefficients[i] = (Bi - Ai) / max(Ai, Bi)
            else:silhouetteCoefficients[i] = 0.0
                
        return float(np.mean(silhouetteCoefficients))

    


    






