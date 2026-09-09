import numpy as np
from collections import deque
import random

class Model:
    def __init__(self, epsilon, m, dataset):
        self.epsilon = epsilon
        self.m = m
        self.dataset = dataset

        datasetExtended1 = dataset[np.newaxis, :, :]
        datasetExtended2 = dataset[:, np.newaxis, :]

        distances = np.sqrt(np.sum((datasetExtended1 - datasetExtended2)**2, axis=2))

        self.distances = distances
        self.clusters = [-1] * dataset.shape[0]

    def createClusters(self):
        visited = np.zeros(self.dataset.shape[0],dtype=bool)
        queue = deque([])

        clustersCount = 0

        unvisitedIndices = np.where(~visited)[0]

        ind = random.choice(unvisitedIndices)
        Nepsilon, count = self.computeNepsilon(ind)

        while True:
            visited[ind] = True 

            if count >= self.m :
                self.clusters[ind] = clustersCount
                queue.extend(Nepsilon)

                while len(queue) != 0:
                    neighborInd = queue.popleft()

                    if self.clusters[neighborInd] != -1 : continue 

                    if not visited[neighborInd]:
                        visited[neighborInd] = True
                        self.clusters[neighborInd] = clustersCount
                        Nepsilon, count = self.computeNepsilon(neighborInd)

                        if count > self.m: queue.extend(Nepsilon)

                # end of maximal set of density connected points ( cluster )

                clustersCount += 1
            unvisitedIndices = np.where(~visited)[0]
            if len(unvisitedIndices) == 0: break
            ind = random.choice(unvisitedIndices)
            Nepsilon, count = self.computeNepsilon(ind)

        return self.clusters


            
    def computeNepsilon(self, ind):
        Nepsilon = np.where(self.distances[ind] < self.epsilon)[0]
        Nepsilon = Nepsilon[Nepsilon != ind]

        return Nepsilon, Nepsilon.shape[0]
        

        



