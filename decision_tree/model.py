import numpy as np
class Leaf: 
    def __init__(self, target):
        classes, count  = np.unique(target, return_counts=True)
        self.prediction = classes[np.argmax(count)]

    def isLeaf(self):
        return True

class Node:
    def __init__(self, leftTree, rightTree, feature, treshold):
        self.leftTree = leftTree
        self.rightTree= rightTree
        self.feature = feature
        self.treshold = treshold

    def isLeaf(self):
        return False

class DecisionTreeClassifier:

    def __init__(self, depth = None, minSplit = 2, minLeaf = 1): # if the number of samples of a node fall under minSplit then it should be a leaf, minLeaf : if the number of samples of the supposed child nodes is under minLeaf ( the minimum number of nodes a leaf can contain ) then the parent node should be a leaf
        self.depth = depth
        self.minSplit = minSplit
        self.minLeaf = minLeaf
        self.root = None

    def gini(self, target):
        classes, count = np.unique(target, return_counts=True)
        return 1- np.sum((count / target.shape[0]) ** 2)

    def split(self, features, target):

        nbSamples  , nbFeatures = features.shape
        bestGain, bestTreshold, bestFeature = 0.0, None , None
        giniCurr = self.gini(target)
        for feature in range(nbFeatures):
            tresholds = np.unique(features[:,feature])
            for treshold in tresholds:
                maskLeft = features[:,feature] <= treshold
                maskRight = features[:,feature] > treshold

                if not maskRight.sum() or not maskLeft.sum(): continue # skip to the next treshold

                # featuresLeft,featuresRight  = features[maskLeft, :], features[maskRight, :] after writing the thing i realized that i don't need it 
                targetLeft, targetRight = target[maskLeft], target[maskRight]

                giniLeft, giniRight = self.gini(targetLeft), self.gini(targetRight)
                nbSamplesLeft, nbSamplesRight = maskLeft.sum(), maskRight.sum()
                newGain  = giniCurr - ( (nbSamplesRight / nbSamples) * giniRight + (nbSamplesLeft / nbSamples) * giniLeft)

                if newGain > bestGain:
                    bestGain = newGain
                    bestFeature = feature
                    bestTreshold = treshold

        return bestFeature, bestTreshold

    def _buildTree(self, features, target , depth=0):

        if(self.depth is not None and depth>= self.depth): return Leaf(target)
        classes = np.unique(target)
        nbSamples = target.shape[0]
        if(len(classes) == 1 or nbSamples < self.minSplit): return Leaf(target)

        bestFeature, bestTreshold = self.split(features,target)
        if bestFeature is None: return Leaf(target)

        leftMask, rightMask = features[:,bestFeature] <= bestTreshold, features[:,bestFeature] > bestTreshold

        if(leftMask.sum() < self.minLeaf or rightMask.sum() < self.minLeaf): return Leaf(target)
        leftFeatures, leftTarget = features[leftMask,:], target[leftMask]
        rightFeatures, rightTarget = features[rightMask,:], target[rightMask]

        leftTree = self._buildTree(leftFeatures,leftTarget,depth+1)
        rightTree = self._buildTree(rightFeatures,rightTarget,depth+1)

        return Node(leftTree,rightTree,bestFeature,bestTreshold) 

    def buildTree(self, features,target):
        self.root = self._buildTree(features,target,0)

    def _predictOne(self, observation, node):
        if node.isLeaf(): return node.prediction

        if observation[node.feature] <= node.treshold: return self._predictOne(observation,node.leftTree)
        return self._predictOne(observation,node.rightTree)

    def predict(self, features): # i realized that i should have named them observations instead of features because features is really misleading, but well lazy to change the name
        if self.root is None: return None
        return np.asarray([self._predictOne(observation,self.root) for observation in features])


    





