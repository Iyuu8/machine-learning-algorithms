import numpy as np
class Leaf: 
    def __init__(self, target):
        self.prediction = target.mean()

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

class DecisionTreeRegression:

    def __init__(self, depth = None, minSplit = 2, minLeaf = 1, randomFeatures = False): # if the number of samples of a node fall under minSplit then it should be a leaf, minLeaf : if the number of samples of the supposed child nodes is under minLeaf ( the minimum number of nodes a leaf can contain ) then the parent node should be a leaf
        self.depth = depth
        self.minSplit = minSplit
        self.minLeaf = minLeaf
        self.randomFeatures = randomFeatures
        self.root = None

    def mse(self, target):
        if len(target)==0: return 0
        return np.var(target) # 1/len(target) * sum(1,len(target), (yi-E(Y))^2) the mean squared error is basically the variance calculated by numpy

    # we try to minimize the variance in each node ( a low variance means that the data is highly homogeneus in the node )
    def split(self, features, target):

        nbSamples  , chosenFeatures = features.shape[0],self.getFeatures(features,max(1,features.shape[1]//3),random=self.randomFeatures)
        bestGain, bestTreshold, bestFeature = 0.0, None , None
        mseCurr = self.mse(target)
        for feature in chosenFeatures:
            tresholds = np.unique(features[:,feature])
            for treshold in tresholds:
                maskLeft = features[:,feature] <= treshold
                maskRight = features[:,feature] > treshold

                if not maskRight.sum() or not maskLeft.sum(): continue # skip to the next treshold

                # featuresLeft,featuresRight  = features[maskLeft, :], features[maskRight, :] after writing the thing i realized that i don't need it 
                targetLeft, targetRight = target[maskLeft], target[maskRight]

                mseLeft, mseRight = self.mse(targetLeft), self.mse(targetRight)
                nbSamplesLeft, nbSamplesRight = maskLeft.sum(), maskRight.sum()
                newGain  = mseCurr - ( (nbSamplesRight / nbSamples) * mseRight + (nbSamplesLeft / nbSamples) * mseLeft)

                if newGain > bestGain:
                    bestGain = newGain
                    bestFeature = feature
                    bestTreshold = treshold

        return bestFeature, bestTreshold

    def _buildTree(self, features, target , depth=0):

        if(self.depth is not None and depth>= self.depth): return Leaf(target)
        nbSamples = target.shape[0]
        if(len(np.unique(target)) == 1 or nbSamples < self.minSplit): return Leaf(target)

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

    # for a random forest
    def getFeatures(self,features,n, random=False):
        nbFeatures = features.shape[1]
        if random:
            return np.random.choice(nbFeatures, size=n, replace=False)
        return np.arange(nbFeatures)


class RandomForest:
    def __init__(self,nbTrees,bagSize,depth=None,minSplit=2,minLeaf=1):
        self.nbTrees = nbTrees
        self.bagSize = bagSize
        self.params = {"depth":depth,"minSplit":minSplit,"minLeaf":minLeaf}

        self.trees = []

    def buildForest(self,features,target):
        for i in range(self.nbTrees):
            chosenIndices = np.random.choice(features.shape[0], size=int(features.shape[0]*self.bagSize), replace=True)
            chosenFeatures = features[chosenIndices,:]
            chosenTargets = target[chosenIndices]

            tree = DecisionTreeRegression(self.params["depth"],self.params["minSplit"],self.params["minLeaf"],randomFeatures=True)
            tree.buildTree(chosenFeatures,chosenTargets)

            self.trees.append(tree)
            print(i)

    def predict(self, features):
        predicitons = np.array([tree.predict(features) for tree in self.trees])
        return np.mean(predicitons, axis=0)




    





