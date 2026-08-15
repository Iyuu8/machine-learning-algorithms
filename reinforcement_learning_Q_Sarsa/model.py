import math
import random
import math
import numpy as np
from collections import defaultdict
class GameEngine:
    def __init__(self,speedMax, throttleFailProb, etha, track):
        self.ACTIONS = {
            "dir":{"N":0,"NE":1,"E":2,"SE":3,"S":4,"SW":5,"W":6,"NW":7},
            "acc":[-2,-1,0,1,2]
        }
        self.unitVector = {
            "N": (0,3),
            "S": (0,-3),
            "E": (3,0),
            "W": (-3,0),
            "NE": (2,2),
            "NW": (-2,2),
            "SE": ( 2,-2),
            "SW": (-2,-2)
        }
        self.speedMax  = speedMax
        self.throttleFailProb = throttleFailProb
        self.etha = etha
        self.track = track

    def allActions(self):

        actions = []
        for d in self.ACTIONS["dir"].keys():
            for acc in self.ACTIONS["acc"]:
                actions.append({"dir":d, "acc":acc})
        return actions  

    def getLineCells(self, x0, y0, x1, y1):
        cells = []
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        while True:
            cells.append({"x":x, "y":y})
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return cells

    def isOnTrack(self, x, y):
        if y < 0 or y >= len(self.track) or x < 0 or x >= len(self.track[0]):
            return False
        return self.track[y][x] != "#"

    def randomStartCell(self):
        starts = [{"x":c, "y":r} for r, row in enumerate(self.track) for c, ch in enumerate(row) if ch == "S"]
        return random.choice(starts)

    def turnFailureProba(self, speed, dirState,dirAction):
        turnValue = min(abs(self.ACTIONS["dir"][dirAction]-self.ACTIONS["dir"][dirState]),8-abs(self.ACTIONS["dir"][dirAction]-self.ACTIONS["dir"][dirState]))
        return np.clip(self.etha * speed * (math.sin(turnValue * math.pi/8)**2),0,1)

    #action is a dict {dir:string, acc:int}
    #state is a dict {x:int,y:int,speed:int,dir:string}
    def transitionModel(self, state, action):
        probaTurn = random.random()
        probaAcc = random.random()
        nextState = dict()

        if probaAcc > self.throttleFailProb:
            nextState["speed"] = int(np.clip(state["speed"] + action["acc"],0,self.speedMax))
        else: nextState["speed"] = state["speed"]

        if probaTurn > self.turnFailureProba(state["speed"],state["dir"],action["dir"]):
            nextState["dir"]=action["dir"]
        else: nextState["dir"] = state["dir"]

        tentativeNextX = state["x"] + nextState["speed"] * self.unitVector[nextState["dir"]][0]
        tentativeNextY = state["y"] + nextState["speed"] * self.unitVector[nextState["dir"]][1]

        pathBetweenPositions = self.getLineCells(state["x"],state["y"],tentativeNextX,tentativeNextY)
        crashed = False
        for cell in pathBetweenPositions[1:]:
            if not self.isOnTrack(cell["x"],cell["y"]): crashed = True
            if crashed: break

        if crashed: # repeat from the very beginning
            start = self.randomStartCell()
            nextState["x"]=start["x"]
            nextState["y"]=start["y"]
            nextState["speed"]=0
            nextState["dir"]=state["dir"]
            nextState["crashed"]=True
            reward = -2

            return nextState , reward , False # not done yet

        nextState["x"], nextState["y"] = tentativeNextX, tentativeNextY
        nextState["crashed"]=False

        if self.track[nextState["y"]][nextState["x"]]=="F": 
            reward = 20
            return nextState, reward, True # reached the finish line

        reward = -1
        return nextState, reward, False

# take tuple return dict
def stateKey(state):
    return (state["x"], state["y"], state["speed"], state["dir"])

def actionKey(action):
    return (action["dir"], action["acc"])

#take dict return tuple
def stateDict(state):
    return {"x":state[0],"y":state[1],"speed":state[2],"dir":state[3]}

def actionDict(action):
    return {"dir":action[0],"acc":action[1]}


# start is a dict : {"x":int,"y":int}
class QLearning:
    def __init__(self, track,speedMax, epsilon, throttleFailProb, etha, alpha, gamma):
        self.track = track
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.qTable  = defaultdict(float)
        self.visitCount = defaultdict(int)
        self.engine = GameEngine(speedMax,throttleFailProb,etha,track)


    def chooseAction(self, state):
        randomActionProba = random.random()
        possibleActions = self.engine.allActions()
        if randomActionProba <= self.epsilon:
            return random.choice(possibleActions)
        else: 
            qValues = [ self.qTable[(stateKey(state),actionKey(action))] for action in possibleActions]
            qMax = max(qValues)
            bestActions = [ action for action in possibleActions if self.qTable[(stateKey(state),actionKey(action))]==qMax]
            return random.choice(bestActions)

    def chooseOptimalAction(self,state):
        possibleActions = self.engine.allActions()
        qValues = [ self.qTable[(stateKey(state),actionKey(action))] for action in possibleActions]
        qMax = max(qValues)
        bestActions = [ action for action in possibleActions if self.qTable[(stateKey(state),actionKey(action))]==qMax]
        return random.choice(bestActions)
    def train(self, nbEpisodes, nbStepsPerEpisode):
        possibleActions = self.engine.allActions()
        episodeReturns = []
        decaySteps = nbEpisodes
        epsilonStart = self.epsilon
        epsilonMin = 0.01 
        for i in range(nbEpisodes):
            frac = min(1.0, i / decaySteps)
            self.epsilon = epsilonStart + frac * (epsilonMin - epsilonStart)

            start = self.engine.randomStartCell()
            state = {"x":start["x"],"y":start["y"],"speed":0,"dir":"N"}
            totalReward = 0.0
            j = 0
            done = False
            
            while not done and j<nbStepsPerEpisode:
                
                action = self.chooseAction(state)
                self.visitCount[(stateKey(state),actionKey(action))] +=1
                ALPHA = 1.0 / (1 + self.visitCount[(stateKey(state),actionKey(action))] * 0.01)
                #ALPHA = max(1.0 / self.visitCount[(stateKey(state),actionKey(action))], self.alpha)
                nextState, reward, done = self.engine.transitionModel(state,action)
                totalReward += reward
                bestNext = 0.0 if done else max(self.qTable[(stateKey(nextState),actionKey(action2))] for action2 in possibleActions)

                self.qTable[(stateKey(state),actionKey(action))] = self.qTable[(stateKey(state),actionKey(action))] + ALPHA * (reward + self.gamma * bestNext - self.qTable[(stateKey(state),actionKey(action))])
                state = nextState
                j+=1

            episodeReturns.append(totalReward)
        return self.qTable, episodeReturns

    def run(self, maxSteps):
        start = self.engine.randomStartCell()
        state = {"x":start["x"],"y":start["y"],"speed":0,"dir":"N"}
        states = [state]
        done = False
        j=0
        while not done and j<maxSteps:
            action = self.chooseOptimalAction(state)
            stateNext, reward, done = self.engine.transitionModel(state,action)
            states.append(stateNext)
            state = stateNext
            j+=1
        return states
    
class Sarsa:
    def __init__(self, track,speedMax, epsilon, throttleFailProb, etha, alpha, gamma):
        self.track = track
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.qTable  = defaultdict(float)
        self.visitCount = defaultdict(int)
        self.engine = GameEngine(speedMax,throttleFailProb,etha,track)


    def chooseAction(self, state):
        randomActionProba = random.random()
        possibleActions = self.engine.allActions()
        if randomActionProba <= self.epsilon:
            return random.choice(possibleActions)
        else: 
            qValues = [ self.qTable[(stateKey(state),actionKey(action))] for action in possibleActions]
            qMax = max(qValues)
            bestActions = [ action for action in possibleActions if self.qTable[(stateKey(state),actionKey(action))]==qMax]
            return random.choice(bestActions)

    def chooseOptimalAction(self,state):
        possibleActions = self.engine.allActions()
        qValues = [ self.qTable[(stateKey(state),actionKey(action))] for action in possibleActions]
        qMax = max(qValues)
        bestActions = [ action for action in possibleActions if self.qTable[(stateKey(state),actionKey(action))]==qMax]
        return random.choice(bestActions)
    def train(self, nbEpisodes, nbStepsPerEpisode):
        possibleActions = self.engine.allActions()
        decaySteps = nbEpisodes
        episodeReturns = []
        epsilonStart = self.epsilon
        epsilonMin = 0.01 
        for i in range(nbEpisodes):
            frac = min(1.0, i / decaySteps)
            self.epsilon = epsilonStart + frac * (epsilonMin - epsilonStart)

            start = self.engine.randomStartCell()
            state = {"x":start["x"],"y":start["y"],"speed":0,"dir":"N"}
            action = self.chooseAction(state)
            totalReward = 0.0
            j = 0
            done = False
            
            while not done and j<nbStepsPerEpisode:
                
                
                self.visitCount[(stateKey(state),actionKey(action))] +=1
                ALPHA = 1.0 / (1 + self.visitCount[(stateKey(state),actionKey(action))] * 0.01)
                #ALPHA = max(1.0 / self.visitCount[(stateKey(state),actionKey(action))], self.alpha)
                nextState, reward, done = self.engine.transitionModel(state,action)
                nextAction = self.chooseAction(nextState)
                totalReward += reward

                qUpdate = 0.0 if done else self.qTable[(stateKey(nextState),actionKey(nextAction))]

                self.qTable[(stateKey(state),actionKey(action))] = self.qTable[(stateKey(state),actionKey(action))] + ALPHA * (reward + self.gamma * qUpdate - self.qTable[(stateKey(state),actionKey(action))])

                state = nextState
                action = nextAction
                j+=1

            episodeReturns.append(totalReward)
        return self.qTable, episodeReturns

    def run(self, maxSteps):
        start = self.engine.randomStartCell()
        state = {"x":start["x"],"y":start["y"],"speed":0,"dir":"N"}
        states = [state]
        done = False
        j=0
        while not done and j<maxSteps:
            action = self.chooseOptimalAction(state)
            stateNext, reward, done = self.engine.transitionModel(state,action)
            states.append(stateNext)
            state = stateNext
            j+=1
        return states