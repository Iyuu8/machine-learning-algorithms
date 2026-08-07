import random
import json

class HospitalsMap:
    def __init__(self,nbRows,nbCols,nbHouses):
        data = self.generateRandom(nbRows,nbCols,nbHouses)
        self.grid, self.houses, self.hospitals = data["grid"] , data["houseLocations"] , set()

    def printGrid(self):
        for row in self.grid: print("".join(f"{item:>5}" for item in row)) 

    def generateRandom(self,nbRows,nbCols,nbHouses):
        grid = [["." for _ in range(nbCols)] for _ in range(nbRows)]
        assignedSet = set()
        for i in range(nbRows):
            for j in range(nbCols):
                grid[i][j] = "." # empty cell
        for i in range(nbHouses):
            t = (random.randint(0,nbRows-1), random.randint(0,nbCols-1))
            while t in assignedSet: t = (random.randint(0,nbRows-1), random.randint(0,nbCols-1))
            grid[t[0]][t[1]] = "0" # a house
            assignedSet.add(t)

        return {"grid":grid,"houseLocations":assignedSet}

    def generateRandomHospitals(self,nbHospitals):
        nbRows = len(self.grid)
        nbCols = len(self.grid[0]) if nbRows>0 else 0
        for i in range(nbHospitals):
            t = (random.randint(0,nbRows-1), random.randint(0,nbCols-1))
            while t in self.houses or t in self.hospitals:t = (random.randint(0,nbRows-1), random.randint(0,nbCols-1))
            self.grid[t[0]][t[1]] = "1" # a hospital
            self.hospitals.add(t)

    def manhattan(self,t1,t2):return abs(t1[0]-t2[0]) + abs(t1[1]-t2[1])

    def calcCurrCost(self):
        c = 0
        for house in self.houses:
            d = float("inf")
            for hospital in self.hospitals:
                d = min(d,self.manhattan(house,hospital))
            c +=d
        return c

    def actions(self,hospital): # hospital is a tuple
        res = set()
        if(hospital[0]>0 and self.grid[hospital[0]-1][hospital[1]]=="."): res.add((hospital[0]-1,hospital[1]))
        if(hospital[0]<len(self.grid)-1 and self.grid[hospital[0]+1][hospital[1]]=="."): res.add((hospital[0]+1,hospital[1]))
        if(hospital[1]>0 and self.grid[hospital[0]][hospital[1]-1]=="."): res.add((hospital[0],hospital[1]-1))
        if(len(self.grid)>0 and hospital[1]<len(self.grid[0])-1 and self.grid[hospital[0]][hospital[1]+1]=="."): res.add((hospital[0],hospital[1]+1))

        return res

    def costPostAction(self,action,hospital):

        self.grid[hospital[0]][hospital[1]]="."
        self.grid[action[0]][action[1]]="1"
        self.hospitals.remove(hospital)
        self.hospitals.add(action)

        cost = self.calcCurrCost()

        self.grid[hospital[0]][hospital[1]]="1"
        self.grid[action[0]][action[1]]="."
        self.hospitals.remove(action)
        self.hospitals.add(hospital)
        return cost

    def doAction(self,action,hospital):
        self.grid[hospital[0]][hospital[1]]="."
        self.grid[action[0]][action[1]]="1"
        self.hospitals.remove(hospital)
        self.hospitals.add(action)
        return action # return the new position of the hospital



    def hillClimb(self):
        baseCost = self.calcCurrCost()
        while True:
            prevCost = baseCost
            for hospital in self.hospitals.copy(): # go over every hospital, the position of the hospital may however change throught the process
                newHospital = hospital
                while True:
                    bestAction = None
                    possibleActions = self.actions(newHospital)
                    for action in possibleActions:
                        actionPostCost = self.costPostAction(action,newHospital)
                        if actionPostCost < baseCost: 
                            bestAction = action
                            baseCost = actionPostCost
                    if bestAction is None: break
                    newHospital = self.doAction(bestAction,newHospital)
            if baseCost==prevCost: break

    def randomRestartHillClimb(self, restarts):
        globalBestCost = float('inf')
        globalBestHospitals = set()
        nbHospitals = len(self.hospitals)

        for _ in range(restarts):
            for h in self.hospitals: 
                self.grid[h[0]][h[1]] = "."
            self.hospitals.clear()
            

            self.generateRandomHospitals(nbHospitals)
            self.hillClimb()
            
            # if the result is the best so far save it
            currentCost = self.calcCurrCost()
            if currentCost < globalBestCost:
                globalBestCost = currentCost
                globalBestHospitals = self.hospitals.copy()

        # after all restarts makek the board take the best state so far
        for h in self.hospitals: 
            self.grid[h[0]][h[1]] = "."
        self.hospitals = globalBestHospitals
        for h in self.hospitals: 
            self.grid[h[0]][h[1]] = "1"

    # by claude
    def saveGridToImage(self, filename="grid.png", cellSize=6, showGridLines=True):
        from PIL import Image, ImageDraw

        nbRows = len(self.grid)
        nbCols = len(self.grid[0]) if nbRows > 0 else 0

        imgW, imgH = nbCols * cellSize, nbRows * cellSize
        img = Image.new("RGB", (imgW, imgH), (245, 245, 245))  # light gray background
        draw = ImageDraw.Draw(img)

        colors = {
            ".": (255, 255, 255),   # empty cell - white, distinct from bg
            "0": (30, 144, 255),    # house - blue
            "1": (220, 20, 60),     # hospital - red
        }

        for i in range(nbRows):
            for j in range(nbCols):
                color = colors.get(self.grid[i][j], (0, 0, 0))
                x0, y0 = j * cellSize, i * cellSize
                x1, y1 = x0 + cellSize, y0 + cellSize
                draw.rectangle([x0, y0, x1, y1], fill=color)

        if showGridLines:
            lineColor = (200, 200, 200)
            for j in range(nbCols + 1):
                x = j * cellSize
                draw.line([(x, 0), (x, imgH)], fill=lineColor)
            for i in range(nbRows + 1):
                y = i * cellSize
                draw.line([(0, y), (imgW, y)], fill=lineColor)

        img.save(filename)

    # by claude again 

    def exportState(self, name="state", cellSize=6, showGridLines=True):
        """
        Saves two files:
        - {name}.png  -> visual grid (for humans)
        - {name}.json -> exact coordinates + cost (for solvers / AI verification)
        """
        from PIL import Image, ImageDraw

        nbRows = len(self.grid)
        nbCols = len(self.grid[0]) if nbRows > 0 else 0

        # --- image (same as before) ---
        imgW, imgH = nbCols * cellSize, nbRows * cellSize
        img = Image.new("RGB", (imgW, imgH), (245, 245, 245))
        draw = ImageDraw.Draw(img)
        colors = {".": (255, 255, 255), "0": (30, 144, 255), "1": (220, 20, 60)}
        for i in range(nbRows):
            for j in range(nbCols):
                color = colors.get(self.grid[i][j], (0, 0, 0))
                x0, y0 = j * cellSize, i * cellSize
                draw.rectangle([x0, y0, x0 + cellSize, y0 + cellSize], fill=color)
        if showGridLines:
            lineColor = (200, 200, 200)
            for j in range(nbCols + 1):
                draw.line([(j * cellSize, 0), (j * cellSize, imgH)], fill=lineColor)
            for i in range(nbRows + 1):
                draw.line([(0, i * cellSize), (imgW, i * cellSize)], fill=lineColor)
        img.save(f"{name}.png")

        # --- exact data (ground truth) ---
        data = {
            "nbRows": nbRows,
            "nbCols": nbCols,
            "houses": sorted([list(h) for h in self.houses]),
            "hospitals": sorted([list(h) for h in self.hospitals]),
            "cost": self.calcCurrCost(),
        }
        with open(f"{name}.json", "w") as f:
            json.dump(data, f, indent=2)

        return data





            
        
grid = HospitalsMap(100,150,10)
grid.generateRandomHospitals(5)
# grid.saveGridToImage("before.png")
grid.exportState("before")
print(grid.calcCurrCost())

# grid.hillClimb()
grid.randomRestartHillClimb(1000)
print(grid.calcCurrCost())
# grid.saveGridToImage("after.png")
grid.exportState("after")

