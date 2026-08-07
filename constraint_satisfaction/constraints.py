import time, random

# the problem : 
"""
    we have 7 subjects and 4 students 
    student 1: A , B  , C
    student 2: B , D  , E
    student 3: C , E  , F
    student 4: E , F  , G

    a student can't have more than one subject in a single day
    each exam can be in one of the following days : Tue, Mon , Wed


"""


class ConstraintProblem:
    def __init__(self, vars, constraints, domainValues):
        self.vars = vars
        self.assignments = set()
        self.constraints = constraints
        self.domain = domainValues

        # Pre-calculated neighbor map for fast degree and consistency checks
        self.neighbors = {}
        for var in self.vars:
            self.neighbors[var] = set()

        for constraint in self.constraints:
            self.neighbors[constraint[0]].add(constraint[1])
            self.neighbors[constraint[1]].add(constraint[0])

    def backTrackSearch(self):
        if self.completeAssignement(): return self.assignments
        var = self.selectUnassignedVar()
        for val in self.possibleValues(var):
            if self.consistent((var,val)):
                self.assignments.add((var,val))
                res = self.backTrackSearch()
                if res != None: return res
                self.assignments.remove((var,val))
        return None

    def completeAssignement(self):
        return len(self.assignments) == len(self.vars)

    def selectUnassignedVar(self):
        assignedVars = set()
        for pair in self.assignments:
            assignedVars.add(pair[0])

        unassigned = []
        for var in self.vars:
            if var not in assignedVars:
                unassigned.append(var)

        if len(unassigned) == 0:
            return None

        bestVar = None
        minValuesCount = float("inf")
        maxDegree = -1

        for var in unassigned:
            # 1. Minimum Remaining Values (MRV) check
            numValues = len(self.possibleValues(var))

            # 2. Fast Degree Heuristic check
            degree = 0
            for neighbor in self.neighbors[var]:
                if neighbor in unassigned:
                    degree = degree + 1

            # Select variable with fewest options; break ties with higher degree
            if numValues < minValuesCount:
                minValuesCount = numValues
                maxDegree = degree
                bestVar = var
            elif numValues == minValuesCount:
                if degree > maxDegree:
                    maxDegree = degree
                    bestVar = var

        return bestVar

    def possibleValues(self,var):
        validValues = []
        for val in self.domain:
            if self.consistent((var,val)):
                validValues.append(val)
        return validValues

    def consistent(self,assignment):
        var = assignment[0]
        val = assignment[1]

        for pair in self.assignments:
            assignedVar = pair[0]
            assignedVal = pair[1]

            if assignedVal == val and assignedVar in self.neighbors[var]:
                return False
        return True
        



vars = set()
for r in range(10):
    for c in range(10):
        vars.add(f"Cell_{r}_{c}")


constraints = set()
for r1 in range(10):
    for c1 in range(10):
        for r2 in range(10):
            for c2 in range(10):
                if (r1 == r2 or c1 == c2) and not (r1 == r2 and c1 == c2):
                    constraints.add((f"Cell_{r1}_{c1}", f"Cell_{r2}_{c2}"))

# 7 Colors to choose from
domain = ["Red", "Green", "Blue", "Yellow", "Purple", "Orange", "Black","White","Grey","Cyan"]
problem = ConstraintProblem(vars, constraints, domain)

print("Starting search...")

# Start the timer
startTime = time.time()

# Run the algorithm
res = problem.backTrackSearch()

endTime = time.time()

print("Result:", res)
# Print the difference
print(f"Execution Time: {endTime - startTime} seconds")
            


