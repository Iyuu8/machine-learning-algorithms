class TicTacToeAgent:
    def __init__(self):
        pass

    def boardFull(self,state):
        for row in state:
            for col in row:
                if col == ".": return False
        return True

    def winner(self,state):

        # rows
        for row in state:
            if row[0] != "." and row[0] == row[1] == row[2]:
                return row[0]
        # columns
        for col in range(3):
            if state[0][col] != "." and state[0][col] == state[1][col] == state[2][col]:
                return state[0][col]
        # diagonal 1
        if state[0][0] != "." and state[0][0] == state[1][1] == state[2][2]:
            return state[0][0]
        # diagonal 2
        if state[0][2] != "." and state[0][2] == state[1][1] == state[2][0]:
            return state[0][2]

        return None

    def terminal(self,state):

        if self.winner(state) is not None:
            return True

        if self.boardFull(state):
            return True

        return False

    # this method is called only when there terminal is true
    def value(self,state):
        if self.terminal(state):
            x = self.winner(state)
            if x=="x": return 1
            elif x=="o": return -1
            else: return 0
        return None

    def player(self,state):
        #X plays when both counts are equal, and O plays when X has one more mark than O
        count =0
        for row in state:
            for col in row:
                if(col=="x"): count+=1
                elif(col=="o"):count-=1
        return "x" if count==0 else "o"

    def actions(self,state):
        actions = []
        if self.terminal(state): return None
        player = self.player(state)
        for i,row in enumerate(state):
            for j,col in enumerate(row):
                if col==".":
                    actions.append({
                        "row":i,
                        "col":j,
                    })

        return {
            "player":player,
            "listActions":actions
        }

    def setMoveCopyState(self,state,move,player):
        state2 = [row.copy() for row in state]
        state2[move["row"]][move["col"]] = player
        return state2

    def setMove(self,state,move,player):
        if(move is not None): state[move["row"]][move["col"]] = player

    def isBetterForMax(self, candidate, current):
        if candidate["value"] != current["value"]:
            return candidate["value"] > current["value"]
        v = candidate["value"]
        if v > 0:   # winning = prefer sooner (lower depth)
            if candidate["depth"] != current["depth"]:
                return candidate["depth"] < current["depth"]
        elif v < 0: # losing  = prefer later (higher depth)
            if candidate["depth"] != current["depth"]:
                return candidate["depth"] > current["depth"]
        return candidate["avg"] > current["avg"]

    def isBetterForMin(self, candidate, current):
        if candidate["value"] != current["value"]:
            return candidate["value"] < current["value"]
        v = candidate["value"]
        if v < 0:   # winning = prefer sooner (lower depth)
            if candidate["depth"] != current["depth"]:
                return candidate["depth"] < current["depth"]
        elif v > 0: # losing  = prefer later (higher depth)
            if candidate["depth"] != current["depth"]:
                return candidate["depth"] > current["depth"]
        return candidate["avg"] < current["avg"]

    def minValue(self, state, depth=0):
        if self.terminal(state):
            v = self.value(state)
            return {"value": v, "depth": depth, "avg": float(v), "count": 1, "move": None}

        resActions = self.actions(state)
        player = resActions["player"]
        best = None
        total_sum = 0.0
        total_count = 0

        for move in resActions["listActions"]:
            res = self.maxValue(self.setMoveCopyState(state, move, player), depth + 1)
            # accumulate subtree avg across ALL children (not just best)
            total_sum += res["avg"] * res["count"]
            total_count += res["count"]
            candidate = {"value": res["value"], "depth": res["depth"],
                        "avg": res["avg"], "move": move}
            if best is None or self.isBetterForMin(candidate, best):
                best = candidate

        return {"value": best["value"], "depth": best["depth"],
                "avg": total_sum / total_count, "count": total_count, "move": best["move"]}

    def maxValue(self, state, depth=0):
        if self.terminal(state):
            v = self.value(state)
            return {"value": v, "depth": depth, "avg": float(v), "count": 1, "move": None}

        resActions = self.actions(state)
        player = resActions["player"]
        best = None
        total_sum = 0.0
        total_count = 0

        for move in resActions["listActions"]:
            res = self.minValue(self.setMoveCopyState(state, move, player), depth + 1)
            total_sum += res["avg"] * res["count"]
            total_count += res["count"]
            candidate = {"value": res["value"], "depth": res["depth"],
                        "avg": res["avg"], "move": move}
            if best is None or self.isBetterForMax(candidate, best):
                best = candidate

        return {"value": best["value"], "depth": best["depth"],
                "avg": total_sum / total_count, "count": total_count, "move": best["move"]}

    def setNextMove(self,state,role):
        if role != self.player(state): return None
        else:
            res = self.maxValue(state) if role=="x" else self.minValue(state)
            self.setMove(state,res["move"],role)





            
agent = TicTacToeAgent()

board = [
    [".", ".", "."],
    [".", ".", "."],
    [".", ".", "."]
]

while not agent.terminal(board):
    print()
    for row in board:
        print(" ".join(row))
    print()
    current = agent.player(board)
    if current == "x":
        print("Your turn (X)")
        row = int(input("Row (0-2): "))
        col = int(input("Col (0-2): "))

        if row < 0 or row > 2 or col < 0 or col > 2:
            print("Invalid position.")
            continue

        if board[row][col] != ".":
            print("Cell already occupied.")
            continue

        board[row][col] = "x"

    else:
        print("AI is thinking...")
        agent.setNextMove(board, "o")


print()
for row in board:
    print(" ".join(row))

winner = agent.winner(board)

if winner == "x":
    print("You win!")
elif winner == "o":
    print("AI wins!")
else:
    print("Draw!")
    

