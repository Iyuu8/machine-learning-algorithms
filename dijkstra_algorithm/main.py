import heapq


# the following implementation extracts the shortest path from a starting node A to every other node in the graph, if we care about one single node only we can stop the loop once our exact node is marked as visited, because in that case the distance to that node is final
graph = {
    "A" : [(5,"D"),(2,"F"),(1,"B")],
    "B" : [(1,"A"),(7,"G")],
    "C" : [(3,"D"),(2,"E"),(3,"H"),(4,"G")],
    "D" : [(3,"C"),(5,"A")],
    "E" : [(4,"F"),(2,"C")],
    "F" : [(2,"A"),(8,"G"),(4,"E")],
    "G" : [(4,"C"),(8,"F"),(7,"B")],
    "H" : [(3,"C")],
}

start = "A"
end = "H"
dist = []
bestWeights = dict()
for node in graph:
    bestWeights[node] = 0 if node == start else float("inf")
    dist.append((0 if node == start else float("inf"),node,))
heapq.heapify(dist)
visited = set()
tree = dict()

while len(dist) > 0:
    if not dist[0][1] in visited:
        node = heapq.heappop(dist)
        visited.add(node[1])
        for connectedEdge in graph[node[1]]:
            if connectedEdge[1] not in visited:
                newDist = bestWeights[node[1]] + connectedEdge[0]
                if newDist < bestWeights[connectedEdge[1]]: 
                    heapq.heappush(dist,(newDist,connectedEdge[1]))
                    bestWeights[connectedEdge[1]] = newDist
                    tree[connectedEdge[1]] = node[1]
    else: heapq.heappop(dist)

path = []
node = end
while True:
    path.append(node)
    node = tree[node] if node in tree else None
    if not node: break

path.reverse()

print(dist)
print(bestWeights)
print(path)


