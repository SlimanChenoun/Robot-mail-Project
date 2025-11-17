import heapq

def astar(start, goal, neighbors, cost, heuristic):
    open_heap = [(0, start)]
    came_from = {start: None}
    g = {start: 0}

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current == goal:
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for nxt in neighbors(current):
            tentative_g = g[current] + cost(current, nxt)
            if tentative_g < g.get(nxt, float('inf')):
                came_from[nxt] = current
                g[nxt] = tentative_g
                f = tentative_g + heuristic(nxt, goal)
                heapq.heappush(open_heap, (f, nxt))

    return None


# --- Grille : 0 = libre, 1 = obstacle ---
grid = [
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0]
]

# Fonction pour obtenir les voisins valides
def voisins(c):
    (x, y) = c
    for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < 3 and 0 <= ny < 4 and grid[nx][ny] == 0:
            yield (nx, ny)

# Heuristique : distance de Manhattan
def h(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Coût uniforme : chaque déplacement vaut 1
def cost(a, b):
    return 1

# Exécution de A*
chemin = astar((0, 0), (2, 3), voisins, cost, h)
print("Chemin trouvé :", chemin)
