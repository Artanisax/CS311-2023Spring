import sys
import time
import queue
import random
import numpy as np

TIME_BUFFER = 1
start_time = time.time()

# input
data = sys.argv[1]
termination = int(sys.argv[3])
seed = int(sys.argv[5])

random.seed(seed)

content = []
with open(data) as f:
    for line in f:
        line = line.strip()
        if line == 'END':
            break
        content.append(line.split())

# decode
name = content[0][2]
vertices = int(content[1][2])
depot = int(content[2][2])
required_edges = int(content[3][3])
non_required_edges = int(content[4][3])
vehicles = int(content[5][2])
capacity = float(content[6][2])
total = float(content[7][6])
table = content[9:]

edge = [[] for _ in range(vertices)]
requirements = []
for line in table:
    u, v, w, d = int(line[0])-1, int(line[1])-1, float(line[2]), float(line[3])
    edge[u].append((v, w))
    edge[v].append((u, w))
    if d != 0:
        requirements.append((u, v, w, d))
requirements = np.array(requirements)

class Solution:
    def __init__(self):
        self.routes = [[] for _ in range(vehicles)]
        self.cost = np.inf
    
    def __str__(self):
        s = 's '
        for route in self.routes:
            s += '0,'
            for e in route:
                s += '('+str(e[0])+str(e[1])+')'
            s += '0,'
        s.rstrip(',')
        s += '\nq '+str(self.cost)
        return s

    def update(self):
        self.cost = 0
        for route in self.routes:
            w = 0
            for e in route:
                w += requirements[e[2]][2]
                if w > capacity:
                    self.cost = np.inf
                    return
            self.cost += w
        return NotImplementedError

# SP
dist = np.full((vertices, vertices), np.inf)
for s in range(vertices):
    dist[s][s] = 0
    q = queue.PriorityQueue()
    q.put((0, s))
    while not q.empty():
        top = q.get()
        u, d = top[1], top[0]
        if d != dist[s][u]:
            continue
        for e in edge[u]:
            v, w = e[0], e[1]
            if dist[s][u]+w < dist[s][v]:
                dist[s][v] = dist[s][u]+w
                q.put((dist[s][v], v))

# stochastic solution/ initialize pool


def get_next(u, w, rest, rule):
    if not rest:
        return None
    if rule == 0:
        return 
    
    return NotImplementedError

def path_scan(rule):
    solution = Solution()
    rest = [i for i in range(required_edges)]
    for route in solution.routes:
        u, w = 0, 0
        while True:
            e = get_next(u, w, rest, rule)
            if not e:
                break
            w += requirements[e[2]][2]
            route.append(e)
            rest.pop(e)
    solution.update()
    return solution

best = Solution()
pool = []
for rule in range(5):
    solution = path_scan(rule)
    pool.append(path_scan(rule))
    if solution.cost > best.cost:
        best = solution

def rand_init():
    
    return NotImplementedError

def mutex(solution, type):
    
    return NotImplementedError

def reproduce(death_rate, survive_rate):
    
    return NotImplementedError

def selection(disaster_rate, survive_rate, K):
    
    return NotImplementedError

print(best)