import sys
import time
import queue
import random
import threading

import numpy as np

start_time = time.time()

TIME_BUFFER = 1
INT_MAX = np.iinfo(int).max

input
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
depot = int(content[2][2])-1
required_edges = int(content[3][3])
non_required_edges = int(content[4][3])
vehicles = int(content[5][2])
capacity = int(content[6][2])
total = int(content[7][6])
table = content[9:]

edge = [[] for _ in range(vertices)]
requirements = []
for line in table:
    u, v, w, d = int(line[0])-1, int(line[1])-1, float(line[2]), float(line[3])
    edge[u].append((v, w))
    edge[v].append((u, w))
    if d != 0:
        requirements.append((u, v, w, d))
requirements = np.array(requirements, dtype=int)

class Solution:
    def __init__(self):
        self.routes = [[] for _ in range(vehicles)]
        self.cost = INT_MAX
    
    def __str__(self):
        s = 's '
        for route in self.routes:
            s += '0,'
            for e in route:
                s += '('+str(e[0]+1)+','+str(e[1]+1)+'),'
            s += '0,'
        s = s.rstrip(',')
        s += '\nq '+str(self.cost)
        return s

    def update(self):
        self.cost = 0
        for route in self.routes:
            u, w = depot, 0
            for e in route:
                w += requirements[e[2]][3]
                if w > capacity:
                    self.cost = np.inf
                    return
                self.cost += dist[u][e[0]]+requirements[e[2]][2]
                u = e[1]
            self.cost += dist[u][depot]

# SP
dist = np.full((vertices, vertices), INT_MAX)

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
def get_next(last, allowance, rest, rule, cost=-1):
    if not rest:
        return None
    ret = None
    if rule == 0:  # close
        for id in rest:
            e = requirements[id]
            if e[3] > allowance:
                continue
            u, v = e[0], e[1]
            if dist[last][v] < dist[last][u]:
                u, v = v, u
            if not ret or dist[last][u] < dist[last][ret[0]] \
            or (dist[last][ret[0]] == dist[last][u] and dist[v][depot] < dist[ret[1]][depot]):
                ret = (u, v, id)
    elif rule == 1:  # far
        for id in rest:
            e = requirements[id]
            if e[3] > allowance:
                continue
            u, v = e[0], e[1]
            if dist[last][v] < dist[last][u]:
                u, v = v, u
            if not ret or dist[last][u] < dist[last][ret[0]] \
            or (dist[last][ret[0]] == dist[last][u] and dist[v][depot] > dist[ret[1]][depot]):
                ret = (u, v, id)
    elif rule == 2:
        if allowance > capacity/2:
            ret = get_next(last, allowance, rest, 1)
        else:
            ret = get_next(last, allowance, rest, 0)
    elif rule == 3:
        if allowance > capacity*2/3 or allowance < capacity/3:
            ret = get_next(last, allowance, rest, 0)
        else:
            ret = get_next(last, allowance, rest, 1)
    elif rule == 4:
        if allowance > capacity*3/4 or allowance < capacity/4:
            ret = get_next(last, allowance, rest, 0)
        else:
            ret = get_next(last, allowance, rest, 1)
            
    elif rule == 5:  # close
        for id in rest:
            e = requirements[id]
            if e[3] > allowance:
                continue
            u, v = e[0], e[1]
            if dist[last][v] < dist[last][u]:
                u, v = v, u
            if not ret or dist[last][u] < dist[last][ret[0]] \
            or (dist[ret[1]][depot]-dist[v][depot] > dist[last][u]-dist[last][ret[0]]):
                ret = (u, v, id)
    elif rule == 6:  # far
        for id in rest:
            e = requirements[id]
            if e[3] > allowance:
                continue
            u, v = e[0], e[1]
            if dist[last][v] < dist[last][u]:
                u, v = v, u
            if not ret or dist[last][u] < dist[last][ret[0]] \
            or (dist[v][depot]-dist[ret[1]][depot] > dist[last][u]-dist[last][ret[0]]):
                ret = (u, v, id)
    elif rule == 7:
        if allowance > capacity/2:
            ret = get_next(last, allowance, rest, 6)
        else:
            ret = get_next(last, allowance, rest, 5)
    elif rule == 8:
        if allowance > capacity*2/3 or allowance < capacity/3:
            ret = get_next(last, allowance, rest, 5)
        else:
            ret = get_next(last, allowance, rest, 6)
    elif rule == 9:
        if allowance > capacity*3/4 or allowance < capacity/4:
            ret = get_next(last, allowance, rest, 5)
        else:
            ret = get_next(last, allowance, rest, 6)
    elif rule == 10:  # no way back
        d, p = capacity-allowance, 0
        for id in rest:
            e = requirements[id]
            if e[3] > allowance:
                continue
            u, v = e[0], e[1]
            if dist[last][v] < dist[last][u]:
                u, v = v, u
            if (d+e[3])/(cost+dist[last][u]+e[2]) > p:
                ret = (u, v, id)
    elif rule == 11:  # way back
        d, p = capacity-allowance, 0
        for id in rest:
            e = requirements[id]
            if e[3] > allowance:
                continue
            u, v = e[0], e[1]
            if dist[last][v]+dist[u][depot] < dist[last][u]+dist[v][depot]:
                u, v = v, u
            if (d+e[3])/(cost+dist[last][u]+e[2]+dist[v][depot]) > p:
                ret = (u, v, id)
    elif rule == 12:
        if allowance > capacity/2:
            ret = get_next(last, allowance, rest, 10, cost)
        else:
            ret = get_next(last, allowance, rest, 11, cost)
    elif rule == 13:
        if allowance > capacity/3:
            ret = get_next(last, allowance, rest, 10, cost)
        else:
            ret = get_next(last, allowance, rest, 11, cost)
    elif rule == 14:
        for id in rest:
            e = requirements[id]
            if e[3] > allowance:
                continue
            u, v = e[0], e[1]
            if dist[last][v] < dist[last][u]:
                u, v = v, u
            if not ret or e[3] < requirements[ret[2]][3] \
            or (e[3] == requirements[ret[2]][3] and dist[last][u] < dist[last][ret[0]]):
                ret = (u, v, id)
    elif rule == 15:
        for id in rest:
            e = requirements[id]
            if e[3] > allowance:
                continue
            u, v = e[0], e[1]
            if dist[last][v] < dist[last][u]:
                u, v = v, u
            if not ret or e[3] > requirements[ret[2]][3] \
            or (e[3] == requirements[ret[2]][3] and dist[last][u] < dist[last][ret[0]]):
                ret = (u, v, id)
    return ret

def path_scan(rule):
    solution = Solution()
    rest = [i for i in range(required_edges)]
    for route in solution.routes:
        u, w, c = 0, 0, 0
        while True:
            e = get_next(u, capacity-w, rest, rule, c)
            if not e:
                break
            w += requirements[e[2]][3]
            c += dist[u][e[0]]+requirements[e[2]][2]
            route.append(e)
            rest.remove(e[2])
            u = e[1]
    if rest:
        return None
    solution.update()
    return solution
best = Solution()
pool = []
for rule in range(16):
    solution = path_scan(rule)
    if not solution:
        continue
    pool.append(solution)
    if solution.cost < best.cost:
        best = solution

def generate(n):
    for _ in range(n):
        solution = Solution()
        
    return NotImplementedError

def mutex(solution, type):
    
    return NotImplementedError

def reproduce(death_rate, survive_rate):
    
    return NotImplementedError

def selection(disaster_rate, survive_rate, K):
    
    return NotImplementedError

class MyThread(threading.Thread):
    
    def __init__(self, id, micro, K, pool):
        self.id = id
        self.micro = micro
        self.K = K
        self.pool = pool
        
    def run(self):
        
        return NotImplementedError

print(best)