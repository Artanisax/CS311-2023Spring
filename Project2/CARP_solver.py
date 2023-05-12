import sys
import time
import queue
import random
import threading

import numpy as np

start_time = time.time()
def runtime():
    return time.time()-start_time


TIME_BUFFER = 0.7
INT_MAX = (2**31)-1

# input
data = sys.argv[1]
termination = int(sys.argv[3])
seed = int(sys.argv[5])

random.seed(seed)
np.random.seed(random.randint(0, INT_MAX))

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

INIT_LIFE = max(int(np.sqrt(required_edges)), 2)
class Solution:
    def __init__(self):
        self.routes = []
        self.cost = INT_MAX
        self.life = INIT_LIFE
    
    def __lt__(self, other):
        if self.life:
            return not other.life or self.cost < other.cost
        return False
    
    def __str__(self):
        s = 's '
        for route in self.routes:
            if not route:
                continue
            s += '0,'
            for e in route:
                s += '('+str(e[0]+1)+','+str(e[1]+1)+'),'
            s += '0,'
        s = s.rstrip(',')
        s += '\nq '+str(self.cost)
        return s

    def copy(self):
        solution = Solution()
        solution.routes = self.routes.copy()
        for i in range(len(self.routes)):
            solution.routes[i] = self.routes[i].copy()
        solution.cost = self.cost
        return solution

    def refresh(self):
        self.cost = 0
        for route in self.routes:
            u, w = depot, 0
            for e in route:
                w += requirements[e[2]][3]
                if w > capacity:
                    self.cost = INT_MAX
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
            u, v = e[:2]
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
            u, v = e[:2]
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
            u, v = e[:2]
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
            u, v = e[:2]
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
            u, v = e[:2]
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
            u, v = e[:2]
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
        if allowance > capacity/4:
            ret = get_next(last, allowance, rest, 10, cost)
        else:
            ret = get_next(last, allowance, rest, 11, cost)
    elif rule == 14:
        for id in rest:
            e = requirements[id]
            if e[3] > allowance:
                continue
            u, v = e[0:2]
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
            u, v = e[0:2]
            if dist[last][v] < dist[last][u]:
                u, v = v, u
            if not ret or e[3] > requirements[ret[2]][3] \
            or (e[3] == requirements[ret[2]][3] and dist[last][u] < dist[last][ret[0]]):
                ret = (u, v, id)
    return ret

def path_scan(rule):
    solution = Solution()
    rest = [i for i in range(required_edges)]
    while True:
        route = []
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
        solution.routes.append(route)
        if not rest:
            break
    if rest:
        return None
    solution.refresh()
    return solution

best = Solution()
init_pool = []
for rule in range(16):
    solution = path_scan(rule)
    init_pool.append(solution)
    if solution.cost < best.cost:
        best = solution

MUTATION_TYPE = 5
class Population():
    def __init__(self, K, pool, rates, micro=1):
        self.K = K
        self.pool = pool
        self.backup = pool.copy()
        self.rates = rates
        self.micro = micro
        self.best = Solution()
    
    def restart(self):
        self.pool = self.backup
    
    def generate(self, n):  # random solution (unguaranteed)
        ret = []
        for _ in range(n):
            solution = Solution()
            edges = []
            idx = np.random.randint(0, len(self.pool))
            num = min(required_edges, len(self.pool[idx].routes))
            for i in range(required_edges):
                u, v = requirements[i][0], requirements[i][1]
                if np.random.rand() < 0.5:
                    u, v = v, u
                edges.append((u, v, i))
            random.shuffle(edges)                   # randomly arrage the edges 
            idx = np.random.randint(0, required_edges, num-1)  # chop the routes
            idx = np.sort(idx)
            l = 0
            for i in range(num-1):
                solution.routes.append(edges[l:idx[i]])
                l = idx[i]
            solution.routes.append(edges[l:required_edges])
            solution.refresh()
            ret.append(solution)
        return ret

    def mutate(self, solution, type):
        if np.random.rand() > self.rates[type]*self.micro:
            return 0
        if type == 0:  # reverse a single edge
            idx = np.random.randint(0, len(solution.routes))
            route = solution.routes[idx]
            if not route:
                return 0
            idx = np.random.randint(0, len(route))
            edge = route[idx]
            route[idx] = (edge[1], edge[0], edge[2])
        elif type == 1:  # swap 2 edges in one route
            idx = np.random.randint(0, len(solution.routes))
            route = solution.routes[idx]
            if len(route) < 2:
                return 0
            idx = np.random.randint(0, len(route), 2)
            route[idx[0]], route[idx[1]] = route[idx[1]], route[idx[0]]
        elif type == 2:  # move one edge from a route to another
            routes = solution.routes
            idx = np.random.randint(0, len(routes), 2)
            route = (routes[idx[0]], routes[idx[1]])
            if not route[0]:
                return 0
            idx = np.random.randint(0, len(route[0]))
            edge = route[0][idx]
            route[0].pop(idx)
            idx = np.random.randint(0, len(route[1])+1)
            route[1].insert(idx, edge)
        elif type == 3:  # swap 2 edges from different routes
            routes = solution.routes
            idx = np.random.randint(0, len(routes), 2)
            route = (routes[idx[0]], routes[idx[1]])
            if not route[0] or not route[1]:
                return 0
            idx = (np.random.randint(0, len(route[0])), 
                   np.random.randint(0, len(route[1])))
            route[0][idx[0]], route[1][idx[1]] = route[1][idx[1]], route[0][idx[0]]
        elif type == 4:  # add an empty route
            solution.routes.append([])
        solution.refresh()
        return 1

    def reproduce(self, death_rate):
        old_pool = self.pool.copy()
        for parent in old_pool:
            if not parent.life:
                continue
            for _ in range(2):
                child = parent.copy()
                t = np.random.randint(1, 3)
                for _ in range(t):
                    self.mutate(child, np.random.randint(0, MUTATION_TYPE))
                if child.cost == INT_MAX and np.random.rand() < death_rate:
                    continue
                child.refresh()
                if child.cost < self.best.cost:
                    self.best = child
                self.pool.append(child)
            parent.life -= 1
            
    def selection(self):
        if len(self.pool) < self.K:
            return
        self.pool.sort()
        self.pool = self.pool[:28]
    
p = Population(4096, init_pool, (0.75, 0.8, 0.7, 0.6, 0.2))

# single-thread
while termination-(runtime()) > TIME_BUFFER:
    p.reproduce(0.12)
    p.selection()

if p.best.cost < best.cost:
    best = p.best

# muti-thread
thread_pool = [init_pool[0:2],
               init_pool[2:3],
               init_pool[3:5],
               init_pool[5:8],
               init_pool[8:10],
               init_pool[10:12],
               init_pool[12:14],
               init_pool[14:16]]
thread_rates = [(0.75, 0.8, 0.7, 0.6, 0.2),
                (0.75, 0.8, 0.7, 0.6, 0.2),
                (0.75, 0.8, 0.7, 0.6, 0.2),
                (0.75, 0.8, 0.7, 0.6, 0.2),
                (0.75, 0.8, 0.7, 0.6, 0.2),
                (0.75, 0.8, 0.7, 0.6, 0.2),
                (0.75, 0.8, 0.7, 0.6, 0.2),
                (0.75, 0.8, 0.7, 0.6, 0.2)]
thread_death_rate = [0.12,
                     0.12,
                     0.12,
                     0.12,
                     0.12,
                     0.12,
                     0.12,
                     0.49]
thread_K = [4096,
            4096,
            4096,
            4096,
            4096,
            4096,
            4096,
            4096]
class MyThread(threading.Thread):
    def __init__(self, K, pool, rates, death_rate, micro=1):
        threading.Thread.__init__(self)
        self.death_rate = death_rate
        self.population = Population(K, pool, rates, micro)
        
    def run(self):
        while termination-(runtime()) > TIME_BUFFER:
            self.population.reproduce(self.death_rate)
            self.population.selection()
            
threads = []
for i in range(8):
    threads.append(MyThread(thread_K[i],
                            thread_pool[i],
                            thread_rates[i],
                            thread_death_rate[i]))
    threads[i].start()

# for t in threads:
#     t.join()
#     t_best = t.population.best
#     if t_best.cost < best.cost:
#         best = t_best
        
print(best)

print(runtime())