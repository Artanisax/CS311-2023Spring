import sys
import time
import queue
import random
import multiprocessing

import numpy as np

TIME_BUFFER = 2.33
INT_MAX = (2**31)-1
INIT_LIFE = 2
MUTATION_TYPE = 5

class Info:
    def __init__(self, start_time, termination, seed, depot, capacity, requirements, dist):
        self.start_time = start_time
        self.termination = termination
        self.seed = seed
        self.depot = depot
        self.capacity = capacity
        self.requirements = requirements
        self.dist = dist

class Solution:
    def __init__(self, info):
        self.routes = []
        self.cost = INT_MAX
        self.life = INIT_LIFE
        self.info = info
    
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
        solution = Solution(self.info)
        solution.routes = self.routes.copy()
        for i in range(len(self.routes)):
            solution.routes[i] = self.routes[i].copy()
        solution.cost = self.cost
        return solution

    def refresh(self):
        self.cost = 0
        for route in self.routes:
            u, w = self.info.depot, 0
            for e in route:
                w += self.info.requirements[e[2]][3]
                if w > self.info.capacity:
                    self.cost = INT_MAX
                    return
                self.cost += self.info.dist[u][e[0]]+self.info.requirements[e[2]][2]
                u = e[1]
            self.cost += self.info.dist[u][self.info.depot]

class Population:
    def __init__(self, info, K, pool, rates, death_rate, size, micro=1):
        self.K = K
        self.pool = pool
        self.backup = pool.copy()
        self.rates = rates
        self.death_rate = death_rate
        self.size= size
        self.micro = micro
        self.best = Solution(info)
    
    def restart(self):
        self.pool = self.backup

    def mutate(self, solution, type):
        if np.random.rand() > self.rates[type]**self.micro:
            return False
        if type == 0:  # reverse a single edge
            idx = np.random.randint(0, len(solution.routes))
            route = solution.routes[idx]
            if not route:
                return False
            idx = np.random.randint(0, len(route))
            edge = route[idx]
            route[idx] = (edge[1], edge[0], edge[2])
        elif type == 1:  # swap 2 edges in one route
            idx = np.random.randint(0, len(solution.routes))
            route = solution.routes[idx]
            if len(route) < 2:
                return False
            idx = np.random.randint(0, len(route), 2)
            route[idx[0]], route[idx[1]] = route[idx[1]], route[idx[0]]
        elif type == 2:  # move one edge from a route to another
            routes = solution.routes
            idx = np.random.randint(0, len(routes), 2)
            route = (routes[idx[0]], routes[idx[1]])
            if not route[0]:
                return False
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
                return False
            idx = (np.random.randint(0, len(route[0])), 
                   np.random.randint(0, len(route[1])))
            route[0][idx[0]], route[1][idx[1]] = route[1][idx[1]], route[0][idx[0]]
        elif type == 4:  # add an empty route
            solution.routes.append([])
        solution.refresh()
        return True

    def reproduce(self):
        old_pool = self.pool.copy()
        for parent in old_pool:
            # if not parent.life:
            #     continue
            for _ in range(4):
                child = parent.copy()
                flag = False
                for _ in range(1):
                    flag |= self.mutate(child, np.random.randint(0, MUTATION_TYPE-1))
                self.mutate(child, MUTATION_TYPE-1)
                # if not flag:
                #     self.mutate(child, np.random.randint(0, MUTATION_TYPE))
                if child.cost >= parent.cost and np.random.rand() < self.death_rate:
                    continue
                child.refresh()
                if child.cost < self.best.cost:
                    self.best = child
                self.pool.append(child)
            # parent.life -= 1
            # if parent.life:
            #     new_pool.append(parent)
            
    def selection(self):
        if len(self.pool) < self.K:
            return
        self.pool.sort()
        new_pool = self.pool[:int(2*self.size/3)]
        random.shuffle(self.pool)
        new_pool.extend(self.pool[:int(self.size/3)])
        self.pool = new_pool

# muti-process
class MyProcess(multiprocessing.Process):
    def __init__(self, info, K, pool, rates, death_rate, size, q, micro=1):
        super(MyProcess, self).__init__()
        self.info = info
        self.population = Population(info, K, pool, rates, death_rate, size, micro)
        self.q = q
        
    def run(self):
        random.seed(self.info.seed)
        np.random.seed(random.randint(0, INT_MAX))
        cnt = 0
        flag = 10
        while self.info.termination-(time.time()-self.info.start_time) > TIME_BUFFER:
            self.population.reproduce()
            self.population.selection()
            # self.population.micro *= 0.99
        #     cnt += 1
        #     if time.time()-self.info.start_time >= flag:
        #         print(self.name, cnt, self.population.best.cost)
        #         flag += 10
        # print(self.name, cnt, self.population.best.cost)
        self.q.put(self.population.best)

if __name__ == '__main__':
    start_time = time.time()
    
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
    info = Info(start_time, termination, seed, depot, capacity, requirements, dist)
    
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
        solution = Solution(info)
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

    best = Solution(info)
    init_pool = []
    for rule in range(16):
        solution = path_scan(rule)
        init_pool.append(solution)
        if solution.cost < best.cost:
            best = solution
    
    CORE = 8
    
    proc_K = [1024,
              1024,
              1024,
              1024,
              1024,
              1024,
              1024,
              1024]
    proc_pool = [init_pool,
                 init_pool,
                 init_pool,
                 init_pool[5:16],
                 init_pool[2:5],
                 init_pool[2:5],
                 init_pool[2:5],
                 init_pool]
    proc_rates = [(0.86, 0.8, 0.85, 0.9, 0.05),
                  (0.8, 0.9, 0.75, 0.65, 0.05),
                  (0.75, 0.9, 0.8, 0.7, 0.05),    # case 0
                  (0.75, 0.85, 0.75, 0.65, 0.05),
                  (0.75, 0.8, 0.85, 0.75, 0.05),
                  (0.75, 0.9, 0.85, 0.9, 0.05),
                  (0.8, 0.9, 0.75, 0.7, 0.05),
                  (0.93, 0.93, 0.93, 0.93, 0.07)]
    proc_death_rate = [0.96,
                       0.91,
                       0.92,
                       0.93,
                       0.94,
                       0.95,
                       0.96,
                       0.86]
    proc_size = [61,
                 78,
                 64,
                 69,
                 21,
                 44,
                 37,
                 49]
    q = multiprocessing.Manager().Queue()
    processes = []
    for i in range(CORE):
        processes.append(MyProcess(info,
                                   proc_K[i],
                                   proc_pool[i],
                                   proc_rates[i],
                                   proc_death_rate[i],
                                   proc_size[i],
                                   q
                                   ))
        processes[i].start()
    
    for proc in processes:
        proc.join()
    while not q.empty():
        solution = q.get()
        if solution.cost < best.cost:
            best = solution
            
    # single-process
    # p = Population(info, 4096, init_pool, (0.7, 0.9, 0.75, 0.7, 0.15), 0.12, 28)
    # while termination-(time.time()-start_time) > TIME_BUFFER:
    #     p.reproduce()
    #     p.selection()
    # if p.best.cost < best.cost:
    #     best = p.best
    
    print(best)

    # print(time.time()-start_time)