from collections.abc import Callable, Iterable, Mapping
import sys
import time
import queue
import random
import multiprocessing
from typing import Any

TIME_BUFFER = 2.33
INT_MAX = (2**31)-1
CORE = 8

class Info():
    def __init__(self, arg: list[str]) -> None:
        start_time = time.time()
        file = sys.argv[1]
        termination = int(sys.argv[3])
        seed = int(sys.argv[5])
        random.seed(seed)
        
        content = []
        with open(file) as f:
            for line in f:
                line = line.strip()
                if line == 'END':
                    break
                content.append(line.split())

        n = int(content[1][2])
        s = int(content[2][2])-1
        k = int(content[3][3])
        cap = int(content[6][2])
        table = content[9:]
        
        edge = [[] for _ in range(n)]
        req = []
        for line in table:
            u, v, w, d = int(line[0])-1, int(line[1])-1, int(line[2]), int(line[3])
            edge[u].append((v, w))
            edge[v].append((u, w))
            if d != 0:
                req.append((u, v, w, d))
        
        dis = [[INT_MAX for j in range(n)] for i in range(n)]
        for s in range(n):
            dis[s][s] = 0
            q = queue.PriorityQueue()
            q.put((0, s))
            while not q.empty():
                top = q.get()
                u, d = top[1], top[0]
                if d != dis[s][u]:
                    continue
                for e in edge[u]:
                    v, w = e[0], e[1]
                    if dis[s][u]+w < dis[s][v]:
                        dis[s][v] = dis[s][u]+w
                        q.put((dis[s][v], v))
        
        self.n = n
        self.s = s
        self.k = k
        self.cap = cap
        self.dis = dis
        self.req = req
        
class Solution():
    def __init__(self, info: Info, routes: list[tuple]) -> None:
        self.info = info
        self.routes = routes
        self.refresh()
    
    def __lt__(self, another) -> bool:
        return self.cost < another.cost
    
    def __str__(self) -> str:
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
    
    def refresh(self) -> None:
        self.cost = 0
        for route in self.routes:
            u, w = self.info.s, 0
            for e in route:
                w += self.info.req[e[2]][3]
                if w > self.info.cap:
                    self.cost = INT_MAX
                    return
                self.cost += self.info.dis[u][e[0]]+self.info.req[e[2]][2]
                u = e[1]
            self.cost += self.info.dis[u][self.info.s]

    def mutate(self) -> None:
        # flip, swap, cross, split, merge
        return NotImplementedError

class Population():
    def __init__(self, info: Info, hyper: tuple, pool: list[Solution]) -> None:
        return NotImplementedError
    
    def reproduce(self):
        return NotImplementedError

    def select(self):
        return NotImplementedError

class MyProcess(multiprocessing.Process):
    def __init__(self, info: Info, hyper: tuple, pool: list[Solution]) -> None:
        super(MyProcess, self).__init__()
        self.population = Population(info, hyper, pool)


if __name__ == '__main__':
    info = Info(sys.argv)
    
    '''
    To Be Implemented
    '''
    
    q = multiprocessing.Manager().Queue()
    process = []
    for i in range(CORE):
        process.append(MyProcess(info, hyper[i], pool[i]))
        process[i].start()
    for proc in process:
        proc.join()
    while not q.empty():
        sol = q.get()
        if sol.cost < best.cost:
            best = sol
    print(best)
