import sys
import numpy as np

# required_edges = 100
# num = int(np.ceil(np.random.rand()**(1/64)*required_edges))
# idx = np.random.randint(0, num, 8)
# idx = np.sort(idx)
# print(idx, idx[1:6:2])

edges = np.array([(0,0) for _ in range(10)])

edges[0] = (1, 1)
edges[2] = (0, 3)

print(edges)