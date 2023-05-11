import sys
import numpy as np

# required_edges = 100
# num = int(np.ceil(np.random.rand()**(1/64)*required_edges))
# idx = np.random.randint(0, num, 8)
# idx = np.sort(idx)
# print(idx, idx[1:6:2])

ls = [[(1, 2), (1, 2), (1, 2)], [(0, 0), (0, 0)]]

t = ls.copy()
for i in range(len(t)):
    t[i] = ls[i].copy()
    t[i][0] = (-1, -1)

print(ls, t)