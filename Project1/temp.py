import math
import numpy

import Main

a = math.inf

ib = np.array([[0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,-1,1,1,1,0,0],
               [0,0,-1,1,-1,1,0,0],
               [0,0,1,1,1,1,0,0],
               [0,0,-1,0,0,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0]])

ai = AI(ib, COLOR_BLACK, 3)
ai.go(ib)
print(ai.candidate_list)