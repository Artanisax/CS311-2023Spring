import numpy as np
import math
import random
import time

COLOR_BLACK=-1
COLOR_WHITE=1
COLOR_NONE=0
BUFFER_TIME = 0.3
DIRECTION = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
VALUE = [[1, 8, 3, 7, 7, 3, 8, 1],
         [8, 3, 2, 5, 5, 2, 3, 8],
         [3, 2, 6, 6, 6, 6, 2, 3],
         [7, 5, 6, 4, 4, 6, 5, 7],
         [7, 5, 6, 4, 4, 6, 5, 7],
         [3, 2, 6, 6, 6, 6, 2, 3],
         [8, 3, 2, 5, 5, 2, 3, 8],
         [1, 8, 3, 7, 7, 3, 8, 1]]

random.seed(time.time())
#don't change the class name
class AI(object):
    #chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        # You are white or black
        self.color = color
        # the max time you should use, your algorithm's run time must not exceed the time limit.
        self.time_out = time_out
        # You need to add your decision to your candidate_list. The system will get the end of your candidate_list as your decision.
        self.candidate_list = []
    
    def check(self, chessboard, pos, color):
        for direction in DIRECTION:
            x, y = pos[0]+direction[0], pos[1]+direction[1]
            if (x < 0 or x >= 8 or y < 0 or y >= 8 or chessboard[x][y] != -color):
                continue
            while (x >= 0 and x < 8 and y >= 0 and y < 8 and chessboard[x][y] == -color):
                x += direction[0]
                y += direction[1]
            if (x >= 0 and x < 8 and y >= 0 and y < 8 and chessboard[x][y] == color):
                return True
        return False
    
    def get_candidate(self, chessboard, color):
        candidate = []
        idx = np.where(chessboard == COLOR_NONE)
        idx = list(zip(idx[0], idx[1]))
        for pos in idx:
            if self.check(chessboard, pos, color):
                candidate.append(pos)
        return candidate
    
    def judge(self, chessboard):
        return 0
        x, y = np.count_nonzero(chessboard == self.color), np.count_nonzero(chessboard == -self.color)
        if x < y:
            return 36
        elif x == y:
            return 0
        else:
            return -36
    
    def evaluate(self, chessboard):
        return VALUE
        
    def alphabeta_pruning(self, chessboard):
        def min_search(step, alpha, beta):
            if time.time()-self.start > self.time_out-BUFFER_TIME:
                return None, None, True
            nonlocal chessboard, limit
            if step == limit:
                return None, 0, False
            candidate = self.get_candidate(chessboard, -self.color)
            if not candidate:
                return None, self.judge(chessboard), False
            valueboard = self.evaluate(chessboard)
            value, move = math.inf, None
            for pos in candidate:
                chessboard[pos[0]][pos[1]] = -self.color
                _, temp, flag = max_search(step, alpha, beta)
                chessboard[pos[0]][pos[1]] = COLOR_NONE
                if flag:
                    return None, None, True
                temp += valueboard[pos[0]][pos[1]]
                if temp <= alpha:
                    return None, temp, False
                if temp < value:
                    move = pos
                    value = temp
                    beta = temp
            return move, value, False
        
        def max_search(step, alpha, beta):
            if time.time()-self.start > self.time_out-BUFFER_TIME:
                return None, None, True
            nonlocal chessboard
            candidate = self.get_candidate(chessboard, self.color)
            if not candidate:
                return None, self.judge(chessboard), False
            valueboard = self.evaluate(chessboard)
            value, move = -math.inf, None
            for pos in candidate:
                chessboard[pos[0]][pos[1]] = self.color
                _, temp, flag = min_search(step+1, alpha, beta)
                chessboard[pos[0]][pos[1]] = COLOR_NONE
                if flag:
                    return None, None, True
                temp += valueboard[pos[0]][pos[1]]
                if temp >= beta:
                    return None, temp, False
                if temp > value:
                    move = pos
                    value = temp
                    alpha = temp
            return move, value, False
        move, limit, flag = None, 1, False
        while not flag:
            temp, _, flag = max_search(0, -math.inf, math.inf)
            if not flag:
                move = temp
            limit += 1
        return move
    
    # The input is the current chessboard. Chessboard is a numpy array.
    def go(self, chessboard):
        self.start = time.time()
        # Clear candidate_list, must do this step
        self.candidate_list.clear()
        #==================================================================
        # Write your algorithm here
        # Here is the simplest sample:Random decision
        self.candidate_list = self.get_candidate(chessboard, self.color)
        if (len(self.candidate_list) == 0):
            return
        # self.candidate_list.append(self.candidate_list[random.randrange(len(self.candidate_list))])
        #==============Find new pos========================================
        # Make sure that the position of your decision on the chess board is empty.
        # If not, the system will return error.
        # Add your decision into candidate_list, Records the chessboard
        # You need to add all the positions which are valid
        # candidate_list example: [(3,3),(4,4)]
        # You need append your decision at the end of the candidate_list,
        # candidate_list example: [(3,3),(4,4),(4,4)]
        # we will pick the last element of the candidate_list as the position you choose.
        # In above example, we will pick (4,4) as your decision.
        # If there is no valid position, you must return an empty
        self.candidate_list.append(self.alphabeta_pruning(chessboard))
        # print(time.time()-self.start)

# ib = np.array([[0,0,0,0,0,0,0,0],
#                [0,0,0,0,0,0,0,0],
#                [0,0,0,0,0,0,0,0],
#                [0,0,0,-1,1,0,0,0],
#                [0,0,0,1,-1,0,0,0],
#                [0,0,0,0,0,0,0,0],
#                [0,0,0,0,0,0,0,0],
#                [0,0,0,0,0,0,0,0]])

# ai = AI(ib, COLOR_BLACK, 5)
# ai.go(ib)
# print(ai.candidate_list)