import numpy as np
import math
import random
import time

COLOR_BLACK=-1
COLOR_WHITE=1
COLOR_NONE=0
BUFFER_TIME = 0.3

AP = 16
MID = 30
corner = [(0, 0), (0, 7), (7, 0), (7, 7)]
DIRECTION = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
VALUE = [[-500, 49, -5, 7, 7, -5, 49, -500],
         [49, 36, 2, 4, 4, 2, 36, 49],
         [-5, 2, 1, 6, 6, 1, 2, -5],
         [7, 4, 6, 5, 5, 6, 4, 7],
         [7, 4, 6, 5, 5, 6, 4, 7],
         [-5, 2, 1, 6, 6, 1, 2, -5],
         [49, 36, 2, 4, 4, 2, 36, 49],
         [-500, 49, 2, 7, 7, 2, 49, -500]]

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

    def place(self, chessboard, pos, color):
        newboard = chessboard.copy()
        for direction in DIRECTION:
            x, y = pos[0]+direction[0], pos[1]+direction[1]
            if (x < 0 or x >= 8 or y < 0 or y >= 8 or chessboard[x][y] != -color):
                continue
            while (x >= 0 and x < 8 and y >= 0 and y < 8 and chessboard[x][y] == -color):
                x += direction[0]
                y += direction[1]
            if (x >= 0 and x < 8 and y >= 0 and y < 8 and chessboard[x][y] == color):
                while (x != pos[0] or y != pos[1]):
                    x -= direction[0]
                    y -= direction[1]
                    newboard[x][y] = color
        return newboard
    
    def judge(self, chessboard):
        x, y = np.count_nonzero(chessboard == self.color), np.count_nonzero(chessboard == -self.color)
        return y-x
    
    def evaluate(self, chessboard):
        table = VALUE.copy()
        idx = np.where(chessboard == self.color)
        idx = list(zip(idx[0], idx[1]))
        # for pos in corner:
        #     if pos in idx:
        #         for i in range(8):
        #             x, y = pos[0]+DIRECTION[i][0], pos[1]+DIRECTION[i][1]
        #             if x >= 0 and x < 8 and y >= 0 and y < 8:
        #                 table[x][y] = -table[x][y]
        ret = 0
        for pos in idx:
            ret += table[pos[0]][pos[1]]
        idx = np.where(chessboard == -self.color)
        idx = list(zip(idx[0], idx[1]))
        for pos in idx:
            ret -= table[pos[0]][pos[1]]
        return ret
    
    def stable(self, chessboard):
        
        return
    
    def beginning(self, chessboard):
        def min_search(step, board, alpha, beta):
            nonlocal limit
            if time.time()-self.start > self.time_out-BUFFER_TIME:
                return None, None, True
            candidate = self.get_candidate(board, -self.color)
            if step == limit:
                return None, self.evaluate(board)-AP*len(candidate)*abs(MID-np.count_nonzero(board != COLOR_NONE))/MID, False
            if not candidate:
                return None, self.judge(board)+self.judge(board)*64, False
            value, move = math.inf, None
            for pos in candidate:
                _, temp, flag = max_search(step+1, self.place(board, pos, -self.color), alpha, beta)
                if flag:
                    return None, None, True
                if temp <= alpha:
                    return None, temp, False
                if temp < value:
                    move = pos
                    value = temp
                    beta = temp
            return move, value, False
        
        def max_search(step, board, alpha, beta):
            nonlocal limit
            if time.time()-self.start > self.time_out-BUFFER_TIME:
                return None, None, True
            candidate = self.get_candidate(board, self.color)
            if step == limit:
                return None, self.evaluate(board)+AP*len(candidate)*abs(MID-np.count_nonzero(board != COLOR_NONE))/MID, False
            if not candidate:
                return None, self.evaluate(board)+self.judge(board)*64, False
            value, move = -math.inf, None
            for pos in candidate:
                _, temp, flag = min_search(step+1, self.place(board, pos, self.color), alpha, beta)
                if flag:
                    return None, None, True
                if temp >= beta:
                    return None, temp, False
                if temp > value:
                    move = pos
                    value = temp
                    alpha = temp
            return move, value, False
        
        move, limit, flag = None, 3, False
        while True:
            temp, _, flag = max_search(0, chessboard, -math.inf, math.inf)
            if flag:
                break
            move = temp
            # print(limit, move)
            limit += 1
        return move
    
    def ending(self, chessboard):
        def min_search(step, board, alpha, beta):
            nonlocal limit
            if time.time()-self.start > self.time_out-BUFFER_TIME:
                return None, None, True
            if step == limit:
                return None, self.judge(board), False
            candidate = self.get_candidate(board, self.color)
            if not candidate:
                return None, self.judge(board), False
            value, move = math.inf, None
            for pos in candidate:
                _, temp, flag = max_search(step+1, self.place(board, pos, -self.color), alpha, beta)
                if flag:
                    return None, None, True
                if temp <= alpha:
                    return None, temp, False
                if temp < value:
                    move = pos
                    value = temp
                    beta = temp
            return move, value, False
        
        def max_search(step, board, alpha, beta):
            nonlocal limit
            if time.time()-self.start > self.time_out-BUFFER_TIME:
                return None, None, True
            if step == limit:
                return None, self.judge(board), False
            candidate = self.get_candidate(board, self.color)
            if not candidate:
                return None, self.judge(board), False
            value, move = -math.inf, None
            for pos in candidate:
                _, temp, flag = min_search(step+1, self.place(board, pos, self.color), alpha, beta)
                if flag:
                    return None, None, True
                if temp >= beta:
                    return None, temp, False
                if temp > value:
                    move = pos
                    value = temp
                    alpha = temp
            return move, value, False
        
        move, limit, flag = None, 3, False
        while True:
            temp, _, flag = max_search(0, chessboard, -math.inf, math.inf)
            if flag:
                break
            move = temp
            # print(limit, move)
            limit += 1
        return move
    
    def go(self, chessboard):
        self.start = time.time()
        self.candidate_list.clear()
        self.candidate_list = self.get_candidate(chessboard, self.color)
        if (len(self.candidate_list) == 0):
            return
        self.candidate_list.append(self.candidate_list[random.randrange(len(self.candidate_list))])
        if np.count_nonzero(chessboard == COLOR_NONE) <= 4:
            move = self.ending(chessboard)
        else:
            move = self.beginning(chessboard)
        self.candidate_list.append(move)

# ib = np.array([[0,0,0,0,0,0,0,0],
#                [0,1,0,0,0,0,0,0],
#                [0,-1,1,1,1,0,0,0],
#                [0,0,0,1,-1,0,0,0],
#                [0,0,0,-1,-1,-1,0,0],
#                [0,0,0,0,-1,1,0,0],
#                [0,0,0,0,0,0,0,0],
#                [0,0,0,0,0,0,0,0]])

# print(ib)

# ai = AI(ib, COLOR_BLACK, 5)

# ai.go(ib)

# print(ai.candidate_list)