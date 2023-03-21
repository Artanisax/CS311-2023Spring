import numpy as np
import random
import time

COLOR_BLACK=-1
COLOR_WHITE=1
COLOR_NONE=0
DIRECTION = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
# random.seed(0)
#don't change the class name
class AI(object):
    #chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        # You are white or black
        self.color = color
        if (color == COLOR_WHITE):
            self.enemy = COLOR_BLACK
        else:
            self.enemy = COLOR_WHITE
        # the max time you should use, your algorithm's run time must not exceed the time limit.
        self.time_out = time_out
        # You need to add your decision to your candidate_list. The system will get the end of your candidate_list as your decision.
        self.candidate_list = []

    # The input is the current chessboard. Chessboard is a numpy array.
    def go(self, chessboard):
        # Clear candidate_list, must do this step
        self.candidate_list.clear()
        #==================================================================
        # Write your algorithm here
        # Here is the simplest sample:Random decision
        idx = np.where(chessboard == COLOR_NONE)
        idx = list(zip(idx[0], idx[1]))
        for pos in idx:
            if check(self, chessboard, pos):
                self.candidate_list.append(pos)
        if (len(self.candidate_list) == 0):
            return
        self.candidate_list.append(self.candidate_list[random.randrange(len(self.candidate_list))])
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
    
    def check(self, chessboard, pos):
        for direction in DIRECTION:
            x, y = pos[0]+direction[0], pos[1]+direction[1]
            while (x > 0 and x < 7 and y > 0 and y < 7 and chessboard[x][y] == self.enemy):
                x += direction[0]
                y += direction[1]
            if (x >= 0 and x < 8 and y >= 0 and y < 8 and chessboard[x][y] == self.color):
                return True
        return False