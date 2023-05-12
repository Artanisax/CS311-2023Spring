import subprocess
import time

solver = 'CARP_solver.py'
data = 'egl-e1-A.dat'
termination = 60
seed = 114514

solution = subprocess.run(['python', solver, data, '-t', str(termination), '-s', str(seed)])

# solution = subprocess.run(['python', solver, data, '-t', str(termination), '-s', str(seed)], capture_output=True, text=True)

# print(solution.stdout)
