import subprocess
import time

solver = 'CARP_solver.py'
data = 'sample.dat'
termination = 5
seed = int(time.time())

solution = subprocess.run(['python', solver, data, '-t', str(termination), '-s', str(seed)])

# solution = subprocess.run(['python', solver, data, '-t', str(termination), '-s', str(seed)], capture_output=True, text=True)

# print(solution.stdout)
