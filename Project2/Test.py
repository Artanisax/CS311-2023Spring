import subprocess

solver = 'CARP_solver.py'
data = 'sample.dat'
termination = 5
seed = 114514

solution = subprocess.run(['python', solver, data, '-t', str(termination), '-s', str(seed)], capture_output=True, text=True)

print(solution.stdout)
