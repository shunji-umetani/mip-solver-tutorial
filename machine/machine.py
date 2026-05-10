#!/opt/homebrew/bin/python3.11
# -*- coding: utf-8 -*-

# import modules
import sys, os, time, csv
from pyscipopt import Model, quicksum

# constant
TIME_LIMIT = 60

# read input data
input = open(sys.argv[1], 'r', encoding='utf-8-sig')
data = [_ for _ in csv.reader(input)]
input.close()
num = int(data[1][0])
job = set()
proc, wt, due = {}, {}, {}
big_M = 0
for line in data[3:]:
    job.add(line[0])
    proc[line[0]] = int(line[1])
    wt[line[0]] = int(line[2])
    due[line[0]] = int(line[3])
    big_M += int(line[1])

# set MIP model
start_time = time.time()
model = Model()
s,t = {},{}
for i in sorted(job):
    s[i] = model.addVar(vtype='C', name=f's({i})')
    t[i] = model.addVar(vtype='C', name=f't({i})')
x = {}
loop = [(i,j) for i in job for j in job if i != j]
for i,j in sorted(loop):
    x[(i,j)] = model.addVar(vtype='B', name=f'x({i},{j})')
for i,j in sorted(loop):
    model.addCons(s[i] - s[j] + big_M * x[(i,j)] <= big_M - proc[i], name=f'PREC({i},{j})')
    model.addCons(x[(i,j)] + x[(j,i)] == 1, name=f'ORDER({i},{j})')
for i in sorted(job):
    model.addCons(s[i] - t[i] <= due[i] - proc[i], name=f'TARD({i})')
model.setObjective(quicksum(wt[i] * t[i] for i in sorted(job)), sense='minimize')

# print LP data
base, ext = os.path.splitext(os.path.basename(sys.argv[1]))
model.writeProblem(filename=f'{base}.lp')

# solve MIP model
model.setParam('limits/time', TIME_LIMIT)
#model.setParam("display/verblevel", 0)  # silent mode
model.optimize()

# get solution
sol = {}
if model.getNSols() > 0:
    obj = model.getObjVal()
    for i in s:
        sol[i] = (model.getVal(s[i]), model.getVal(t[i]))
model.freeProb()  # clear model
end_time = time.time()

# print result
print('\ninstance')
print(f'N= {num}')
print('job\tproc\twt\tdue')
for id in sorted(job):
    print(f'{id}\t{proc[id]}\t{wt[id]}\t{due[id]}')
print('\nsolution')
print(f'obj= {obj}')
print(f'sol= {sorted(sol.items(), key=lambda x: x[1])}')

print(f'\nCPU Time:\t{end_time - start_time:<.3f} sec')
