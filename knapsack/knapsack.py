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
num, cap = int(data[1][0]), float(data[1][1])
val, wt = [], []
for line in data[3:]:
    val.append(float(line[0]))
    wt.append(float(line[1]))

# set MIP model
start_time = time.time()
model = Model()
x = {}
for i in range(num):
    x[i] = model.addVar(vtype='B', name=f'x({i})')
model.addCons(quicksum(wt[i] * x[i] for i in range(num)) <= cap)
model.setObjective(quicksum(val[i] * x[i] for i in range(num)), sense='maximize')

# print LP data
base, ext = os.path.splitext(os.path.basename(sys.argv[1]))
model.writeProblem(filename=f'{base}.lp')

# solve MIP model
model.setParam('limits/time', TIME_LIMIT)
#model.setParam("display/verblevel", 0)  # silent mode
model.optimize()

# get solution
sol = set()
if model.getNSols() > 0:
    obj = model.getObjVal()
    for i in range(num):
        if model.getVal(x[i]) > 0.5: sol.add(i)
model.freeProb()  # clear model
end_time = time.time()

# print result
print('\ninstance')
print(f'N= {num}\nC= {cap}')
print('val\twt')
for i in range(num): print(f'{val[i]}\t{wt[i]}')
print('\nsolution')
print(f'obj= {obj}')
print(f'items= {sol}')

print(f'\nCPU Time:\t{end_time - start_time:<.3f} sec')
