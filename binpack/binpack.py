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
items = set()
wt = {}
for line in data[3:]:
    items.add(line[0])
    wt[line[0]] = float(line[1])

# set MIP model
start_time = time.time()
model = Model()
x = {}
loop = ((i,j) for i in range(num) for j in items)
for i,j in sorted(loop):
    x[(i,j)] = model.addVar(vtype='B', name=f'x({i},{j})')
y = {}
for i in range(num):
    y[i] = model.addVar(vtype='B', name=f'y({i})')
for i in range(num):
    model.addCons(quicksum(wt[j] * x[(i,j)] for j in sorted(items)) - cap * y[i] <= 0, name=f'BIN({i})')
for j in sorted(items):
    model.addCons(quicksum(x[(i,j)] for i in range(num)) == 1, name=f'ITEM({j})')
for i in range(num-1):
    model.addCons(y[i] - y[i+1] >= 0, name=f'ORDER({i},{i+1})')
model.setObjective(quicksum(y[i] for i in range(num)), sense='minimize')

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
    for i,j in x:
        if model.getVal(x[(i,j)]) > 0.5: sol[j] = i
model.freeProb()  # clear model
end_time = time.time()

# print result
print('\ninstance')
print(f'N= {num}\nC= {cap}')
print(f'id\twt')
for id in sorted(items):
    print(f'{id}\t{wt[id]}')
print('\nsolution')
print(f'obj= {obj}')
print(f'sol= {sorted(sol.items(), key=lambda x: x[1])}')

print(f'\nCPU Time:\t{end_time - start_time:<.3f} sec')
