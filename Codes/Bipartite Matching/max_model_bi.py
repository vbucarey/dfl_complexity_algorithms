# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 12:15:08 2024

@author: sophi
"""

import gurobipy as gp
from gurobipy import GRB
import load_data_bi as ld
    
def minmax(filename, omega, env):
    
    print('~'*20+'Min Max'+'~'*20)
    
    #Read file
    x, V, E, N, s, t, c, A, b, bT, d=ld.readfile(filename)
    w=d+1
    
    #----------Optimization Model----------
    model = gp.Model("Maximize V", env = env)
    model.setParam("DualReductions",0)
    model.setParam("Threads",1)
    
    #Read omega
    omega = {eval(key): value for key, value in omega.items()}
    
    #Variables
    v = [{(j,k): model.addVar(vtype = GRB.CONTINUOUS, lb = 0, ub = 1, name = "v[{}][({},{})]".format(i,j,k)) for (j,k) in E} for i in range(len(N))]
    ro = [{n: model.addVar(vtype = GRB.CONTINUOUS, lb=0,ub = GRB.INFINITY, name = "ro[{}][{}]".format(i,n)) for n in V} for i in range(len(N))]
    alph = [{(j,k): model.addVar(vtype = GRB.CONTINUOUS, lb = -GRB.INFINITY, ub = 0, name = "alph[{}][({},{})]".format(i,j,k)) for (j,k) in E} for i in range(len(N))]
    
    #Objective function
    mobj = (1/len(N))*gp.quicksum(c[i][a]*v[i][a] for i in range(len(N)) for a in E)
    model.setObjective(mobj, GRB.MAXIMIZE)
    
    #Constraints
    for n in V:
       model.addConstrs(gp.quicksum(A[n][a]*v[i][a] for a in E) >= b[0] for i in range(len(N)))
    
    for i in range(len(N)):
      for e in E:
          model.addConstr(gp.quicksum(A[n][e]*ro[i][n] for n in V) + alph[i][e] <= gp.quicksum(omega[e][l]*x[i][e][l] for l in range(w)))
    
    model.addConstrs(gp.quicksum(omega[a][l]*x[i][a][l]*v[i][a] for l in range(w) for a in E) - gp.quicksum(ro[i][n]*b[0] for n in V) - gp.quicksum(alph[i][a] for a in E) <= 0 for i in range(len(N)))
    
    #Optimize and save ObjVal
    model.optimize()
    
    return model.objVal