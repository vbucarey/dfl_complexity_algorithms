# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 19:28:56 2023

@author: sophia
"""
import shortest_path as sp
import gurobipy as gp
from gurobipy import GRB
import load_data_sp as ld

def regret_function(omega, costoreal, filename):
    
    print('~'*20 + 'Regret Calculator' + '~'*20)
    #Read File
    x, V, E, N, s, t, c, A, b, bT, d = ld.readfile(filename)
    
    #Number of atribbutes
    w = d + 1
    
    #Z is the value of the shortest path per instance
    Z = []
    
    for i in range(len(N)):
        zeta = sp.shortestpath(c[i], A, b, E, V)
        Z.append(zeta[0].objVal)
    
    #--------------------Optimization Model--------------------
    model = gp.Model("Maximize V")
    model.setParam("DualReductions", 0)
    model.setParam("Threads", 1)
    
    omega = {eval(key): value for key, value in omega.items()}
    
    #--------------------Optimization Model--------------------
    v = [{(j,k): model.addVar(vtype = GRB.CONTINUOUS, lb = 0, ub = 1, name="v[{}][({},{})]".format(i, j, k)) for (j, k) in E} for i in range(len(N))]
    ro = [{n: model.addVar(vtype = GRB.CONTINUOUS, lb = 0, ub = GRB.INFINITY, name = "ro[{}][{}]".format(i, n)) for n in V} for i in range(len(N))]
    alph = [{(j,k): model.addVar(vtype = GRB.CONTINUOUS, lb = -GRB.INFINITY, ub = 0, name = "alph[{}][({},{})]".format(i, j, k)) for (j, k) in E} for i in range(len(N))]
    
    #--------------------Objective Function--------------------
    mobj = gp.quicksum(c[i][a]*v[i][a] for i in range(len(N)) for a in E)
    model.setObjective(mobj - sum(Z), GRB.MAXIMIZE)
    
    #--------------------Constraints--------------------
    for n in V:
        model.addConstrs(gp.quicksum(A[a][n]*v[i][a] for a in E) >= b[n] for i in range(len(N)))
    
    for i in range(len(N)):
        for (j,k) in E:
            model.addConstr(ro[i][j] - ro[i][k] + alph[i][(j,k)] <= gp.quicksum(omega[(j, k)][l]*x[i][(j, k)][l] for l in range(w)))
    
    model.addConstrs(gp.quicksum(omega[a][l]*x[i][a][l]*v[i][a] for l in range(w) for a in E) - gp.quicksum(ro[i][n]*b[n] for n in V) - gp.quicksum(alph[i][a] for a in E) <= 0 for i in range(len(N)))
    
    #--------------------Optimize---------------------------
    model.optimize()
    regret = model.objVal
        
    return regret