# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 00:29:42 2023

@author: sophi
"""
import gurobipy as gp
from gurobipy import GRB
import load_data_sp as ld
    
def minmax(filename, omega):
    
    #Read file
    x, V, E, N, s, t, c, A, b, bT, d = ld.readfile(filename)
    
    #Number of atribbutes
    w = d + 1
    
    #--------------------Optimization Model--------------------
    model = gp.Model("Maximize V")
    model.setParam("OutputFlag", 0)
    model.setParam("DualReductions", 0)
    model.setParam("Threads", 1)
    
    omega = {eval(key): value for key, value in omega.items()}
    
    #--------------------Variables--------------------
    v = {i: {(j, k): model.addVar(vtype = GRB.CONTINUOUS, lb = 0, ub = 1, name = "v[{}][({},{})]".format(i, j, k)) for (j, k) in E} for i in range(len(N))}
    ro = {i: {n: model.addVar(vtype = GRB.CONTINUOUS, lb = 0, ub = GRB.INFINITY, name = "ro[{}][{}]".format(i, n)) for n in V} for i in range(len(N))}
    alph = {i: {(j,k): model.addVar(vtype = GRB.CONTINUOUS, lb = -GRB.INFINITY, ub = 0, name = "alph[{}][({},{})]".format(i, j, k)) for (j, k) in E} for i in range(len(N))}
    
    #--------------------Objective Function--------------------
    fo = (1/len(N))*gp.quicksum(c[i][a]*v[i][a] for i in range(len(N)) for a in E)
    model.setObjective(fo, GRB.MAXIMIZE)
    
    #--------------------Constraints--------------------
    for n in V:
        model.addConstrs(gp.quicksum(A[a][n]*v[i][a] for a in E) >= b[n] for i in range(len(N)))
    
    for i in range(len(N)):
        for (j, k) in E:
            model.addConstr(ro[i][j] - ro[i][k] + alph[i][(j, k)] <= gp.quicksum(omega[(j, k)][l]*x[i][(j, k)][l] for l in range(w)))
    
    model.addConstrs(gp.quicksum(omega[a][l]*x[i][a][l]*v[i][a] for l in range(w) for a in E) - gp.quicksum(ro[i][n]*b[n] for n in V) - gp.quicksum(alph[i][a] for a in E) <= 0 for i in range(len(N)))
    
    #--------------------Optimize---------------------------
    model.optimize()
    
    return model.objVal