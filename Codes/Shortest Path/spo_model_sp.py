# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 20:20:29 2023

@author: sophi
"""
import gurobipy as gp
from gurobipy import GRB
import load_data_sp as ld
import shortest_path as sp

def spo_mod(filename, alpha, costoreal):   
    print('~'*20+'SPO'+'~'*20)
    
    x, V, E, N, s, t, c, A, b, bT, d = ld.readfile(filename)
    w = d + 1
    
    Z = []
    Vi = {i: {a: 0 for a in E} for i in range(len(N))}
    
    for i in range(len(N)):
        zeta = sp.shortestpath(c[i], A, b, E, V)
        Z.append(zeta[0].objVal)
        
        for a in E:
            Vi[i][a] = zeta[1][a].X
    
    #--------------------Optimization Model--------------------
    model = gp.Model("Proposición 7")
    model.setParam("Threads",1)
    
    #--------------------Variables--------------------
    omega = {(i,j): [model.addVar(vtype = GRB.CONTINUOUS, name = "omega[({},{})][{}]".format(i, j, l), lb = -1, ub = 1) for l in range(w)] for (i,j) in E}
    ro = {i: {n: model.addVar(vtype = GRB.CONTINUOUS, lb = -GRB.INFINITY, ub = GRB.INFINITY, name = "ro[{}][{}]".format(i, n)) for n in V} for i in range(len(N))}
  
    #--------------------Objective Function--------------------
    fo = gp.quicksum(ro[i][s] for i in range(len(N))) + alpha*gp.quicksum(omega[a][k]*x[i][a][k]*Vi[i][a] for k in range(w) for a in E for i in range(len(N))) - gp.quicksum(Z)
    model.setObjective((1/len(N)) * fo, GRB.MINIMIZE)

    #--------------------Constraints--------------------
    for i in range(len(N)):
        for (j, k) in E:
            model.addConstr(ro[i][j] - ro[i][k] >= c[i][(j,k)] - alpha*gp.quicksum(omega[(j,k)][l]*x[i][(j,k)][l] for l in range(w)))

    model.addConstrs(ro[i][t] == 0 for i in range(len(N)))
    
    #--------------------Optimize---------------------------
    model.optimize()

    omega_spo = {str(a): [omega[a][k].X for k in range(w)] for a in E}
        
    return model, omega_spo