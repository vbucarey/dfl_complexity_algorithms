# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 20:20:29 2023

@author: sophi
"""

import gurobipy as gp
from gurobipy import GRB
import load_data_bi as ld
import bipartite_matching as mb

def spo_bi(filename, alpha, costoreal):   
        
    #Read File
    x, V, E, N, s, t, c, A, b, bT, d = ld.readfile(filename)
    w = d + 1
    
    #Calculate the total cost of the shortest path with the real costs (Z*)
    Z = []
    
    #Obtain the shortest path solution with real costs for each observation v*
    Vi = [{a: 0 for a in E} for i in range(len(N))]
    
    for i in range(len(N)):
        zeta = mb.matching_bi(c[i], A, b, E, V)
        Z.append(zeta[0].objVal)
        for a in E:
            Vi[i][a] = zeta[1][a].X
    
    #----------------SPO+ Optimization Model----------------
    model = gp.Model("SPO+ Bipartite Matching")
    model.setParam("Threads", 1) 
    
    #Variables
    omega = {(i,j): [model.addVar(vtype = GRB.CONTINUOUS, name = "omega[({},{})][{}]".format(i, j, l), lb = -1, ub = 1) for l in range(w)] for (i, j) in E}
    ro7 = {i: {n: model.addVar(vtype = GRB.CONTINUOUS, lb = 0, ub = GRB.INFINITY, name = "ro[{}][{}]".format(i, n)) for n in V} for i in range(len(N))}
      
    #Objective Function
    fo = gp.quicksum(ro7[i][n] for i in range(len(N)) for n in V) + alpha*gp.quicksum(omega[a][k]*x[i][a][k]*Vi[i][a] for k in range(w) for a in E for i in range(len(N))) - gp.quicksum(Z)
    
    model.setObjective((1/len(N))*fo, GRB.MINIMIZE)
    
    #Constraints
    for i in range(len(N)):
        for a in E:
            model.addConstr(ro7[i][a[0]] + ro7[i][a[1]] >= c[i][a] - alpha*gp.quicksum(omega[a][l]*x[i][a][l] for l in range(w)))
    
    #Optimize and save variables
    model.optimize()
    
    omega_spo = {str(a): [omega[a][k].X for k in range(w)] for a in E}
        
    return model, omega_spo