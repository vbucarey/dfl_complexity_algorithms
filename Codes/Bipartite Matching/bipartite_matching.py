# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 23:17:50 2023

@author: sophi
"""

import gurobipy as gp
from gurobipy import GRB

def matching_bi(c, A, b, E, V):
    
    model = gp.Model("Bipartite Matching")
    model.setParam("OutputFlag", 0)
    model.setParam("Threads", 1)
    
    #Variables
    v = {(i, j): model.addVar(vtype = GRB.BINARY, name = "v[({},{})]".format(i, j), lb = 0, ub = 1) for (i, j) in E}
    
    #Objective function
    fo = gp.quicksum(c[a]*v[a] for a in E)
    model.setObjective(fo, GRB.MINIMIZE)
    
    #Constraints
    for i in V:
        model.addConstr(gp.quicksum(A[i][a]*v[a] for a in E) >= b[0])
    
    #Optimize
    model.optimize()
    model.objVal
    
    return model, v