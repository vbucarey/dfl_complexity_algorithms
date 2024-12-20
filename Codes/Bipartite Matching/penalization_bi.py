# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:44:13 2024

@author: sophi
"""
import gurobipy as gp
from gurobipy import GRB
import load_data_bi as ld

def penalization(filename, ka, timelimit, omega0, realcost, env):
    
    #Read file
    x, V, E, N, s, t, c, A, b, bT, d = ld.readfile(filename)
    w = d + 1
    
    #Average real costs of observations
    promsp = sum(realcost[i] for i in range(len(N)))/len(N)
    
    #----------Optimization Model----------
    model = gp.Model("Penalization Model", env = env)
    model.setParam("NonConvex", 2)
    model.setParam("DualReductions", 0)
    model.setParam(GRB.Param.TimeLimit, timelimit)
    model.setParam("Threads", 1)
    
    #Variables
    mu = [{n: model.addVar(vtype = GRB.CONTINUOUS, name = "mu[{}][{}]".format(i, n), lb = -GRB.INFINITY, ub = 0) for n in V} for i in range(len(N))]
    delta = [{(i,j): model.addVar(vtype = GRB.CONTINUOUS, name = "delta[{}][({},{})]".format(k,i,j), lb = 0, ub = GRB.INFINITY) for (i,j) in E} for k in range(len(N))]
    omega = {(i,j): [model.addVar(vtype = GRB.CONTINUOUS, name = "omega[({},{})][{}]".format(i,j,l), lb = -1, ub = 1) for l in range(w)] for (i,j) in E}

    k=ka
    
    #Read omega
    omega0 = {eval(key): value for key, value in omega0.items()}
    
    #Warmstart
    for i in range(w):
        for a in E:
            omega[a][i].Start = omega0[a][i]
        
    #Ranges
    rint = range(len(N)) 
    rest = range(w)
    
    #Objective function
    pred = [{a: gp.quicksum(omega[a][j] * x[i][a][j] for j in rest) for a in E} for i in rint]
    fo = gp.quicksum(bT[0]*mu[i][n] for n in V for i in rint) + gp.quicksum(pred[i][a]*delta[i][a] for a in E for i in rint)
    model.setObjective(fo, GRB.MINIMIZE)
       
    #Constraints
    for a in E:
        for i in rint:
            model.addConstr(gp.quicksum(A[j][a]*mu[i][j] for j in V) + gp.quicksum(x[i][a][r]*omega[a][r]*k for r in rest) >= (1/len(N))*c[i][a])
    
    for j in V:
        model.addConstrs(gp.quicksum(A[j][a]*delta[i][a] for a in E) - b[0]*k >= 0 for i in rint)
        
    model.addConstrs(delta[i][a] <= k for i in rint for a in E)
    
    model.addConstr(fo >= promsp)
    
    #Callbacks
    def mycallback(model, where):
        if where == GRB.Callback.MIPSOL:
            i = len(model._omegavals)  
            vals = {str(a): [model.cbGetSolution(omega[a][k]) for k in model._rest] for a in model._E}
            model._omegavals[i] = vals
            model._time[i] = model.cbGet(GRB.Callback.RUNTIME)
            
    model._omega = omega 
    model._omegavals = {}
    model._rest = rest
    model._E = E
    model._time = {}
    
    #Optimize and save
    model.optimize(mycallback)
    
    if model.SolCount>0:
        val_omega = {str(a): [omega[a][k].X for k in rest] for a in E}

    else:
        val_omega = omega0
        
    return model, val_omega, model._omegavals, model._time