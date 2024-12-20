# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:06:50 2024

@author: sophi
"""
import gurobipy as gp
from gurobipy import GRB
import load_data_sp as ld

def reformulated(filename, tiempo, omega0, real_cost):
    
    print('~'*20 + 'Exact Model' + '~'*20)
    
    #Read File
    x, V, E, N, s, t, c, A, b, bT, d = ld.readfile(filename)
    
    #Number of atribbutes
    w = d + 1
    
    #Average real costs of observations
    promsp = sum(real_cost[i] for i in range(len(N)))/len(N)
    
    #--------------------Optimization Model--------------------
    model = gp.Model("Exact Model")
    model.setParam("NonConvex", 2)
    model.setParam("DualReductions", 0)
    model.setParam("Threads", 1)
    model.setParam(GRB.Param.TimeLimit, tiempo)
    
    #--------------------Variables--------------------
    mu = {i: {n: model.addVar(vtype = GRB.CONTINUOUS, name = "mu[{}][{}]".format(i, n), lb = -GRB.INFINITY, ub = 0) for n in V} for i in range(len(N))}
    theta = {i: {(i, j): model.addVar(vtype = GRB.CONTINUOUS, name = "theta[{}][({},{})]".format(i, j, k), lb = 0, ub = GRB.INFINITY) for (j, k) in E} for i in range(len(N))}
    delta = {i: {(j, k): model.addVar(vtype = GRB.CONTINUOUS, name = "delta[{}][({},{})]".format(i, j, k), lb = 0, ub = GRB.INFINITY) for (j, k) in E} for i in range(len(N))}
    gamma = {i: model.addVar(vtype = GRB.CONTINUOUS, name = "gamma[{}]".format(i), lb = 0, ub = GRB.INFINITY) for i in range(len(N))}
    omega = {(j, k): [model.addVar(vtype = GRB.CONTINUOUS, name = "omega[({},{})][{}]".format(j, k, l), lb = -1, ub = 1) for l in range(w)] for (j, k) in E}

    #--------------------Initial solution--------------------
    omega0 = {eval(key): value for key, value in omega0.items()}
    
    for i in range(w):
        for a in E:
            omega[a][i].Start = omega0[a][i]
    
    #--------------------Range--------------------
    rnod = range(len(V)) 
    rint = range(len(N)) 
    rest = range(w) 
    
    #--------------------Objective Function--------------------
    
    pred = [{a: gp.quicksum(omega[a][l] * x[i][a][l] for l in rest) for a in E} for i in rint]
    fo = gp.quicksum(bT[n]*mu[i][n] for n in V for i in rint) + gp.quicksum(pred[i][a]*delta[i][a] for a in E for i in rint) + gp.quicksum(theta[i][a] for i in rint for a in E)
    model.setObjective(fo, GRB.MINIMIZE)
    
    #--------------------Constraints--------------------
    for a in E:
        for i in rint:
            model.addQConstr(gp.quicksum(A[a][n]*mu[i][n] for n in V) + gp.quicksum(x[i][a][l]*omega[a][l]*gamma[i] for l in rest) + theta[i][a] >= (1/len(N))*c[i][a])
         
    for n in V:
        model.addConstrs(gp.quicksum(A[a][n]*delta[i][n] for a in E) - b[n]*gamma[i] >= 0 for i in rint)
    
    model.addConstrs(delta[i][a] <= gamma[i] for i in rint for a in E)
    
    model.addConstr(fo >= promsp)
    
    #--------------------Callbacks---------------------------
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
    
    #--------------------Optimize--------------------
    model.optimize(mycallback)
    
    val_omega = {str(a): [omega[a][k].X for k in rest] for a in E}
    
    
    return model, val_omega, model._omegavals, model._time