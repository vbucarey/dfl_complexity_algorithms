# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 17:12:32 2023

@author: sophi
"""
import gurobipy as gp
from gurobipy import GRB
import load_data_bi as ld

def exact(filename, timelimit, omega0, realcost, env):    
    print('~'*20 + 'Exact Reformulation' + '~'*20)
    
    #Read file
    x, V, E, N, s, t, c, A, b, bT, d = ld.readfile(filename)
    w = d+1
    
    #Average real costs of observations
    promsp = sum(realcost[i] for i in range(len(N)))/len(N)
    
    #----------Optimization Model----------
    model = gp.Model("Exact Reformulation", env = env)
    model.setParam("NonConvex",2)
    model.setParam("DualReductions",0)
    model.setParam("Threads",1)
    model.setParam(GRB.Param.TimeLimit, timelimit)
    
    #Variables
    mu = {i: {n: model.addVar(vtype = GRB.CONTINUOUS, name = "mu[{}][{}]".format(i, n), lb = -GRB.INFINITY, ub = 0) for n in V} for i in range(len(N))}
    delta = {i: {(j, k): model.addVar(vtype = GRB.CONTINUOUS, name = "delta[{}][({},{})]".format(i, j, k), lb = 0, ub = GRB.INFINITY) for (j, k) in E} for i in range(len(N))}
    gamma = {i: model.addVar(vtype = GRB.CONTINUOUS, name = "gamma[{}]".format(i), lb = 0, ub = GRB.INFINITY) for i in range(len(N))}
    omega = {(j, k): [model.addVar(vtype = GRB.CONTINUOUS, name = "omega[({},{})][{}]".format(j, k, l), lb = -1, ub = 1) for l in range(w)] for (j,k) in E} #betas
    
    #Read omega
    omega0 = {eval(key): value for key, value in omega0.items()}
    
    #Warmstart
    for l in range(w):
        for a in E:
            omega[a][l].Start = omega0[a][l]
    
    #Ranges
    rint = range(len(N))
    rest = range(w)
    
    #Objective function
    pred = [{a: gp.quicksum(omega[a][l] * x[i][a][l] for l in rest) for a in E} for i in rint]
    fo = gp.quicksum(bT[0]*mu[i][n] for n in V for i in rint) + gp.quicksum(pred[i][a]*delta[i][a] for a in E for i in rint)
    model.setObjective(fo, GRB.MINIMIZE)
    
    #Constraints
    for a in E:
        for i in rint:
            model.addQConstr(gp.quicksum(A[n][a]*mu[i][n] for n in V) + gp.quicksum(x[i][a][l]*omega[a][l]*gamma[i] for l in rest) >= (1/len(N))*c[i][a])
    
    for n in V:
        model.addConstrs(gp.quicksum(A[n][a]*delta[i][a] for a in E) - b[0]*gamma[i] >= 0 for i in rint)
    
    model.addConstrs(delta[i][a] <= gamma[i] for i in rint for a in E)
    
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
    
    val_omega = {str(a): [omega[a][k].X for k in rest] for a in E}
    
    return model, val_omega, model._omegavals, model._time