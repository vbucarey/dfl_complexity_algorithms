# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 10:47:09 2024

@author: sophi
"""
import gurobipy as gp
from gurobipy import GRB
import load_data_sp as ld

def penalization(filename, kappa, tiempo, omega0, costoreal, env):
    
    print('~'*20 + 'Penalization Model' + '~'*20)
    
    #Read File
    x, V, E, N, s, t, c, A, b, bT, d = ld.readfile(filename)
    
    #Number of atribbutes
    w = d + 1
    
    #Average real costs of observations
    promsp = sum(costoreal[i] for i in range(len(N)))/len(N)
    
    #--------------------Optimization Model---------------------------
    
    model = gp.Model("penalizado", env)
    model.setParam("NonConvex", 2)
    model.setParam("DualReductions", 0)
    model.setParam(GRB.Param.TimeLimit, tiempo)
    model.setParam("Threads", 1)
    
    #--------------------Variables---------------------------
    mup = {i: {n: model.addVar(vtype = GRB.CONTINUOUS, name = "mu[{}][{}]".format(i, n), lb = -GRB.INFINITY, ub = 0) for n in V} for i in range(len(N))}
    thetap = {i: {(j, k): model.addVar(vtype = GRB.CONTINUOUS, name = "theta[{}][({},{})]".format(i, j, k), lb = 0, ub = GRB.INFINITY) for (j, k) in E} for i in range(len(N))}
    deltap = {i: {(j, k): model.addVar(vtype = GRB.CONTINUOUS, name = "delta[{}][({},{})]".format(i, j, k), lb = 0, ub = GRB.INFINITY) for (j, k) in E} for i in range(len(N))}
    omegap = {(j, k): [model.addVar(vtype = GRB.CONTINUOUS, name = "omega[({},{})][{}]".format(j, k, l), lb = -1, ub = 1) for l in range(w)] for (j, k) in E}
    
    omega0 = {eval(key): value for key, value in omega0.items()}
    
    for i in range(w):
        for a in E:
            omegap[a][i].Start = omega0[a][i]
        
    #--------------------Range---------------------------
    rnod = range(len(V))
    rint = range(len(N))
    rest = range(w)
    
    #--------------------Objective Function---------------------------
    pred = [{a: gp.quicksum(omegap[a][l] * x[i][a][l] for l in rest) for a in E} for i in rint]
    fo = gp.quicksum(bT[n]*mup[i][n] for n in V for i in rint) + gp.quicksum(pred[i][a]*deltap[i][a] for a in E for i in rint) + gp.quicksum(thetap[i][a] for i in rint for a in E)
    model.setObjective(fo, GRB.MINIMIZE)
       
    #--------------------Constraints---------------------------
    
    for a in E:
        model.addConstrs(gp.quicksum(A[a][n]*mup[i][n] for n in V) + gp.quicksum(x[i][a][l]*omegap[a][l]*kappa for l in rest) + thetap[i][a] >= (1/len(N))*c[i][a] for i in rint)
    
    for n in rnod:
        model.addConstrs(gp.quicksum(A[a][n]*deltap[i][a] for a in E) - b[n]*kappa >= 0 for i in rint)
        
    model.addConstrs(deltap[i][a] <= kappa for i in rint for a in E)
    
    model.addConstr(fo >= promsp)
    
    #--------------------Callbacks---------------------------
    def mycallback(mod, where):
        if where == GRB.Callback.MIPSOL:
            i = len(model._omegavals)  
            vals = {str(a): [mod.cbGetSolution(omegap[a][k]) for k in mod._rest] for a in mod._E}
            model._omegavals[i] = vals
            model._time[i] = model.cbGet(GRB.Callback.RUNTIME)
            
    model._omega = omegap 
    model._omegavals = {}
    model._rest = rest
    model._E = E
    model._time = {}
    
    #--------------------Optimize---------------------------
    model.optimize(mycallback)
    
    if model.SolCount > 0:
        val_omegap = {str(a): [omegap[a][l].X for l in rest] for a in E}

    else:
        val_omegap = omega0
        
    return model, val_omegap, model._omegavals, model._time