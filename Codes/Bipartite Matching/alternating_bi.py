# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 18:00:08 2024

@author: sophi
"""

import gurobipy as gp
from gurobipy import GRB
import load_data_bi as ld
import time

def alt_omegafix(filename, timelimit, duals, omega0, realcost, iteration, env):
    
    print('~'*20+'Alternating descent algorithm: fix omega'+'~'*20)
    
    #Read File
    x, V, E, N, s, t, c, A, b, bT, d = ld.readfile(filename)
    w = d + 1
    
    #Average real costs of observations
    promsp = sum(realcost[i] for i in range(len(N)))/len(N)
    
    #----------Optimization Model----------
    model = gp.Model("Fix Omega", env = env)
    model.setParam("Threads", 0)
    model.setParam(GRB.Param.TimeLimit, timelimit)
    model.setParam("Method", 3)
    
    #Variables
    mu = {i: {n: model.addVar(vtype = GRB.CONTINUOUS, name = "mu[{}][{}]".format(i, n), lb = -GRB.INFINITY, ub = 0) for n in V} for i in range(len(N))}
    delta = {i: {(j, k): model.addVar(vtype = GRB.CONTINUOUS, name = "delta[{}][({},{})]".format(i, j, k), lb = 0, ub = GRB.INFINITY) for (j, k) in E} for i in range(len(N))}
    gamma = {i: model.addVar(vtype = GRB.CONTINUOUS, name = "gamma[{}]".format(i), lb = 0, ub = GRB.INFINITY) for i in range(len(N))}

    #Ranges 
    rint = range(len(N))
    rest = range(w)
    
    omega1 = {eval(key): value for key, value in omega0.items()}
    
    if iteration != None:
        for i in rint:
            for (j, k) in E:
                delta[i][(j, k)].PStart = duals[0][i][(j, k)]
                
        for i in rint:
            gamma[i].PStart = duals[1][i]
    
    #Objective Function
    pred = {i: {a: gp.quicksum(omega1[a][l]*x[i][a][l] for l in rest) for a in E} for i in rint}
    fo = gp.quicksum(bT[0]*mu[i][n] for n in V for i in rint) + gp.quicksum(pred[i][a]*delta[i][a] for a in E for i in rint)
    model.setObjective(fo, GRB.MINIMIZE)
    
    #Constraints
    for a in E:
        for i in rint:
            model.addConstr(gp.quicksum(A[n][a]*mu[i][n] for n in V) + gp.quicksum(x[i][a][l]*omega1[a][l]*gamma[i] for l in rest) >= (1/len(N))*c[i][a])
    
    model.addConstr(fo >= promsp)
    
    for n in V:
        model.addConstrs(gp.quicksum(A[n][a]*delta[i][a] for a in E) - b[0]*gamma[i] >= 0 for i in rint)
    
    model.addConstrs(delta[i][a] <= gamma[i] for i in rint for a in E)
    
    #Optimize
    model.optimize()
       
    if model.SolCount > 0:  
      deltas = {}
      for i in rint:
          deltas[i] = {}
          for (j, k) in E:
              deltas[i][(j, k)] = delta[i][(j, k)].x
      
      gamas = {}
      for i in rint:
          gamas[i] = gamma[i].x
          
      dual = [deltas, gamas]
      val_omega = omega0
    
    else:
      val_omega = omega0
      dual = duals
    
    return dual, val_omega, model.objVal, model.RunTime


def alt_dualesfix(filename, timelimit, duals, omega0, costoreal, iteracion, env):
    
    print('~'*20+'Alternating descent algorithm: fix gamma and delta'+'~'*20)
    
    #Read File
    x, V, E, N, s, t, c, A, b, bT, d = ld.readfile(filename)
    w = d + 1
     
    #Average real costs of observations
    promsp = sum(costoreal[i] for i in range(len(N)))/len(N)
    
    #----------Optimization Model----------
    model = gp.Model("Fix gamma and delta", env = env)
    model.setParam("Threads", 0)
    model.setParam(GRB.Param.TimeLimit, timelimit)
    model.setParam("Method", 3)
    
    #Obtain values ​​of fixed dual variables
    delta = {i: {(j,k): 0 for (j,k) in E} for i in range(len(N))}
    gamma = {i: 0 for i in range(len(N))}
    
    for i in range(len(N)):
        for (j, k) in E:
            delta[i][(j, k)] = duals[0][i][(j, k)]
            
    for i in range(len(N)):
        gamma[i] = duals[1][i]

    #Variables
    mu = {i: {n: model.addVar(vtype = GRB.CONTINUOUS, name = "mu[{}][{}]".format(i,n), lb = -GRB.INFINITY, ub = 0) for n in V} for i in range(len(N))}
    omega = {(j, k): [model.addVar(vtype = GRB.CONTINUOUS, name = "omega[({},{})][{}]".format(j, k, l), lb = -1, ub = 1) for l in range(w)] for (j, k) in E}
    
    omega1 = {eval(key): value for key, value in omega0.items()}
    
    #Iniciar con PStart
    if iteracion != None:
        for l in range(w):
            for a in E:
                omega[a][l].PStart = omega1[a][l]
    
    #Ranges
    rint = range(len(N))
    rest = range(w)
    
    #Objective Function
    pred = [{a: gp.quicksum(omega[a][l]*x[i][a][l] for l in rest) for a in E} for i in rint]
    fo = gp.quicksum(bT[0]*mu[i][n] for n in V for i in rint) + gp.quicksum(pred[i][a]*delta[i][a] for a in E for i in rint)
    model.setObjective(fo, GRB.MINIMIZE)
    
    #Constraints
    for a in E:
        for i in rint:
            model.addConstr(gp.quicksum(A[n][a]*mu[i][j] for n in V) + gp.quicksum(x[i][a][l]*omega[a][l]*gamma[i] for l in rest) >= (1/len(N))*c[i][a])
    
    model.addConstr(fo >= promsp)
    
    #Optimize
    model.optimize()
    
    if model.SolCount > 0:
      val_omega = {str(a): [omega[a][l].X for l in rest] for a in E}
      
      deltas = {}
      for i in range(len(N)):
          deltas[i] = {}
          for (j, k) in E:
              deltas[i][(j, k)] = delta[i][(j, k)]
      
      gamas = {}
      for i in range(len(N)):
          gamas[i] = gamma[i]
          
      dual = [deltas, gamas]

    else:
      val_omega = omega0
      dual = duals
    
    return dual, val_omega, model.objVal, model.RunTime
    
#------------------------Calculation of initial solutions------------------------

def iter_alt(iterations, filename, timelimit, omega0, realcost, total_time, env):
    
    t0 = time.time()
    
    #Initial solution
    alt = alt_omegafix(filename, timelimit, [], omega0, realcost, None, env)
    
    tiempo_iteracion_omega = []
    tiempo_iteracion_dual = []
    tiempo_iteracion_omega.append(alt[3])
    
    v_obj = []
    v_obj.append(alt[2])
    results = []
    results.append(alt)
    
    all_omegas = []
    all_omegas.append(alt[1])
    
    for i in range(iterations):
        if i % 2 == 0:
            alt = alt_dualesfix(filename, timelimit, alt[0], alt[1], realcost, i, env)
            tiempo_iteracion_dual.append(alt[3])
            
        else:
            alt = alt_omegafix(filename, timelimit, alt[0], alt[1], realcost, i, env)
            tiempo_iteracion_omega.append(alt[3])
            
        if time.time() - t0 > total_time:
            print('\n\n\n Time limit reached %f \n\n\n' %(total_time))
            break
        
        results.append(alt)
        all_omegas.append(alt[1])
        v_obj.append(alt[2])
    
    return all_omegas[-1], all_omegas, v_obj, tiempo_iteracion_omega, tiempo_iteracion_dual