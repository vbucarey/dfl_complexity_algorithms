# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 17:51:15 2024

@author: sophi
"""
import gurobipy as gp
from gurobipy import GRB
import load_data_sp as ld
import time

def alt_omegafix(filename, time_limit, duals, omega0, real_cost):
    
    print('~'*20+'Alternating descent algorithm: fix omega'+'~'*20)
    
    #Read File
    x, V, E, N, s, t, c, A, b, bT, d = ld.readfile(filename)
    w = d + 1
    
    #Average real costs of observations
    promsp = sum(real_cost[i] for i in range(len(N)))/len(N)
    
    #----------Optimization Model----------
    model = gp.Model("Fix Omega")
    model.setParam("Threads", 1)
    model.setParam(GRB.Param.TimeLimit, time_limit)
    model.setParam("Method", 3)
    
    #Variables
    mu = {i: {n: model.addVar(vtype = GRB.CONTINUOUS, name = "mu[{}][{}]".format(i, n), lb = -GRB.INFINITY, ub = 0) for n in V} for i in range(len(N))}
    delta = {i: {(j,k): model.addVar(vtype = GRB.CONTINUOUS, name = "delta[{}][({},{})]".format(i, j, k), lb = 0, ub = GRB.INFINITY) for (j, k) in E} for i in range(len(N))}
    gamma = {i: model.addVar(vtype = GRB.CONTINUOUS, name = "gamma[{}]".format(i), lb = 0, ub = GRB.INFINITY) for i in range(len(N))}
    theta = {i: {(j,k): model.addVar(vtype = GRB.CONTINUOUS, name = "theta[{}][({},{})]".format(i, j, k), lb = 0, ub = GRB.INFINITY) for (j, k) in E} for i in range(len(N))}
    
    #Ranges
    rnod = range(len(V)) 
    rint = range(len(N))
    rest = range(w)
    
    #Omega fix
    omega1 = {eval(key): value for key, value in omega0.items()}
    
    #Objective function
    pred = [{a: gp.quicksum(omega1[a][l] * x[i][a][l] for l in rest) for a in E} for i in rint]
    fo = gp.quicksum(bT[n]*mu[i][n] for n in V for i in rint) + gp.quicksum(pred[i][a]*delta[i][a] for a in E for i in rint) + gp.quicksum(theta[i][a] for i in rint for a in E)
    model.setObjective(fo, GRB.MINIMIZE)
    
    #Constraints
    for a in E:
        for i in rint:
            model.addConstr(gp.quicksum(A[a][n]*mu[i][n] for n in V) + gp.quicksum(x[i][a][l]*omega1[a][l]*gamma[i] for l in rest) + theta[i][a] >= (1/len(N))*c[i][a])
    
    for n in rnod:
        model.addConstrs(gp.quicksum(A[a][n]*delta[i][a] for a in E) - b[n]*gamma[i] >= 0 for i in rint)
        
    model.addConstrs(delta[i][a] <= gamma[i] for i in rint for a in E)
    
    model.addConstr(fo >= promsp)
    
    #Optimize
    
    model.optimize()
       
    if model.SolCount > 0:  
      deltas = {}
      for k in range(len(N)):
          deltas[k] = {}
          for (i, j) in E:
              deltas[k][(i, j)] = delta[k][(i, j)].x
      
      gamas = {}
      for i in range(len(N)):
          gamas[i] = gamma[i].x
          
      dual = [deltas, gamas]
      
      val_omega = omega0
    
    else:
      val_omega = omega0
      dual = duals
    
    return dual, val_omega, model.objVal, model.RunTime

def alt_dualesfix(filename, time_limit, duals, omega0, costoreal):
    
    print('~'*20+'Alternating descent algorithm: fix gamma and delta'+'~'*20)
    
    #Read File
    x, V, E, N, s, t, c, A, b, bT, d = ld.readfile(filename)
    
    #Number of atribbutes
    w = d + 1
    
    #Average real costs of observations
    promsp = sum(costoreal[i] for i in range(len(N)))/len(N)
    
    #----------Optimization Model----------
    model = gp.Model("Fix gamma and delta")
    model.setParam("Threads", 1)
    model.setParam(GRB.Param.TimeLimit, time_limit)
    model.setParam("Method", 3)
    
    #Obtain values ​​of fixed dual variables
    delta = {i: {(j, k): 0 for (j, k) in E} for i in range(len(N))}
    gamma = {i: 0 for i in range(len(N))}
    
    for i in range(len(N)):
        for (j, k) in E: #AQUI ESTA EL PROBLMEMA
            delta[i][(j, k)] = duals[0][i][(j, k)]
            
    for i in range(len(N)):
        gamma[i] = duals[1][i]
        
    #Variables
    mu = {i: {n: model.addVar(vtype = GRB.CONTINUOUS, name = "mu[{}][{}]".format(i, n), lb = -GRB.INFINITY, ub = 0) for n in V} for i in range(len(N))}
    theta = {i: {(j,k): model.addVar(vtype = GRB.CONTINUOUS, name = "theta[{}][({},{})]".format(i, j, k), lb = 0, ub = GRB.INFINITY) for (j, k) in E} for i in range(len(N))}
    omega = {(j,k): [model.addVar(vtype = GRB.CONTINUOUS, name = "omega[({},{})][{}]".format(j, k, l), lb = -1, ub = 1) for l in range(w)] for (j,k) in E}
    
    #Ranges
    rint = range(len(N))
    rest = range(w)
    
    #Objective function
    pred = [{a: gp.quicksum(omega[a][l] * x[i][a][l] for l in rest) for a in E} for i in rint]
    fo = gp.quicksum(bT[n]*mu[i][n] for n in V for i in rint) + gp.quicksum(pred[i][a]*delta[i][a] for a in E for i in rint) + gp.quicksum(theta[i][a] for i in rint for a in E)
    model.setObjective(fo, GRB.MINIMIZE)
    
    for a in E:
        for i in rint:
            model.addConstr(gp.quicksum(A[a][n]*mu[i][n] for n in V) + gp.quicksum(x[i][a][l]*omega[a][l]*gamma[i] for l in rest) + theta[i][a] >= (1/len(N))*c[i][a])
    
    model.addConstr(fo >= promsp)
    
    #Optimize and save variables
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

#------------------------Calculo de soluciones iniciales------------------------

def iter_alt(iterations, filename, time_limit, omega0, real_cost, total_time = 3600):
    
    t0 = time.time()
    
    #Initial solution
    alt = alt_omegafix(filename, time_limit, [], omega0, real_cost)
    
    time_iter_omega = []
    time_iter_dual = []
    time_iter_omega.append(alt[3])
    
    v_obj = []
    v_obj.append(alt[2])
    results = []
    results.append(alt)
    
    all_omegas = []
    all_omegas.append(alt[1])
    
    for i in range(iterations):
        if i % 2 == 0:
            alt = alt_dualesfix(filename, time_limit, alt[0], alt[1], real_cost)
            time_iter_dual.append(alt[3])
            
        else:
            alt = alt_omegafix(filename, time_limit, alt[0], alt[1], real_cost)
            time_iter_omega.append(alt[3])
            
        if time.time() - t0 > total_time:
            print('\n\n\n Time limit reached: %f \n\n\n' %(total_time))
            break
        
        results.append(alt)
        all_omegas.append(alt[1])
        v_obj.append(alt[2])
    
    return all_omegas[-1], all_omegas, v_obj, time_iter_omega, time_iter_dual