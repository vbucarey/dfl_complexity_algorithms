# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 23:17:50 2023

@author: sophi
"""
import gurobipy as gp
from gurobipy import GRB

def shortestpath(c, A, b, ARC, NOD):
  model = gp.Model("Shortest Path")
  model.setParam("OutputFlag", 0)
  model.setParam("Threads", 1)

  #Variables
  x = {(i,j): model.addVar(vtype = GRB.BINARY, name="x[({},{})]".format(i, j)) for (i, j) in ARC}

  #Objective Function
  fo = gp.quicksum(c[a]*x[a] for a in ARC)
  model.setObjective(fo, GRB.MINIMIZE)

  #Constraints
  model.addConstrs(gp.quicksum(A[a][i]*x[a] for a in ARC ) == b[i] for i in range(len(NOD)))

  model.optimize()

  return(model, x)