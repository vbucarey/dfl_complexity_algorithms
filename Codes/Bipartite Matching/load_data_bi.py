# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 11:31:26 2023

@author: sophi
"""
import csv
import numpy as np
import re

def readfile(filename):
    
    data = [int(a) for a in re.findall(r'-?\d+\.?\d*', filename)]
    d = data[1]
    
    #Nodes
    V = []
    
    #Edges
    E = []
    
    #Observations
    N = []
    
    #Attributes
    x = []
    
    with open(filename, newline = '') as File:  
        df = csv.DictReader(File)

        for row in df:
            N.append(int(row['data']))
            E.append((int(row['node_init']), int(row['node_term'])))
            V.append(int(row['node_init']))
            V.append(int(row['node_term']))
            
    V = list(set(V))
    E = list(set(E))
    N = list(set(N))
    s = min(V)
    t = max(V)
    
    c = [{a: i for a in E} for i in N]
    x = [{a: i for a in E} for i in N]
    
    with open(filename, newline = '') as csvfile:
        reader = csv.DictReader(csvfile)
       
        for row in reader:
            c[int(row['data'])][(int(row['node_init']), int(row['node_term']))] = -float(row['c'])
            x[int(row['data'])][(int(row['node_init']), int(row['node_term']))] = [float(row['at{}'.format(k)]) for k in range(d + 2) if k >= 1]
       
        #Creating matrices A and b
        
        A = {h: {(j,k): -1 if h in (j,k) else 0 for (j,k) in E} for h in V}
        
        b = [-1]
        
        #Transposed matrices
        bT = np.transpose(b)
        
    return x, V, E, N, s, t, c, A, b, bT, d