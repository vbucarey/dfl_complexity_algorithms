# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 11:31:26 2023

@author: sophi
"""
import csv
import numpy as np
import re

def readfile(filename):
    
    #Obtener cantidad de atributos
    file_name = filename
    data = [int(a) for a in re.findall(r'-?\d+\.?\d*', file_name)]
    d = data[1]
        
    #Nodes
    V = []
    
    #Arcs
    E = []
    
    #Instances
    N = []
    
    #Attributes 
    x = []
    
    with open(filename, newline='') as File:  
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
    
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
       
        for row in reader:
            c[int(row['data'])][(int(row['node_init']), int(row['node_term']))] = float(row['c'])
            x[int(row['data'])][(int(row['node_init']), int(row['node_term']))] = [float(row['at{}'.format(k)]) for k in range(d + 2) if k >= 1]
       
        m = range(len(V))
        n = range(len(E))
        A = [[0 for j in n] for i in m]
        
        b = [[0 for j in range(1)] for i in m]
        
        for i in m:
            for j in n:
                if i == E[j][0]:
                    A[i][j] = 1
                elif i == E[j][1]:
                    A[i][j] = -1
                else:
                    A[i][j] = 0
        
        for i in m:
            if i == s:
                b[i] = 1
            elif i == t:
                b[i] = -1
            else:
                b[i] = 0
            
        bT = np.transpose(b)
        AT = np.transpose(A)
        
        Aa = {(i,j): AT for ((i,j), AT) in zip(E, AT)}   
        
    return x, V, E, N, s, t, c, Aa, b, bT, d
