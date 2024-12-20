# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 01:12:20 2023

@author: sophi
"""
import random
import load_data_sp as ld
import max_model_sp as mm
import numpy as np
import time

random.seed(8)

#--------------------Build neighborhood--------------------
def build_v(T, omegaD, eps, d, E):
    W = []
    omegaD = {eval(key): value for key, value in omegaD.items()}
    
    for t in range(T):
        w = {a: [0 for i in range(d + 1)] for a in E}
        
        for a in E:
            for i in range(len(omegaD[a])):
                w[a][i] = omegaD[a][i] + 1*random.normalvariate(0, 1)
                
        normas = {a: 0 for a in E}
        
        for a in E:
            normas[a] = max(abs(w[a][i]) for i in range(d + 1))
            
            for i in range(5):
                if normas[a] != 0:
                    w[a][i] = (1/normas[a])*w[a][i]
                    
                else:
                    w[a][i] = w[a][i]
        W.append(w)
        
    We = [0 for i in range(T)]
    
    for i in range(T):
        We[i] = {str(a): W[i][a] for a in E}
        
    return We

#--------------------Local Search Algorithm--------------------

def local_search(filename, iterlimit, T, eps, omega0, timelimit):

    print('~'*20 + 'Local Search' + '~'*20)
    
    #Read file
    x, V, E, N, s, t, c, A, b, bT, d = ld.readfile(filename)

    minvalue = mm.minmax(filename, omega0)
    argvalue = np.copy(omega0)
    argvalue = argvalue.tolist()
    val_iter = [minvalue]
    omegas_iter = [argvalue]
    
    #Initial time
    t0 = time.time()

    for i in range(iterlimit):
        WT = build_v(T, argvalue, eps, d, E)
        
        cont = 0
        
        for w in WT:
            valuesaux = mm.minmax(filename, w)
            
            print(cont, valuesaux)
            
            if valuesaux < minvalue:
                argvalue = w
                minvalue = valuesaux
                print('\n\n\nFound better omega \n\n\n')
            cont += 1
        
        val_iter.append(minvalue)
        omegas_iter.append(argvalue)
        
        if time.time() - t0 > timelimit:
            print('\n\n\n Time limit reached %f \n\n\n' %(timelimit))
            break
    
    return argvalue, val_iter, omegas_iter