# -*- coding: utf-8 -*-
"""
Created on Tue May  7 12:39:01 2024

@author: sophi
"""

import load_data_bi as ld
import spo_bi as spo
import local_search_bi as ls
import penalization_bi as pm
import exact_bi as ref
import bipartite_matching as mb
import alternating_bi as alter
import re
import time

#------------------------Experiment---------------------------------

def experiment(filename, time_limit, cota):
    
    #------------------------Read file------------------------
    x, V, E, N, s, t, c, A, b, bT, d = ld.readfile(filename)
    
    #Calculation of real costs
    real_cost = []
    
    for i in range(len(N)):
      real_cost.append(mb.matching_bi(c[i], A, b, E, V)[0].objVal)
    
    #------------------------Obtain initial solutions from SPO+------------------------
    mspo, wspo = spo.spo_bi(filename, 2, real_cost)

    #------------------------Hyperameters------------------------
    #Local Search
    iterlimit = 20
    eps = 1
    T = 20
    
    #Penalization Formulation
    kappa = 0.1
    
    #Time Limit for SPO-LS
    time_lim_spols = (1/3) * time_limit
    
    #Resolution Time of SPO+
    time_spo = mspo.RunTime

    #------------------------Initial solution of Omegas with local search algorithm------------------------
    
    start_spo_ls = time.time()
    ls_SPO, fobjSPO, omegasSPO = ls.local_search(filename, iterlimit, T, eps, wspo, time_lim_spols - time_spo)
    end_spo_ls = time.time()
    time_spo_ls = end_spo_ls - start_spo_ls
    
    model_time = time_limit - (time_spo + time_spo_ls)
    
    #------------------------Sequence experiments------------------------
    
    sequences = ['SPO', 'SPO-LS', 'SPO-LS-EXA', 'SPO-LS-PEN', 'SPO-LS-ALT', 'SPO-ALT']
    
    #Dictionaries for storing data
    w = {i: 0 for i in sequences}
    times = {i: 0 for i in sequences}
    gap = {i: 0 for i in sequences}
    f_obj = {i: 0 for i in sequences}
    omega_iter = {i: 0 for i in sequences}
    time_iter = {i: 0 for i in sequences}
    time_alt_omega = {i: 0 for i in sequences}
    time_alt_dual = {i: 0 for i in sequences}
    
    #Results of SPO
    w[sequences[0]] = wspo
    times[sequences[0]] = mspo.RunTime
    gap[sequences[0]] = 0
    f_obj[sequences[0]] = mspo.ObjVal
    omega_iter[sequences[0]] = 0
    time_iter[sequences[0]] = 0
    time_alt_omega[sequences[0]] = 0
    time_alt_dual[sequences[0]] = 0
        
    #Results of SPO-LS
    w[sequences[1]] = ls_SPO
    times[sequences[1]] = time_spo_ls
    gap[sequences[1]] = 0
    f_obj[sequences[1]] = fobjSPO
    omega_iter[sequences[1]] = 0
    time_iter[sequences[1]] = 0
    time_alt_omega[sequences[1]] = 0
    time_alt_dual[sequences[1]] = 0
    
    #Results of SPO-LS-EXA
    print('~'*20 + sequences[2] + '~'*20)
    m_exa, wrefSPO, iter_omega_ref, iter_time_ref = ref.exact(filename, model_time, ls_SPO, real_cost)
    w[sequences[2]] = wrefSPO
    times[sequences[2]] = m_exa.RunTime
    gap[sequences[2]] = m_exa.MIPGap*100
    f_obj[sequences[2]] = m_exa.ObjVal
    omega_iter[sequences[2]] = iter_omega_ref
    time_iter[sequences[2]] = iter_time_ref
    time_alt_omega[sequences[2]] = 0
    time_alt_dual[sequences[2]] = 0
   
    #Results of SPO-LS-PEN
    print('~'*20 + sequences[3] + '~'*20)
    m_pen, w_pen, iter_omega_pen, iter_time_pen = pm.penalization(filename, kappa, model_time, ls_SPO, real_cost)
    w[sequences[3]] = w_pen
    times[sequences[3]] = m_pen.RunTime
    gap[sequences[3]] = m_pen.MIPGap*100
    f_obj[sequences[3]] = m_pen.ObjVal
    omega_iter[sequences[3]] = iter_omega_pen
    time_iter[sequences[3]] = iter_time_pen
    time_alt_omega[sequences[3]] = 0
    time_alt_dual[sequences[3]] = 0
    
    #Results of SPO-LS-ALT
    print('~'*20 + sequences[4] + '~'*20)
    start_alt = time.time()
    w_alt, omegas_alt, obj_alt, time_omega1, time_dual1 = alter.iter_alt(10000, filename, model_time, ls_SPO, real_cost, model_time)
    end_alt = time.time()
    time_alt = end_alt - start_alt
    w[sequences[4]] = w_alt
    times[sequences[4]] = time_alt
    gap[sequences[4]] = omegas_alt
    f_obj[sequences[4]] = obj_alt
    omega_iter[sequences[4]] = 0
    time_iter[sequences[4]] = 0
    time_alt_omega[sequences[4]] = time_omega1
    time_alt_dual[sequences[4]] = time_dual1
    
    #Results of SPO-ALT
    print('~'*20 + sequences[5] + '~'*20)
    start_alt2 = time.time()
    w_alt2, omegas_alt2, obj_alt2, time_omega2, time_dual2 = alter.iter_alt(10000, filename, time_limit - time_spo, wspo, real_cost, time_limit - time_spo)
    end_alt2 = time.time()
    time_alt2 = end_alt2 - start_alt2
    w[sequences[5]] = w_alt2
    times[sequences[5]] = time_alt2
    gap[sequences[5]] = omegas_alt2
    f_obj[sequences[5]] = obj_alt2
    omega_iter[sequences[5]] = 0
    time_iter[sequences[5]] = 0
    time_alt_omega[sequences[5]] = time_omega2
    time_alt_dual[sequences[5]] = time_dual2

    return w, times, gap, f_obj, omega_iter, time_iter, time_alt_omega, time_alt_dual

def data_instances(filename):
    data = [int(a) for a in re.findall(r'-?\d+\.?\d*', filename)]
    if data[-1] == 5:
        data[-1] = 0.5
    obs = data[0]
    deg = data[4]
    noise = data[5]
    
    return obs, deg, noise

def test_file(filename):
    test = filename.replace('train.csv', 'test.csv')
    return test