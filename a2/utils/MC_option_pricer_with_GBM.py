# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 09:39:00 2022

@author: LPras
"""

import numpy as np

#for put option
def putMCPricer(S0, K, vol, T, r, paths, steps):
    
    dt = 1/steps
    total_steps = steps *T
    
    S_t = np.zeros(shape=(total_steps,paths))
    S_t[0,] = S0
    

    #start with 1 so we can check the last price and current price for stock
    #price and for delta
    for i in range(1,total_steps):
        for j in range(0,paths):
            RV = np.random.normal()

            S_t[i,j] = S_t[i-1,j]*np.exp((r-0.5*vol**2)*dt+vol*np.sqrt(dt)*RV)            
                
    P = np.maximum(0, K - S_t[-1,:])

    P = P * np.exp(-r * T)

    average_P = np.mean(P)
    sterr_P = np.std(P)/np.sqrt(paths)
    
    return average_P, sterr_P, S_t

def callMCPricer(S0, K, vol, T, r, paths, steps, seed):
    
    if seed == 'Yes':
        np.random.seed(0)

    dt = 1/steps
    total_steps = steps *T
    
    S_t = np.zeros(shape=(total_steps,paths))
    S_t[0,] = S0
    

    #start with 1 so we can check the last price and current price for stock
    #price and for delta
    for i in range(1,total_steps):
        for j in range(0,paths):
            RV = np.random.normal()

            S_t[i,j] = S_t[i-1,j]*np.exp((r-0.5*vol**2)*dt+vol*np.sqrt(dt)*RV)            
                
    P = np.maximum(0, S_t[-1,:] - K)

    P = P * np.exp(-r * T)

    average_P = np.mean(P)
    sterr_P = np.std(P)/np.sqrt(paths)
    
    return average_P, sterr_P, S_t




