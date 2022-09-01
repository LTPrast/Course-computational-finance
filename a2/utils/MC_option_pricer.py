# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 09:39:00 2022

@author: LPras
"""

import numpy as np

#call option
def callMCPricer(S0, K, vol, T, r, paths, seed):
    
    if seed == 'Yes':
        np.random.seed(0)


    S_T_without_brownian= S0 * np.exp(r - 0.5 * vol**2)*T
    
    multiple_P = np.array([])
    for i in range(0,paths):
        RV = np.random.normal()

        S_T_with_brownian = S_T_without_brownian * np.exp(vol*np.sqrt(T)*RV)

        P = np.maximum(0, S_T_with_brownian-K)
        multiple_P = np.append(multiple_P, P)
    
    multiple_P = multiple_P * np.exp(-r * T)
    average_P = np.mean(multiple_P)
    sterr_P = np.std(multiple_P)/np.sqrt(paths)
    
    return average_P, sterr_P

#call option
def putMCPricer(S0, K, vol, T, r, paths, seed):
    
    if seed == 'Yes':
        np.random.seed(0)


    S_T_without_brownian= S0 * np.exp(r - 0.5 * vol**2)*T
    
    multiple_payoff = np.array([])
    for i in range(0,paths):
        RV = np.random.normal()

        S_T_with_brownian = S_T_without_brownian * np.exp(vol*np.sqrt(T)*RV)

        P = np.maximum(0, K-S_T_with_brownian)
        multiple_payoff = np.append(multiple_payoff, P)
    
    multiple_option_price = multiple_payoff * np.exp(-r * T)
    average_P = np.mean(multiple_option_price)
    sterr_P = np.std(multiple_payoff)/np.sqrt(paths)
    
    return average_P, sterr_P
