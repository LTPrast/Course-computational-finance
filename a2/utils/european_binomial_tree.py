# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 11:36:23 2022

@author: LPras
"""

import numpy as np
def buildTree(S, vol, T, N):
    dt = T / N

    matrix = np.zeros((N+1, N+1))
    u = np.e**(vol*np.sqrt(dt))
    d = np.e**(-vol*np.sqrt(dt))
    
    #iterate over the lower triangle
    for i in np.arange(N+1): # iterate over rows
        S_new = S * d**(np.arange(i,-1,-1)) * u**(np.arange(0,i+1,1))
        for j in np.arange(i+1) : #iterate over columns            
            matrix[i,j] = S_new[j]

               
    return matrix

def valueEuropeanOptionMatrix(tree, T, r, K,N, vol, option_type):
    dt = T / N
    u = np.e**(vol*np.sqrt(dt))
    d = np.e**(-vol*np.sqrt(dt))
    
    p = (np.e**(r*dt)-d)/(u-d)
    interest = np.e**(-r*dt)
    
    columns = tree.shape[1]
    rows = tree.shape[0]

    # Walk backward , we start in last row of the matrix
    # Add the payoff function in the last row.
    
    # call option
    option_payoff_last = np.zeros(N+1)
    
    for c in np.arange(columns):
        S = tree[rows-1, c] # value in the matrix

        if option_type == "Put":
            C = np.maximum(0, K - S)
            option_payoff_last[c] = C
        elif option_type == "Call":
            C = np.maximum(0, S - K)
            option_payoff_last[c] = C

    
    matrix = np.zeros((N, N+1))
    matrix = np.append(matrix,[option_payoff_last],axis= 0)

    # For all other rows , we need to combine from previous rows
    # We walk backwards , from the last row to the first row
    for i in np.arange(rows-1)[::-1]:       
        for j in range(i+1):
            down = matrix[ i + 1 , j ]
            up = matrix[i + 1 , j + 1 ]

            option_value = interest * (p*up + (1-p)*down)
            
            matrix[i,j] = option_value

    return matrix


  

