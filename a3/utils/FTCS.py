# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 10:04:30 2022

@author: LPras
"""
import numpy as np
import scipy as sp

def spare_matA(dt, sigma, dx, r, x_steps):
    #FCTS
    #this is mine

    a_upper_diag = (r - 1/2*sigma**2)*(dt/(2*dx)) + 1/2*sigma**2*(dt/dx**2)
    a_down_diag = (-r + 1/2*sigma**2)*(dt/(2*dx)) + 1/2*sigma**2*(dt/dx**2)
    a_main_diag = 1 - (sigma**2*dt)/dx**2 - r*dt

    A = sp.sparse.diags([a_down_diag, a_main_diag, a_upper_diag], [-1, 0, 1], shape=(x_steps, x_steps)).tocsc()
    
    return A, a_upper_diag, a_down_diag

def spare_matB(x_steps):
    B = sp.sparse.identity(x_steps).tocsc()
    return B

