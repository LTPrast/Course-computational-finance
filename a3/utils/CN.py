
import numpy as np
import scipy as sp

def spare_matA(dt, sigma, dx, r, x_steps):
    a_upper_diag = (r - 1/2*sigma**2)*(dt/(4*dx)) +1/4*sigma**2*(dt/(dx**2))
    a_down_diag = (-r + 1/2*sigma**2)*(dt/(4*dx)) +1/4*sigma**2*(dt/(dx**2))
    a_main_diag = 1 - 1/2*sigma**2*dt/(dx**2) - 1/2*dt*r

     
    A = sp.sparse.diags([a_down_diag, a_main_diag, a_upper_diag], [-1, 0, 1], shape=(x_steps, x_steps)).tocsc()

    return A, a_upper_diag, a_down_diag

def spare_matB(dt, sigma, dx, r, x_steps):
    
    B = 2*sp.sparse.identity(x_steps) - spare_matA(dt,sigma,dx,r,x_steps)[0]
 
    return B

    