from scipy.stats import norm
import numpy as np
from numpy import log, pi, sqrt, exp

def putBlackScholes(S,T, r, K, vol):
    
    def d1(S,T,K,r,vol):
        return (log(S/K)+(r+0.5*vol**2)*T)/vol*sqrt(T)

    d1 = d1(S,T,K,r,vol)

    def d2(T,vol):
        
        return d1 - vol*sqrt(T)
    
    d2 = d2(T,vol)

    put = exp(-r*T)*K*norm.cdf(-d2) - S*norm.cdf(-d1)
    
    return put

def callBlackScholes(S,T, r, K, vol):
    
    def d1(S,T,K,r,vol):
        return (log(S/K)+(r+0.5*vol**2)*T)/vol*sqrt(T)

    d1 = d1(S,T,K,r,vol)

    def d2(T,vol):
        
        return d1 - vol*sqrt(T)
    
    d2 = d2(T,vol)

    call = S*norm.cdf(d1) - exp(-r*T)*K*norm.cdf(d2)
    return [call, norm.cdf(d1)]