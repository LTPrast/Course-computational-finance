
import numpy as np
from scipy.stats import norm

#call option
def mc_pricer(S0, K, vol, T, r, paths):
    # np.random.seed(42)

    RV = np.random.normal(0,1,size = (paths,1))                  #Generate a matrix of random variable from N(0,1)    
    ST = S0 * np.exp(vol*np.sqrt(T)*RV + (r-0.5*(vol**2))*T )    #Generate S_T matrix

    #Digital call, make S_T follow the indicator function
    pays = ST
    pays = np.where(pays > K, 1, 0)

    multiple_P = pays * np.exp(-r * T)                          #Payoff matrix
    diff = ST-K                                                 #Difference between Price to strike
    coef = S0*vol*np.sqrt(T)
    likelihood_delta =(np.exp(-r*T)/coef)*RV*pays  # Likelihood delta

    #Pathwise delta
    pathwise_arr = np.exp(-r * T)* norm.pdf(diff)* np.exp((r - 0.5 * (vol**2))*T + vol*np.sqrt(T)*RV)
    
    return ST,RV,np.mean(multiple_P),pays,np.mean(likelihood_delta),np.mean(pathwise_arr)
 