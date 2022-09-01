
import numpy as np

#call option
def callMCPricer(S0, K, vol, T, r, paths, seed):
    
    if seed == 'Yes':
        np.random.seed(42)


    S_T_without_brownian= S0 * np.exp((r - 0.5 * vol**2)*T)
    
    multiple_P = np.array([])
    pathwise_indicators = np.array([])
    multiple_pathwise = np.array([])
    likelihood_delta = np.array([])

    for i in range(0,paths):
        RV = np.random.normal()
       

        S_T_with_brownian = S_T_without_brownian * np.exp(vol*np.sqrt(T)*RV)

        P = 1 if S_T_with_brownian>K else 0

        coef = S0*vol*np.sqrt(T)
        deltas =np.exp(-r*T)*P*RV/coef
        likelihood_delta = np.append(likelihood_delta,deltas)


        # likelihood = np.exp(-r * T)* P * RV2/(S0*vol*np.sqrt(T))
        diff = S_T_with_brownian - K

        multiple_P = np.append(multiple_P, P)
        pathwise_indicators = np.append(pathwise_indicators, diff)



    payoffs = multiple_P
    
    multiple_P = multiple_P * np.exp(-r * T)
    average_P = np.mean(multiple_P)
    # average_L = np.mean(multiple_L)

    likelihood_delta = np.mean(likelihood_delta)
    # likelihood_delta = np.mean(multiple_P * (np.abs(np.random.normal(0,1)/(S0)*vol*np.sqrt(T)))) 
    sterr_P = np.std(multiple_P)/np.sqrt(paths)
    
    return average_P, sterr_P, likelihood_delta, payoffs, pathwise_indicators


