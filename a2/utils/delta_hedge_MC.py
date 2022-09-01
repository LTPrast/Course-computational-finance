
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

#for put option
def MCPricer(S0, K, vol, T, r, paths, steps, epsilon):
    
    dt = 1/steps
    total_steps = steps *T
    
    S_t = np.zeros(shape=(total_steps,paths))
    S_t[0,] = S0
    P_t = np.zeros(shape=(total_steps,paths))
    P_t[0,] = K-S0

    
    ##### in this part #####
    #start with step 1 so we can check the last price and make a new stock
    #price
    for i in range(1,total_steps):
        for j in range(0,paths):
            RV = np.random.normal()

            #makes a path for all stock prices
            S_t[i,j] = S_t[i-1,j]*np.exp((r-0.5*vol**2)*dt+vol*np.sqrt(dt)*RV)
            
            #calculate option price for all stock prices with negative 
            #interest rate
            P_t[i,j] = np.maximum(0, K - S_t[i,j])*(np.exp(-r * (i+1)*dt))
    
    print(S_t)
    ####in this part ####
    
    deltas = np.array([])

    #total steps -epsilon since you cant determine delta of last step
    for i in range(0,total_steps-epsilon, epsilon):
        for j in range(0,paths):          
            #delta =  (V(S+epsilon)-V(S))/epsilon
            delta = (P_t[i+epsilon,j] - P_t[i,j] )/epsilon
            deltas = np.append(deltas, delta)
    
    #all last option prices of all paths
    P = P_t[-1,:]
    #average option price of all paths
    average_P = np.mean(P)
    sterr_P = np.std(P)/np.sqrt(paths)
    
    #average stock price of all paths
    S_t_average = np.sum(S_t, axis=1)/paths
    
    #average delta of all paths
    deltas = np.reshape(deltas, (int(len(deltas)/paths),paths))
    total_delta = np.sum(deltas, axis=1)/paths
    
    return average_P, sterr_P, S_t, total_delta, S_t_average

stock_price_T0 = 100
T = 1
r = 0.06
strike_price = 99
#volatility
sigma = 0.2
#steps of the stock
steps = 10
#different paths for MC approx
paths = 1

#increments of delta so 1 means that the delta is measured of all  steps
#except the first and last step so 98 steps are measured. 2 means that delta
#is measured but it skips 1 step for every step it measures etc.
epsilon = 1


price_P = MCPricer(stock_price_T0,strike_price,sigma ,T,r,paths, steps, epsilon)
print('option price: ', price_P[0], 'standard error', price_P[1])


average_all_deltas = price_P[3]
average_S_over_time = price_P[4]

#Stock price over time
plt.plot(np.arange(0, steps),average_S_over_time)
plt.title('Average stock price of all paths')
plt.show()

#MC delta for put option
plt.plot(np.arange(epsilon, len(average_all_deltas)*epsilon+1, epsilon),average_all_deltas)
plt.title('Delta hedge with MC for the put option')
plt.show()

print('delta hedge: ',average_all_deltas[-1])

#analytical delta black scholes for put option
delta_call_over_time = norm.cdf((np.log(average_S_over_time/strike_price) + (r + 0.5 *sigma**2)*T)/(sigma*np.sqrt(T)))
delta_put_over_time = delta_call_over_time -1

plt.plot(np.arange(steps), delta_put_over_time)
plt.title('Delta hedge with BS for put option')
plt.show()
