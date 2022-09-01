import numpy as np

def exact_GBM(M, S0, T, r, sigma):
    dt = T / M
    S = np.empty(M+1, dtype = type(S0))
    S[0] = S0
    for i in range(M):
        zm = np.random.normal(loc=0.0, scale=1.0)
        S[i+1] = S[i] * np.exp(
                (r - 0.5 * sigma**2)*dt + sigma * np.sqrt(dt) * zm
                )
    return S

def euler_GBM(M, S0, T, r, sigma):
    dt = T/M
    S = np.empty(M+1, dtype = type(S0))
    S[0] = S0
    for i in range(M):
        zm = np.random.normal(loc=0.0, scale=1.0)
        S[i+1] = S[i]*(1 + r * dt * S[i] + sigma * np.sqrt(dt) * zm)
    return S
