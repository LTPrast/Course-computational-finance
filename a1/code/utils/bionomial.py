import math
import numpy as np 


def buildTree(S, vol , T, N):
    dt = T / N
    matrix = np.zeros((N + 1, N + 1))
    u = np.exp( vol * np.sqrt(dt) )
    d =  np.exp( -vol * np.sqrt(dt) )
# Iterate over the lower triangle
    for i in np.arange(N + 1): # iterate over rows
        for j in np.arange(i + 1): # iterate over columns
# Hint: express each cell as a combination of up
# and down moves 
            matrix[i, j]=S*u**(j)*d**(i-j)
            
    return matrix



# def valueOptionMatrix(tree , T, r , K, vol ):
#     dt = T / N
#     u = np.exp( vol * np.sqrt(dt) )
#     d =  np.exp( -vol * np.sqrt(dt) )
#     p = (np.exp(r*dt)-d)/(u - d)
#     columns = tree.shape[1] 
#     rows = tree.shape[0]

#     for c in np.arange(columns):
#         S = tree[rows-1, c] # value in the matrix 
#         tree[rows-1, c ] =  # TODO


#     for i in np.arange(rows-1)[::-1]:
#         for j in np.arange(i + 1):
#             down = tree[i + 1, j] 
#             up = tree[i + 1, j + 1]
#             tree[ i , j ] = 0 # TODO

#     return tree

sigma = 0.2 
S = 100 
T=1
N=3
K = 99 
r = 0.06
tree = buildTree(S, sigma, T, N) 
# valueOptionMatrix (tree , T, r , K, sigma )
print(tree)