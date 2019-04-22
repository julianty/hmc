import numpy as np

def F(x, nu):
    # Extract dimensionality
    D = x.shape[0]
    
    # Initialize Output
    L96 = np.zeros(shape=(D))
    
    # Edge cases for i = 1, 2, D
    L96[0] = (x[1] - x[D-2])* x[D-1] - x[0]
    L96[1] = (x[2] - x[D-1])* x[0] - x[1]
    L96[D-1] = (x[0] - x[D-3])* x[D-2] - x[D-1]
    
    # All the rest of the dimensions
    for d in range(2, D-1):
        L96[d] = (x[d+1] - x[d-2])*x[d-1] - x[d]
        
    L96 = L96 + nu
    
    return L96