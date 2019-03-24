import numpy as np

def fun_A(X, Xleft1, hX, Y, dim_obs, M, Rm, Rf):
    kern1 = Rm/(2*M) * np.sum((X[dim_obs,:] - Y)**2)
    
    kern2 = Xleft1 - hX
    kern2 = Rf/(2*M) * np.sum(kern2[:,:M-1]**2) # Is this right? I feel like it's leaving out one column
    
    
    return kern1 + kern2