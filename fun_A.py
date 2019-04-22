import numpy as np

def fun_A(X, Xleft1, hX, dim_obs, M, Y, Rm, Rf):
    kern2 = Xleft1 - hX
    kern2 = Rf/(2*M) * np.sum(kern2[:,:M-1]**2) # Is this right? I feel like it's leaving out one column
    
    kern1 = Rm/(2*M) * np.sum((X[dim_obs,:] - Y)**2)
    
    return kern1 + kern2