import numpy as np

def fun_dAnu(Xleft1, hX, M, dt, Rf, scaling):
    kern = dt*(Xleft1 - hX)
    dAnu = scaling*(-Rf/M*np.sum(np.sum(kern[:,:-1])))
    
    return dAnu