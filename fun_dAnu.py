import numpy as np

def fun_dAnu(Xleft1, Zup1, Zdown2, hX, M, dt, Rf):
    kern = np.multiply(Xleft1 - hX, dt + dt**2/2*(Zup1 - Zdown2 - 1))
    dAnu = -Rf/M * np.sum(kern[:,:-1])
    return dAnu