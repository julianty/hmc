import numpy as np

def fun_getPieces(X, nu, dt):
    Xu1 = np.roll(X, -1, 0)
    Xd1 = np.roll(X, 1, 0)
    Xd2 = np.roll(X, 2, 0)
    
    Xleft1 = np.roll(X, -1, 1)
    
    Xleft1u1 = np.roll(Xleft1, -1, 0)
    Xleft1d1 = np.roll(Xleft1, 1, 0)
    Xleft1d2 = np.roll(Xleft1, 2, 0)
    
    hX = (X + dt/2*(np.multiply(Xu1 - Xd2, Xd1) - X + nu) +
        dt/2*(np.multiply(Xleft1u1 - Xleft1d2, Xleft1d1) - Xleft1 + nu))
    
    return Xu1, Xd1, Xd2, Xleft1, hX
