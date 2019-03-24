import numpy as np

def fun_getPieces(X, nu, dt):
    Xup1 = np.roll(X, -1, 0)
    Xdown1 = np.roll(X, 1, 0)
    Xdown2 = np.roll(X, 2, 0)
    Xleft1 = np.roll(X, -1, 1)
    
    Z = X + dt/2* (np.multiply(Xup1 - Xdown2, Xdown1) - X + nu)
    Zup1 = np.roll(Z, -1, 0)
    Zdown1 = np.roll(Z, 1, 0)
    Zdown2 = np.roll(Z, 2, 0)
    
    hX = X + dt * (np.multiply(Zup1 - Zdown2, Zdown1) - Z + nu)
    
    return Xup1, Xdown2, Xdown1, Xleft1, Zup1, Zdown2, Zdown1, hX
