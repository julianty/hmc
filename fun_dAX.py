import numpy as np

def fun_dAX(X, Xu1, Xd1, Xd2, Xleft1, hX, eyeD, eyeDleft2, eyeDleft1, eyeDright1, D, dim_obs, M, Y, dt, Rm, Rf, scaling):
    
    GXterm1 = np.multiply(eyeDleft1[:,:,None], np.transpose(Xu1[:,:,None] - Xd2[:,:,None],(0,2,1)))
    GXterm2 = np.multiply(eyeDright1[:,:,None] - eyeDleft2[:,:,None], np.transpose(Xd1[:,:,None], (0,2,1)))
    GX = GXterm1 + GXterm2 - eyeD[:,:,None]
    
    kern2 = np.multiply(np.transpose(X[:,:,None] - np.roll(hX, 1, 1)[:,:,None], (0, 2, 1)), eyeD[:,:,None] - dt/2*GX)
    kern2 = Rf/M*np.transpose(np.sum(kern2,0)[None,:,:], (1,2,0))[:,:,0]
    kern2[:,0] = 0

    kern3 = np.multiply(np.transpose(Xleft1[:,:,None]-hX[:,:,None], (0,2,1)), eyeD[:,:,None] + dt/2*GX)
    kern3 = -Rf/M*np.transpose(np.sum(kern3,0)[None,:,:], (1,2,0))[:,:,0]
    kern3[:,-1] = 0
    
    kern1 = np.zeros(shape=(D,M))
    kern1[dim_obs,:] = Rm/M*(X[dim_obs,:] - Y)

    dAX = scaling * (kern1 + kern2 + kern3)
    
    return dAX