import numpy as np

def fun_dAX(X, Xup1, Xdown2, Xdown1, Xleft1, Zup1, Zdown2, Zdown1, hX, Y, eyeD, eyeDleft1, eyeDleft2, eyeDright1, D, dim_obs, M, dt, Rm, Rf):
    
    GX = (np.multiply(eyeDleft1[:,:,None], np.transpose(Xup1[:,:,None] - Xdown2[:,:,None], (0, 2, 1))) 
               + np.multiply(eyeDright1[:,:,None] - eyeDleft2[:,:,None], np.transpose(Xdown1[:,:,None], (0, 2, 1))) - eyeD[:,:,None])
    
    GZ = (np.multiply(eyeDleft1[:,:,None], np.transpose(Zup1[:,:,None] - Zdown2[:,:,None], (0, 2, 1))) 
          + np.multiply(eyeDright1[:,:,None] - eyeDleft2[:,:,None], np.transpose(Zdown1[:,:,None], (0, 2, 1))) - eyeD[:,:,None])
      
    GZGX = np.zeros(shape=(D,D,M))
    for k in xrange(M):
        GZGX[:,:,k] = np.matmul(GZ[:,:,k], GX[:,:,k])   

    
    T = eyeD[:,:,None] + dt*GZ + dt**2 / 2*GZGX
      
    kern3 = -Rf/M * np.transpose(np.sum(np.transpose(Xleft1[:,:,None] - hX[:,:,None], (0, 2, 1)) * T,axis=0)[None,:,:], (1, 2, 0))[:,:,0]
    test = (np.transpose(np.sum(np.transpose(Xleft1[:,:,None] - hX[:,:,None], (0, 2, 1)) * T,axis=0)[None,:,:], (1, 2, 0))[:,:,0])
    
    kern3[:,M-1] = 0 # What is this for?
    
    kern1 = np.zeros(shape=(D,M))
    kern1[dim_obs,:] = float(Rm)/M * (X[dim_obs,:] - Y)
    
    kern2 = Rf/M * (X - np.roll(hX, 1, 1))
    kern2[:,0] = 0
    
        
    dAX = kern1 + kern2 + kern3
    return dAX