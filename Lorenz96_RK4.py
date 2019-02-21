# Objective of this is to create a functional Runge-Kutta
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

class Lorenz96:
    def __init__(self, y0, t0, dimension, forcing=8.17):
        self.y0 = y0
        self.t0 = t0
        self.dim = dimension 
        self.F = forcing
        
    def generate(self, stepsize=0.02, steps=5000, addNoise=True, noise_level=0.4):
        """This function generates a data set using the Lorenz 96 Equations using the Runge Kutta method
            
            Args:
            
            Kwargs:
                stepsize(float): The integration dt
                steps(int): How many time steps to generate
                addNoise(bool): Whether we should add noise or not
                noise_level(float): The standard deviation of the noise to add
            Returns:
                numpy.array. The generated data with shape (dimension, steps)
        
        """
        # Initialize data array
        y = np.zeros((self.dim,steps))
        
        # Set initial conditions
        y[:,0] = self.y0 
        t = self.t0
        
        # Differential Equations
        def yprime(t, y):
            # Find dimensionality
            D = self.dim 
            F = self.F
            # Initialize output
            yprime = np.zeros(D)

            # Take care of edge cases:
            yprime[0] = (y[1] - y[D-2]) * y[D-1] - y[0]
            yprime[1] = (y[2] - y[D-1]) * y[0] - y[1]
            yprime[D-1] = (y[0] - y[D-3]) * y[D-2] - y[D-1]

            # ALl other cases
            for i in range(2, D-1):
                yprime[i] = (y[i+1] - y[i-2]) * y[i-1] - y[i]
            
            # Add forcing
            yprime = yprime + F
            return yprime

        # Runge Kutta
        for i in range(1,steps):
            k1 = stepsize * yprime(t, y[:,i-1])
            k2 = stepsize * yprime(t + stepsize / 2, y[:,i-1] + k1 / 2)
            k3 = stepsize * yprime(t + stepsize / 2, y[:,i-1] + k2 / 2)
            k4 = stepsize * yprime(t + stepsize, y[:,i-1] + k3)


            # Update path
            y[:,i] = y[:,i-1] + (k1 + 2*k2 + 2*k3 + k4)/6
            t = t + stepsize
        
        # Add noise
        if addNoise == True:
            y[:,:] += np.random.normal(0, noise_level, y.shape)
        
        self.data = y
        
        return self.data
    
    def save(self):
        return
    