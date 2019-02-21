# Objective of this is to create a functional Runge-Kutta

import numpy as np
import matplotlib.pyplot as plt

stepsize = 0.01

# Define differential equation
def yprime(t, y):
    return t

# Initial point
y = 0
t = 0

# Record results
path = []

steps = 1000
for i in xrange(steps):
    k1 = stepsize * yprime(t, y)
    k2 = stepsize * yprime(t + stepsize/2, y + k1/2)
    k3 = stepsize * yprime(t + stepsize/2, y + k2/2)
    k4 = stepsize * yprime(t + stepsize, y + k3)
    
    # Update path
    y = y + (k1 + 2*k2 + 2*k3 + k4)/6
    t = t + stepsize
    path.append(y)
    
