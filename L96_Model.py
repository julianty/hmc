def L96(y, F):
    """
    This function models the Lorenz 96 Model
        Args:
            y(array): the state vector for current timestep
            F(float): the Lorenz 96 forcing term
    """
    # Find dimensionality
    D = len(y) 
    
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