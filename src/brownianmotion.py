# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(87655678)

############################################################################### 
# CONTINUOUS-STATE CONTINUOUS-TIME MARKOV PROCESSES
# -----------------------------------------------------------------------------
# BROWNIAN MOTIONS
###############################################################################
def test():
    dt=5*10**-2
    t = np.arange(0.0,10.0+dt,dt)
    plt.figure()
    for w in range(10):
        X = direct_simulation(steps=200, X0=0, D=0.5, dt=dt)
        plt.plot(t, X)
    plt.show()      
  
def direct_simulation(steps, X0=0, D=0.5, dt=5*10**-2):
    dx = np.sqrt(2.0 * D * dt)
    X = np.zeros((steps+1), dtype=np.float64)
    X[0] = X0
    for k in range(1, steps+1):
        X[k] = X[k-1] + dx * np.random.normal()
    return X