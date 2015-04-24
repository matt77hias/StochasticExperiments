# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sampling import discrete_sample

np.random.seed(87655678)

############################################################################### 
# DISCRETE-STATE CONTINUOUS-TIME MARKOV PROCESSES
#------------------------------------------------------------------------------
# CHEMICALLY REACTING SYSTEMS: SCHLÖGEL's MODEL
###############################################################################
k0 = 0.18
k1 = 2.5 * 10**-4
k2 = 2200
k3 = 37.5
def get_propensity(A):
    if A < 0:
        return 0
    elif A == 0:
        return k2
    elif A == 1:
        return k2 + A*k3
    elif A == 2:
        return A*(A-1)*k0 + k2 + A*k3
    else:
        return A*(A-1)*k0 + A*(A-1)*(A-2)*k1 + k2 + A*k3
        
def reaction0(A): return A + 1
def reaction1(A): return A - 1
def reaction2(A): return A + 1
def reaction3(A): return A - 1
reactions = [reaction0, reaction1, reaction2, reaction3]

def get_reaction(A, propensity):
    if A == 0:
        return reaction2
    elif A == 1:
        return reactions[discrete_sample([2, 3]      , (np.array([                              k2, A*k3]) / propensity))]
    elif A == 2:
        return reactions[discrete_sample([0, 2, 3]   , (np.array([A*(A-1)*k0,                   k2, A*k3]) / propensity))]
    else:
        return reactions[discrete_sample([0, 1, 2, 3], (np.array([A*(A-1)*k0, A*(A-1)*(A-2)*k1, k2, A*k3]) / propensity))]
        
def get_A(A, propensity):
    return get_reaction(A, propensity)(A)
            
def get_time_increment(propensity):
    return (- 1.0 / propensity) * np.log(np.random.sample())
    
def test():
    A, t = direct_simulation(steps=500000, A0=0)
    plt.figure()
    plt.plot(t, A)
    plt.xlabel('Time')
    plt.ylabel('Molecules')
    plt.show()

def direct_simulation(steps, A0=0):
    A = np.zeros((steps+1), dtype=int)
    t = np.zeros((steps+1), dtype=np.float64)
    A[0] = A0
    t[0] = 0.0
    for k in range(1, steps+1):
        propensity = get_propensity(A[k-1])
        A[k] = get_A(A[k-1], propensity)
        t[k] = t[k-1] + get_time_increment(propensity)
    return A, t