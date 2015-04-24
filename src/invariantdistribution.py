import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_discrete

np.random.seed(87655678)

############################################################################### 
# TESTS
###############################################################################
def get_T1():
    return np.array([[0.9, 0.075, 0.025], [0.15, 0.8, 0.05], [0.25, 0.25, 0.5]])

def get_M(a, b):
    return np.array([[0.5, 0.5, 0.0, 0.0, 0.0], [0.5, 0.5*(1-a), b, 0.0, 0.0], [0.0, 0.5*a, 0.0, 0.3*a, 0.0], [0.0, 0.0, 1.0-b, 0.3*(1-a), 0.4], [0.0, 0.0, 0.0, 0.7, 0.6]])
    
def get_M1():
    return get_M(0.5, 0.5)
    
def get_M2():
    return get_M(0.5, 0.7)
    
def get_M3():
    return get_M(0.001, 0.5)
    
def test():
    approximation_error_walkers(T=get_M1(), walkers=1000, steps=100, norm=norm2)
    approximation_error_walkers(T=get_M2(), walkers=1000, steps=100, norm=norm2)
    approximation_error_walkers(T=get_M3(), walkers=1000, steps=100, norm=norm2)
    
############################################################################### 
# NORMS
############################################################################### 
def norm2(v):
    return np.linalg.norm(v)
    
def norm1(v):
    return np.sum(np.abs(v))
    
def normalise(v, fnorm):
    norm=fnorm(v)
    if norm==0: 
        return v
    return v/norm

def normalise2(v):
    return normalise(v, norm2)
    
def normalise1(v):
    return normalise(v, norm1)

############################################################################### 
# DISCRETE-STATE DISCRETE-TIME MARKOV PROCESSES
###############################################################################  
def discrete_sample(values, probabilities, ns=1):
    distrib = rv_discrete(values=(range(len(values)), probabilities))
    indices = distrib.rvs(size=ns)
    if ns == 1:
        return map(values.__getitem__, indices)[0]
    else:
        return map(values.__getitem__, indices) 

def direct_simulation(T, steps, X0=0):
    X = np.zeros((steps+1), dtype=int)
    X[0] = X0
    values = range(T.shape[0])
    for k in range(1, steps+1):
        X[k] = discrete_sample(values, T[:,X[k-1]])
    return X
    
############################################################################### 
# INVARIANT DISTRIBUTION
############################################################################### 
def invariant_distribution(T):
    E, V = np.linalg.eig(T)
    I = None
    for i in range(len(E)):
        if (abs(E[i]-1.0) < 10**-7):
            if (I != None):
                raise Error('Multiple eigenvalues are equal to 1.0')
            else:
                I = V[:,i]
        elif (E[i] > 1.0):
            raise Error('Eigenvalue ' + str(E[i]) + ' > 1.0')
    if (I == None):
        raise Error('No eigenvalue is equal to 1.0')
    else:
        return normalise1(I)

def invariant_distribution_via_direct_simulation(T, walkers, steps):
    invd = np.zeros(T.shape[0], dtype=int)
    for w in range(walkers):
        i = direct_simulation(T=T, steps=steps, X0=np.random.choice(range(T.shape[0])))[steps]
        invd[i] = invd[i] + 1
    return np.float64(invd) / np.float64(walkers)
        
def approximation_error_walkers(T, walkers, steps, norm=norm2):
    I = invariant_distribution(T)
    ews = np.zeros((walkers), dtype=np.float64)
    invd = np.zeros(T.shape[0], dtype=int)
    for w in range(walkers):
        i = direct_simulation(T=T, steps=steps, X0=np.random.choice(range(T.shape[0])))[steps]
        invd[i] = invd[i] + 1
        ews[w] = norm(I - (np.float64(invd) / np.float64(w+1)))
        
    plt.figure()
    plt.plot(range(1, walkers+1), ews)
    plt.xlabel('Walkers')
    plt.ylabel('Error')
    plt.show()

class Error(Exception):
    def __init__(self, msg=None):
        self.msg = msg
    def __str__(self):
        print(self.msg)