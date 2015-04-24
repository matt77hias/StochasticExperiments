import numpy as np
import matplotlib.pyplot as plt

np.random.seed(87655678)

def test():
    steps = 100000
    t = range(0, steps+1)
    for b in [1, 10, 20]:
        for a in [0.01, 0.1, 1]:
            X1, A1 = metropolis(trial_move=get_discretization1(a, b), V=get_potential, steps=steps, X0=0)
            X2, A2 = metropolis(trial_move=get_discretization2(a, b), V=get_potential, steps=steps, X0=0)
            
            plt.figure()
            plt.hist(X1, bins=100, range=(-2,2))
            plt.savefig('hist_tm=1_a=' + str(a) + '_b=' + str(b) + '.png')
            plt.close()
            plt.figure()
            plt.plot(t, X1)
            plt.savefig('traject_tm=1_a=' + str(a) + '_b=' + str(b) + '.png')
            plt.close()
            print(float(A1) / steps)
            
            plt.figure()
            plt.hist(X2, bins=100, range=(-2,2))
            plt.savefig('hist_tm=2_a=' + str(a) + '_b=' + str(b) + '.png')
            plt.close()
            plt.figure()
            plt.plot(t, X2)
            plt.savefig('traject_tm=2_a=' + str(a) + '_b=' + str(b) + '.png')
            plt.close()
            print(float(A2) / steps)

############################################################################### 
# METROPOLIS
############################################################################### 

def get_potential(x):
    return x**4 - x**2
    
def get_potential_gradient(x):
    return 4*x**3 - 2*x
    
def get_discretization1(a, b):
    def get_trial_move(X):
        return X - a * get_potential_gradient(X) + np.sqrt(2*a/b) * np.random.normal()
    return get_trial_move
     
def get_discretization2(a, b):
    def get_trial_move(X):
        return X + np.sqrt(2*a/b) * np.random.normal()
    return get_trial_move
    
def metropolis(trial_move, V, steps, X0=0):
    X = np.zeros((steps+1), dtype=np.float64)
    X[0] = X0
    A = 0
    for k in range(steps+1):
        Xt = trial_move(X[k-1])
        dV = V(Xt) - V(X[k-1])
        if np.random.sample() <= np.exp(-dV):
            X[k] = Xt
            A = A + 1
        else:
            X[k] = X[k-1]
    return X, A
    