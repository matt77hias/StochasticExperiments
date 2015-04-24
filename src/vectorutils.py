import numpy as np

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