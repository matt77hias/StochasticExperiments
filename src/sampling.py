from scipy.stats import rv_discrete

def discrete_sample(values, probabilities, ns=1):
    distrib = rv_discrete(values=(range(len(values)), probabilities))
    indices = distrib.rvs(size=ns)
    if ns == 1:
        return map(values.__getitem__, indices)[0]
    else:
        return map(values.__getitem__, indices)