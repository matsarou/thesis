from distributions.GammaDistribution import GammaExponential
from distributions.NormalDistribution import NormalNormalKnownVar
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def prior_t(a, b):
    return GammaExponential(alpha=a, beta=b)

def prior_mean(tu, mu0, tu0):
    return NormalNormalKnownVar(np.math.sqrt(1/tu), mu0, np.math.sqrt(1/tu0))

def sampling(normal_mu, gamma_tu, data):
    #proposed posterior for estimation of mean
    known_tu = gamma_tu.sample()
    normal_mu.update_params(known_tu, normal_mu.mean, normal_mu.var)

    # proposed posterior for estimation of variance
    mean = normal_mu.sample()
    gamma_tu.update_params(len(data)/2, sum([pow((d - mean), 2) for d in data])/2)

    return normal_mu, gamma_tu

#Define hypothesis
mean=12
tu = 0.0625

#Assign prior to each hyperparameter
mu0=12
tu = 0.0625
tu0 = 0.5
prior_mu=prior_mean(tu=tu, mu0=mu0, tu0=tu)
a0 = 25.0
b0 = 0.5
prior_tu=prior_t(a=a0, b=b0)

#plot the priors
x = np.linspace(0, 100, 60)
prior_mu.plot(x, color='b', xlabel='hypothesis', ylabel='density', label= 'mean prior: mu0=' + str(mu0) + ', tu0=' + str(tu))
prior_tu.plot(x, color='r',label = 'tu prior: a=' + str(a0) + ', b=' + str(b0))
plt.show()

# Gather data
data = [10.2]
#sample
normal_mu = prior_mu
gamma_tu = prior_tu
my_df=[]
# MCMC with n=5 trials
n = 5
for it in range(n):
    normal_mu, gamma_tu = sampling(normal_mu=normal_mu, gamma_tu=gamma_tu, data=data)
    d = {
        'Trial': it,
        'Data': data,
        'n': len(data),
        'mu': normal_mu.mean,
        'tu': normal_mu.var,
        'p(m)|p(t)': normal_mu.known_var,
        'a':gamma_tu.alpha,
        'b':gamma_tu.beta,
        'p(t)|p(m)': normal_mu.mean
    }
    my_df.append(d)

#export to a csv
my_df = pd.DataFrame(my_df)

