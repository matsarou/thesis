import utils
from distributions.GammaDistribution import GammaExponential
from distributions.NormalDistribution import NormalNormalKnownVar
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from mcmc import MCMC


def prior_t(a, b):
    return GammaExponential(alpha=a, beta=b)

def prior_mean(tu, mu0, tu0):
    # return NormalNormalKnownVar(np.math.sqrt(1/tu), mu0, np.math.sqrt(1/tu0))
    return NormalNormalKnownVar(tu, mu0, tu0)

class MCMC_Gibbs(MCMC.Engine):
    def plot_characteristics(self):
        plt.figure(1)
        plt.ylabel('Proposal')
        plt.xlabel('Trial')
        plt.title('Traces')

    def sampler(self):
        pass

    def sampling(self, normal_mu, gamma_tu, tu_prop, data, mu_prop=0):
        #proposed posterior for estimation of mean
        normal_mu = normal_mu.update_params(tu_prop, normal_mu.mean, normal_mu.var)

        # proposed posterior for estimation of variance
        while True:
            tu_prop = gamma_tu.sample()
            if tu_prop > 0:
                break
        while True:
            mu_prop = normal_mu.sample()
            if mu_prop > 0:
                break
        try:
            gamma_tu = gamma_tu.update_params(len(data)/2, sum([pow((d - mu_prop), 2) for d in data])/2)
        except OverflowError:
            print('errror = ',mu_prop)

        return normal_mu, normal_mu.sample(), gamma_tu, gamma_tu.sample()

#Define hypothesis
mean=12
tu = 0.0625

#Assign prior to each hyperparameter
tu = 0.0625
mu0=12
tu0 = 4.0
prior_mu=prior_mean(tu=tu, mu0=mu0, tu0=tu)
a0 = 25.0 #shape parameter α0=25
b0 = 0.5 #rate parameter β0=0.5
prior_tu=prior_t(a=a0, b=b0)

#plot the priors
x = np.linspace(0, 100, 60)
prior_mu.plot(x, color='b', xlabel='Hypotheses', ylabel='Density', label= 'μ prior normal distr: μ0=' + str(mu0) + ', τ0=' + str(tu))
prior_tu.plot(x, color='r',label = 'τ prior gamma distr: a0=' + str(a0) + ', b0=' + str(b0))
plt.show()

# Gather data
data = [10.2]
#sample
normal_mu = NormalNormalKnownVar(tu, mu0, tu0, data)
gamma_tu = GammaExponential(alpha=a0, beta=b0)
my_df=[]
# MCMC with n=5 trials
n = 40
mu_prop=0
while True:
 tu_prop=gamma_tu.sample()
 if tu_prop>0:
     break
my_df.append({
        'Trial': 0,
        'Data': data,
        'n': len(data),
        'μ': normal_mu.mean,
        'τ': normal_mu.var,
        'μ_prop': None,
        'a':gamma_tu.alpha,
        'b':gamma_tu.beta,
        'τ_prop': tu_prop
    })

mcmc=MCMC_Gibbs()
mu_prop_traces=[]
tu_prop_traces=[]
for it in range(1,n):
    normal_mu, mu_prop, gamma_tu, tu_prop = mcmc.sampling(normal_mu=normal_mu, gamma_tu=gamma_tu,
                                                     tu_prop = tu_prop, mu_prop = mu_prop, data=data)
    d = {
        'Trial': it,
        'Data': data,
        'n': len(data),
        'μ': normal_mu.mean,
        'τ': normal_mu.var,
        'μ_prop': mu_prop,
        'a':gamma_tu.alpha,
        'b':gamma_tu.beta,
        'τ_prop': tu_prop
    }
    my_df.append(d)
    mu_prop_traces.append(mu_prop)
    tu_prop_traces.append(tu_prop)
    
mcmc.plot_trace(trace=mu_prop_traces, label='Proposals for μ')
mcmc.plot_trace(trace=tu_prop_traces, label='Proposals for τ')
plt.show()
#export to a csv
my_df = pd.DataFrame(my_df)
my_df.to_csv("maple_syrup_mcmc.csv")
