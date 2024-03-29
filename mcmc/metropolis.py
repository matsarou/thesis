import numpy as np
from scipy.stats import poisson
import pandas as pd

import utils
from distributions.GammaDistribution import GammaExponential
from distributions.NormalDistribution import Normal
import matplotlib.pyplot as plt

from mcmc import MCMC


def poisson_pmf(x, l):
    mass_function = poisson.pmf(x, l)
    return mass_function

#computer appreciate the difference in calculation power, if we use log
def logPoisson_pmf(data, p):
    return poisson.logpmf(data, p)

def likelihood_product(data, p, n, pmf):
    if (p < 0 or p > 1):
        return 0
    else:
        return pmf(data, p, n).prod()

class MCMC_Metropolis(MCMC.Engine):

    def plot_characteristics(self):
        plt.figure(1)
        plt.ylabel('current λ')
        plt.xlabel('Trial')
        plt.title('Trace')

    def init_dataset(self):
        self.trials = []
        self.data = []

        # Prior
        self.prior_param_alpha = []
        self.prior_param_beta = []

        #Posterior
        self.l_current=[]
        self.p_current=[]
        self.l_proposed = []
        self.p_proposed = []

        #Decision
        self.ratio = []
        self.p_move = []
        self.random = []
        self.l_accepted = []

    def construct_trace(self, variance_param, p_init, trials, prior, data):
        l_current = p_init
        trace = [l_current]
        self.init_dataset()
        accepted = 0
        burnin = 500
        for trial in range(1, trials+1, 1):
            # Compute posterior probability of current mu
            likelihood_current = poisson_pmf(data, l_current).prod()
            prior_current = prior.pdf(l_current)
            posterior_current = likelihood_current * prior_current

            # suggest new position
            symmetrical_d = Normal(l_current, variance_param)  # μ=λ current and σ=0.5
            l_proposal = symmetrical_d.sample()  # draw from symmetrical distribution with mu=l_current
            if(l_proposal<0):
                print("Proposed negative λ for the poisson distribution")
                continue

            if trial >= burnin:
                self.l_proposed.append(l_proposal)
                self.l_current.append(l_current)

            # Compute posterior probability of proposed mu
            likelihood_proposal = poisson_pmf(data, l_proposal).prod()
            prior_proposal = prior.pdf(l_proposal)
            posterior_proposed = likelihood_proposal * prior_proposal

            if trial >= burnin:
                self.p_current.append(posterior_current)
                self.p_proposed.append(posterior_proposed)

            # Accept proposal?
            ratio = posterior_proposed / posterior_current

            # compare a random number from the uniform U(0,1) with the rate p_accept
            random = np.random.uniform(0,1,1)
            p_move=min(ratio,1.0)
            accept = random < p_move

            if accept:
                # If the random number is less than the p_move,accept the proposed value of λ.
                l_current = l_proposal
                accepted += 1
            if trial >= burnin:
                trace.append(l_current)

            if trial >= burnin:
                #Build the dataset for demonstration reasons
                self.trials.append(trial)
                self.data.append(data)
                self.prior_param_alpha.append(prior.alpha)
                self.prior_param_beta.append(prior.beta)

                self.ratio.append(ratio)
                self.p_move.append(p_move)
                self.random.append(random)
                self.l_accepted.append(l_current)
        acceptance_rate = accepted / (trials-burnin)
        return acceptance_rate, trace

prior = GammaExponential(2.1, 1.0)
data=[5]
mu_init = prior.sample()
trials=10000
mcmc=MCMC_Metropolis()

final_mcmc_trace = mcmc.sampler2(trials=trials, tune_param=3.0, mu_init=mu_init, plot=True, prior=prior, data=data)
print(final_mcmc_trace)

data = {'Trial': mcmc.trials,
        'Data': mcmc.data,
        'a0': mcmc.prior_param_alpha,
        'b0': mcmc.prior_param_beta,
        'Posterior current': mcmc.p_current,
        'λ current': mcmc.l_current,
        'Posterior proposed': mcmc.p_proposed,
        'λ proposed': mcmc.l_proposed,
        'Ratio': mcmc.ratio,
        'p_move': mcmc.p_move,
        'Random': mcmc.random,
        'λ accepted': mcmc.l_accepted
        }
utils.export_csv(filepath='shark_attack_mcmc.csv', data=data)

df = pd.DataFrame(data, columns=data.keys())

utils.summarize_statistics(mcmc.l_accepted)

# calculate the Bayesian credible intervals for λ using the second method, the equal-tailed interval method.
quantiles=[0.010,0.025,0.050,0.250,0.500,0.750,0.950,0.975,0.990]
rows = []
for q in quantiles:
    rows.append({'Quantile': q, 'λ': df['λ accepted'].quantile(q)})
quantile_df = pd.DataFrame(rows)
print("Bayesian credible intervals")
print(quantile_df.head(10))