import numpy as np
from scipy.stats import beta
from scipy.stats import binom
import matplotlib.pyplot as plt

import utils
from distributions.BetaDistribution import BetaDistribution
from mcmc import MCMC

def bionomial_pmf(x, n, p):
    binomial = binom.pmf(x, n, p)
    return binomial

def get_moment_0(moment_1,mean):
    if mean==1:
        mean=0.999 # so as not to throw exception
    return moment_1*mean/(1-mean)

def get_correction_factor(params):
    numerator=params.get("beta current").pdf(params.get("p_proposed"))
    denominator=params.get("beta proposed").pdf(params.get("p_current"))
    return numerator, denominator

class MCMC_Hastings(MCMC.Engine):
    def plot_characteristics(self):
        plt.figure(1)
        plt.ylabel('current p')
        plt.xlabel('Trial')
        plt.title('Trace')

    def init_dataset(self):
        self.trials = []
        self.data = []

        # Prior
        self.prior_param_alpha = []
        self.prior_param_beta = []

        #Posterior
        self.mu_current=[]
        self.p_current=[]
        self.mu_proposed = []
        self.p_proposed = []

        #Decision
        self.ratio = []
        self.p_move = []
        self.random = []
        self.mu_accepted = []

    def construct_trace(self, tune_param=3.0, mu_init=0.5, trials=10, prior=BetaDistribution(3, 3), data=None):
        mu_current = mu_init
        traces = [mu_current]
        self.init_dataset()
        accepted = 0
        for i in range(trials):
            # Prior centered at mu-current
            prior_current = prior.pdf(mu_current)
            n = 1
            likelihood_current = binom.pmf(data, n, mu_current).prod()# Compute likelihood by multiplying probabilities of each data point
            # posterior density of observing the data under the hypothesis that p_c=mu_current
            posterior_current = likelihood_current * prior_current

            # suggest a second vale for p. We have a beta distribution that is centered over our current value, we can draw
            # a random value from it
            beta_center_p_current=BetaDistribution(get_moment_0(tune_param, mu_current), tune_param)
            mu_proposal = beta_center_p_current.sample()
            prior_proposal = prior.pdf(mu_proposal)
            likelihood_proposal = binom.pmf(data, n, mu_current).prod()
            posterior_proposed = likelihood_proposal * prior_proposal

            # Accept proposal?
            beta_center_p_proposed = BetaDistribution(get_moment_0(tune_param, mu_proposal), tune_param)
            params={
                "p_current":mu_current,
                "beta current":beta_center_p_current,
                "p_proposed":mu_proposal,
                "beta proposed": beta_center_p_proposed
            }
            numerator, denominator = get_correction_factor(params)
            ratio = posterior_proposed*numerator / posterior_current*denominator

            # compare a random number from the uniform U(0,1) with the rate p_accept
            random = np.random.uniform(0, 1, 1)
            p_move = min(ratio, 1.0)
            accept = random < p_move

            if accept:
                # Update position
                mu_current = mu_proposal
                accepted += 1
            traces.append(mu_current)

            #Build the dataset for demonstration reasons
            self.trials.append(i)
            self.data.append(data)
            self.prior_param_alpha.append(prior.a)
            self.prior_param_beta.append(prior.b)
            self.mu_current.append(mu_current)
            self.p_current.append(posterior_current)
            self.mu_proposed.append(mu_proposal)
            self.p_proposed.append(posterior_proposed)
            self.ratio.append(ratio)
            self.p_move.append(p_move)
            self.random.append(random)
            self.mu_accepted.append(mu_current)
        acceptance_rate = accepted / trials
        return acceptance_rate, traces

transition_model = lambda x: [x[0],np.random.normal(x[1],0.5,(1,))]
print("transition_model = ", transition_model)

data = [0]
np.random.seed(123)
mcmc=MCMC_Hastings()
prior = beta(0.5, 0.5)
final_mcmc_trace = mcmc.sampler(trials=20, mu_init=0.5, tune_param = 0.5, prior=prior, data=data,plot=True)
print(final_mcmc_trace)

dataset_columns = {'Trial': mcmc.trials,
        'Data': mcmc.data,
        'a0': mcmc.prior_param_alpha,
        'b0': mcmc.prior_param_beta,
        'Posterior current': mcmc.p_current,
        'λ current': mcmc.mu_current,
        'Posterior proposed': mcmc.p_proposed,
        'λ proposed': mcmc.mu_proposed,
        'Ratio': mcmc.ratio,
        'p_move': mcmc.p_move,
        'Random': mcmc.random,
        'λ accepted': mcmc.mu_accepted
        }
utils.export_csv(filepath='whitehouse_mcmc.csv', data=dataset_columns)