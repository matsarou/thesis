import numpy as np
from scipy.stats import beta
from scipy.stats import binom

from distributions.BetaDistribution import BetaDistribution
from mcmc import MCMC

def bionomial_pmf(x, n, p):
    binomial = binom.pmf(x, n, p)
    return binomial

def get_moment_0(moment_1,mean):
    return moment_1*mean/(1-mean)

def get_correction_factor(params):
    numerator=params.get("beta current").pdf(params.get("p_proposed"))
    denominator=params.get("beta proposed").pdf(params.get("p_current"))
    return numerator, denominator

class MCMC_Hastings(MCMC.Engine):
    def construct_posterior(self, variance_param, mu_init, trials, prior):
        mu_current = mu_init
        posterior = [mu_current]
        accepted = 0
        for i in range(trials):
            # Prior
            prior_current = prior.pdf(mu_current)

            # Compute likelihood by multiplying probabilities of each data point
            n = 1
            likelihood_current = bionomial_pmf(1, n, mu_current).prod()
            # posterior density of observing the data under the hypothesis that p_c=mu_current
            posterior_current = likelihood_current * prior_current

            # suggest a second vale for p. We have a beta distribution that is centered over our current value, we can draw
            # a random value from it
            # a = variance_param * mu_current / (1 - mu_current)
            beta_center_p_current=BetaDistribution(get_moment_0(3, mu_current), 3)
            mu_proposal = beta_center_p_current.sample()
            prior_proposal = prior.pdf(mu_proposal)
            likelihood_proposal = bionomial_pmf(1, n, mu_proposal).prod()
            posterior_proposed = likelihood_proposal * prior_proposal

            # Accept proposal?
            beta_center_p_proposed = BetaDistribution(get_moment_0(3, mu_proposal), 3)
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
            posterior.append(mu_current)
        acceptance_rate = accepted / trials
        return acceptance_rate, posterior

transition_model = lambda x: [x[0],np.random.normal(x[1],0.5,(1,))]
print("transition_model = ", transition_model)

data = np.random.randn(20)
np.random.seed(123)
mcmc=MCMC_Hastings()
prior = beta(0.5, 0.5)
posterior_approximate = mcmc.sampler(samples=8, mu_init=0.5, prior=prior, plot=True)
print(posterior_approximate)