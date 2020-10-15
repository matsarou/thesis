import random

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta
from copy import copy

from scipy.stats import norm, binom

from BetaDistribution import BetaDistribution

sns.set_style('white')
sns.set_context('talk')

np.random.seed(123)

def bionomial_pmf(x, n, p):
    binomial = binom.pmf(x, n, p)
    return binomial

def sampler(samples=4, mu_init=.5, plot=False):
    if plot:
        design_plots()
    tune_param = 0.5
    acceptance_rate = 0
    while tune_param <= 10 and (acceptance_rate < 0.2 or acceptance_rate > 0.5):
        acceptance_rate, posterior = construct_posterior(tune_param, mu_init, samples)
        if plot:
            plot_trace(trace=posterior, label='rate='+str(acceptance_rate*100)+'%, b='+str(tune_param))
        tune_param += 0.5
    if tune_param == 10.5:
        print("We didn't construct the best posterior trace.")
    print("Acceptance_rate is {}%".format(acceptance_rate*100))
    print("The variance parameter is ", tune_param-0.5)
    plt.show()
    return np.array(posterior)


def design_plots():
    plt.figure(1)
    plt.ylabel('posterior')
    plt.xlabel('samples')
    plt.title('Trace')


def construct_posterior(variance_param, mu_init, samples):
    p_current = mu_init
    posterior = [p_current]
    accepted = 0
    for i in range(samples):
        # suggest a second vale for p. We have a beta distribution that is centered over our current value, we can draw
        # a random value from it
        a = variance_param * p_current / (1 - p_current)
        p_proposal = np.random.beta(a, variance_param, size=1)[0]

        # Compute likelihood by multiplying probabilities of each data point
        n = 1
        likelihood_current = bionomial_pmf(1, n, p_current)
        likelihood_proposal = bionomial_pmf(1, n, p_proposal)

        # Compute prior probability of current and proposed mu
        rv = beta(0.5, 0.5)
        prior_current = rv.pdf(p_current)
        prior_proposal = rv.pdf(p_proposal)

        posterior_current = likelihood_current * prior_current
        posterior_proposed = likelihood_proposal * prior_proposal

        # Accept proposal?
        p_accept = posterior_proposed / posterior_current

        # Usually would include prior probability, which we neglect here for simplicity
        accept = np.random.rand() < p_accept

        if accept:
            # Update position
            p_current = p_proposal
            accepted += 1
        posterior.append(p_current)
    acceptance_rate = accepted / samples
    return acceptance_rate, posterior


def calc_posterior_analytical(data, x, mu_0, sigma_0):
    sigma = 1.
    n = len(data)
    mu_post = (mu_0 / sigma_0**2 + data.sum() / sigma**2) / (1. / sigma_0**2 + n / sigma**2)
    sigma_post = (1. / sigma_0**2 + n / sigma**2)**-1
    return norm(mu_post, np.sqrt(sigma_post)).pdf(x)

def plot_trace(trace, label):
    plt.figure(1)
    trace = copy(trace)
    # x = np.linspace(0, 9, 9)
    color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
    plt.plot(trace, marker='o', color=color, label=label)
    plt.legend(numpoints=1, loc='upper right')


transition_model = lambda x: [x[0],np.random.normal(x[1],0.5,(1,))]
print("transition_model = ", transition_model)

data = np.random.randn(20)
np.random.seed(123)
posterior_approximate = sampler(samples=8, mu_init=0.5, plot=True)
print(posterior_approximate)