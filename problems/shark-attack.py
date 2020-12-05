import scipy
from scipy.stats import poisson
from seaborn import barplot

import utils
from distributions.GammaDistribution import GammaExponential
import matplotlib.pyplot as plt
import numpy as np
import random

def plot_prior_gamma_distributions(hypotheses, a_params, b_params):
    for i in range(len(a_params)):
        a=random.choice(a_params)
        b=random.choice(b_params)
        gamma = GammaExponential(alpha=a, beta=b)
        label = 'a=' + str(a) + ', b=' + str(b)
        color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        plt.plot(gamma.pdf(hypotheses), alpha = 1.0, color=color, label=label)
    plt.ylabel('density')
    plt.xlabel('Hypotheses for λ')
    plt.legend(numpoints=1, loc='upper right')
    plt.show()

def plot_gamma_distribution(hypotheses, a, b):
    gamma = GammaExponential(alpha=a, beta=b)
    label = 'a=' + str(a) + ', b=' + str(b)
    plt.plot(gamma.pdf(hypotheses), alpha=1.0, color='b', label=label)
    plt.ylabel('density')
    plt.xlabel('Hypotheses for λ')
    plt.legend(numpoints=1, loc='upper right')
    plt.show()

def plot_poisson_pmf(hypotheses, data):
    dist=scipy.stats.poisson(data)
    plt.bar(hypotheses, dist.pmf(hypotheses), color='grey', width=0.4)
    plt.ylabel('Probability')
    plt.xlabel('Attacks per year(X)')
    plt.show()

a=[0.5,1,2.1,3,7.5,9]
b=[0.5,1.0,2.0]
hypotheses= [0,1,1,2,3,3,3,4,4,5,6,14,10,12,4,6,0,9,19,2]
# plot_prior(hypotheses,a,b)

#Choose the prior with the highest pdf
a0=2.1
b0=1.0
# plot_gamma_distribution(hypotheses,a0,b0)

#Collect our data. This year we observed 5 shark attack
data=[5]
# hypotheses= [20,20,2,20,13,13,17,14,19,19.5,16,14,20,12,18,16,20,19,19,20]
plot_poisson_pmf(hypotheses, 5)