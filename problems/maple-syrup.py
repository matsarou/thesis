from conjugates import normal_normal as normal_normal_conj
from distributions.NormalDistribution import NormalNormalKnownVar
from mcmc.gibbs_sampling import MCMC_Gibbs
from utils import random_samples
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

def plot_multiple_pdfs(pdfs, x):
    for i in range(len(pdfs)):
        pdf = pdfs[i]
        label = 'μ=' + str(pdf.mean) + ', σ=' + str(pdf.var)
        color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        plt.plot(x, pdf.pdf(x), alpha=1.0, color=color, label=label)
    plt.legend(numpoints=1, loc='upper right')
    plt.xlabel('Hypotheses for μ')
    plt.ylabel('Density')
    plt.show()

def likelihood(x, mu, stdev):
    normal = NormalNormalKnownVar(stdev, mu, stdev)
    return round(normal.pdf(x),2)

x = random_samples(min=0, max=100, samples=50,dtype=np.int16)
mu = 10.0
stdev = 2.3
normal = NormalNormalKnownVar(stdev, mu, stdev)
# normal.plot(x, color='b', xlabel = 'Millions of Gallons Produced (x)', ylabel='Density', show = True, label='')



def method_name():
    means = [9.5, 9.6, 10.0, 10.3, 10.4, 10.5, 10.8, 11.6, 11.9, 12.5]
    stdevs = [1.97, 1.85, 1.71, 1.85, 1.9, 2.06, 1.78, 2.2, 2.09, 1.79]
    # plot_scatterplot(means, stdevs, 'b')
    # Identify the alternative hypotheses
    hypos = list(zip(means, stdevs))
    print("hypotheses=", hypos)
    # Set priors
    Pr = [1 / len(hypo) for hypo in hypos]
    print("priors=", Pr)
    # Collect data
    x = 10.2
    # Likelihood
    Lr = [likelihood(x, means[i], stdevs[i]) for i in range(len(means))]
    print('likelihood=', Lr)
    # Pr*Lr
    nominator = [Pr[i] * Lr[i] for i in range(len(Pr))]
    # Normalized nominator
    denominator = np.sum(nominator)
    # Posterior
    posterior = nominator / denominator
    posterior = np.around(posterior, decimals=4)
    print('posterior=', posterior)
    # write to dataframe
    data = {'mean': means,
            'stdev': stdevs,
            'Prior': Pr,
            'Data': x,
            'Likelihood': Lr,
            'Pr[i] * Lr[i]': nominator,
            'Denominator': denominator,
            'Posterior': posterior
            }
    df = pd.DataFrame(data, columns=data.keys())
    print(df.head(10))
    df.to_csv('maple-syrup-data.csv')

# Bayesian inference with the Normal-Normal conjugate
def conjugate_solution():
    mus = random_samples(5, 30, np.int16, 10)
    # Plot and decide on the mean
    pdfs = [NormalNormalKnownVar(known_var=2.0, prior_mean=mu, prior_var=2.0) for mu in mus]
    # plot_multiple_pdfs(pdfs, x)

    mean = 12.0
    tu0 = 0.0625 # prior_var = 1/np.sqrt(tu0) = 4.0
    known_tu = 0.25  # var=1/np.sqrt(known_tu) = 2.0

    x = [10.2]    # Gather data
    normal_normal_conj.experiment_pdf(mean, tu0, known_tu, x, 'b')

def mcmc_gibbs_sampling(trials = 5):
    mcmc = MCMC_Gibbs()
    # Assign prior to each hyperparameter μ and τ
    tu0 = 0.0625
    mu0 = 12
    prior_mu = mcmc.prior_mean(mu0=mu0, tu0=tu0, tu=40.123)
    a0 = 25.0  # shape parameter α0=25
    b0 = 0.5  # rate parameter β0=0.5
    prior_tu = mcmc.prior_t(a=a0, b=b0)

    # plot the priors
    x = np.linspace(0, 100, 60)
    prior_mu.plot(x, color='b', label='μ prior normal distr: μ0=' + str(mu0) + ', τ0=' + str(tu0))
    prior_tu.plot(x, color='r', label='τ prior gamma distr: a0=' + str(a0) + ', b0=' + str(b0))
    plt.show()

    # Gather data
    data = [10.2]

    # MCMC with n=5 trials
    mu_prop_traces, tu_prop_traces = mcmc.sampler(trials, data)
    mcmc.plot_trace(trace=mu_prop_traces, label='Proposals for μ')
    mcmc.plot_trace(trace=tu_prop_traces, label='Proposals for τ')
    plt.show()


# method_name()
# conjugate_solution() #conjugate solution
mcmc_gibbs_sampling(trials = 1000) #MCMC with Gibbs sampling






