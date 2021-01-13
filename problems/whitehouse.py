import copy
from scipy.special import btdtri
from conjugates import beta_binomial
from distributions.BetaDistribution import BetaDistribution
import numpy as np
import matplotlib.pyplot as plt
import utils as ut

def plot_random_priors():
    betas = {}
    betas.update({'orange': BetaDistribution(a=1.0, b=1.0)})
    betas.update({'red': BetaDistribution(a=2.0, b=2.0)})
    betas.update({'blue': BetaDistribution(a=4.0, b=2.0)})
    betas.update({'black': BetaDistribution(a=2.0, b=4.0)})
    betas.update({'purple': BetaDistribution(a=0.5, b=0.5)})
    for key in betas.keys():
        pdf = betas.get(key)
        label = 'a=' + str(pdf.a) + ', b=' + str(pdf.b)
        plt.plot(pdf.span, pdf.pdf(), alpha = 1.0, color=key, label=label)
    plt.ylabel('Density')
    plt.xlabel('Hypotheses for p')
    plt.legend(numpoints=1, loc='upper right')
    plt.show()

def naive_hpd(post):
    HPD = np.percentile(post, [2.5, 97.5])
    plt.plot(HPD, [0, 0], label='HPD {:.2f} {:.2f}'.format(*HPD),
      linewidth=8, color='k')
    plt.legend(fontsize=16);
    plt.xlabel(r"$\theta$", fontsize=14)
    plt.gca().axes.get_yaxis().set_ticks([])

#Collect data
data=ut.random_list(success_prob=0.2, size=400)

#bayesian inference with the shortcut solution
beta = BetaDistribution()
betas = []
# Add the uniform
betas.append(copy.copy(beta))
# Inference
betas = betas + beta_binomial.inference(beta, data)
ut.plot_multiple_pdfs(betas, 'b')
plt.ylabel('Density')
plt.xlabel('Hypotheses for p')
plt.show()

#Report the results of the analysis
a_posterior=150
b_posterior=452
posterior=BetaDistribution(a=a_posterior, b=b_posterior)
plt.plot(posterior.span, posterior.pdf(), alpha = 1.0, color='b', label='posterior a=150, b=452')
# ucb = btdtri(a_posterior, b_posterior, 0.975)
# lcb = btdtri(a_posterior, b_posterior, 0.025)
# plt.fill_between(lcb, ucb, posterior.pdf(),color='yellow', alpha=1.0)
naive_hpd(posterior)
plt.ylabel('Density')
plt.xlabel('Hypotheses for p')
plt.legend(numpoints=1, loc='upper right')
plt.show()