from numpy import random

import Utils
from BetaDistribution import BetaDistribution
from Utils import show_all
import copy

def inference(beta, data):
    betas = []
    size = len(data)
    window = int(size / 4)
    batches = [batch for batch in range(0, size, window)]
    for i in range(1, len(batches)):
        curr = batches[i]
        prev = batches[i - 1]
        batch = data[prev:curr]
        beta.update_params(batch)
        betas.append(copy.copy(beta))
    return betas


def plot_cdf(data):
    beta1 = BetaDistribution()
    beta1.update_params(data)
    beta1.show_plot({
        'color': 'b',
        'alpha': 0.4,
        'label': 'a=' + str(beta1.a) + ', b=' + str(beta1.b)
    })

def experiment_cdf():
    # plot the prior with an initial likelihood
    data = Utils.data_serial(0.4, 5)
    plot_cdf(data)

def experiment_pdf(data, color):
    beta = BetaDistribution()
    betas = []
    # Add the uniform
    betas.append(copy.copy(beta))
    # Inference
    more_pdfs = inference(beta, data)
    betas = betas + more_pdfs
    # Plot
    show_all(betas, color)


experiment_cdf()
# Collect data in an oredered list, first 20% of values=1 and then 80% of values=0
experiment_pdf(data=Utils.data_serial(success_prob=0.2, size=400), color='r')
# Collect data in an shuffled list
experiment_pdf(data=Utils.random_list(success_prob=0.2, size=400), color='r')
