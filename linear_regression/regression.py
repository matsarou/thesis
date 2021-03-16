import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import custom_queue as cq
from linear_regression.Gamma_tu import Gamma as Gamma_tu
from linear_regression.Normal_b0 import Normal as Normal_b0
from linear_regression.Normal_b1 import Normal as Normal_b1

def queues(size):
    b0_trial = cq.Queue(size)
    b0_trial.put(6.0)
    b1_trial = cq.Queue(size)
    b1_trial.put(0.3)
    tu_trial = cq.Queue(size)
    tu_trial.put(None)
    variance_trial = cq.Queue(size)
    variance_trial.put(None)
    return b0_trial, b1_trial, tu_trial, variance_trial

def distributions():
    gamma_tu = Gamma_tu(a=0.01, b=0.01)
    normal_b0 = Normal_b0(known_var=0.0001, prior_mean=0.0, prior_var=0.0001)  # mean=0, precision=0.0001
    normal_b1 = Normal_b1(known_var=0.0001, prior_mean=0.0, prior_var=0.0001)  # mean=0, precision=0.0001
    return gamma_tu, normal_b0, normal_b1

def process(samples=10, trials=100, burnin=50, observations=[], predictors=[]):
    # Give the MCMC starting values for b0[t],b1[t],tu[t]
    b0_trial, b1_trial, tu_trial, variance_trial = queues(size=trials + 1)
    gamma_tu, normal_b0, normal_b1 = distributions()
    lst = []
    header = ['Trial', 'b0 Proposal Distribution', 'b1 Proposal Distribution', 'τ Proposal Distribution',
              'variance(1/τ)', 'SSE[t]']
    df1 = pd.DataFrame(lst, columns=header)
    for i in range(0, trials):
        # trial_error = calculate_prediction_error(observations[i], predictors[i], b0_trial.peek(), b1_trial.peek())
        trial_error = round(gamma_tu.sum_error(y=observations, x=predictors, b0=b0_trial.peek(), b1=b1_trial.peek()), 3)
        lst.append([i, b0_trial.peek(), b1_trial.peek(), tu_trial.peek(), variance_trial.peek(), trial_error])
        # update τ and variance
        gamma_tu = gamma_tu.update(num=samples, y=observations, x=predictors, b0_trial=b0_trial.peek(),
                                   b1_trial=b1_trial.peek())
        tu_trial.put(round(gamma_tu.sample(), 3))
        variance_trial.put(1 / tu_trial.peek())
        # update b0
        normal_b0 = normal_b0.update(n=samples, y=observations, x=predictors, tu_trial=tu_trial.peek(),
                                     b1_trial=b1_trial.peek())
        b0 = normal_b0.sample()
        b0_trial.put(round(b0, 3))
        # update b1
        normal_b1 = normal_b1.update(y=observations, x=predictors, tu_trial=tu_trial.peek(), b0_trial=b0_trial.peek())
        b1 = normal_b1.sample()
        b1_trial.put(round(b1, 3))

        df2 = pd.DataFrame(lst, columns=header)
        df3 = df1.append(df2, ignore_index=True)
    df3 = df3.drop(labels=range(0, burnin), axis=0)
    return df3


def plots(b0_post, b1_post, tu_post, variance):
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle('Traceplots')
    ax1.plot(b0_post[:100].values, marker='o', color='blue')
    ax1.set_title("MCMC trial for b0")
    ax2.plot(b1_post[:100].values, marker='o', color='red')
    ax2.set_title("MCMC trial for b1")
    ax3.plot(tu_post[:100].values, marker='o', color='green')
    ax3.set_title("MCMC trial for τ")
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].hist(x=b0_post, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    axs[0, 0].set_title('MCMC results for b0')
    axs[0, 1].hist(x=b1_post, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    axs[0, 1].set_title('MCMC results for b1')
    axs[1, 0].hist(x=tu_post, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    axs[1, 0].set_title('MCMC results for τ')
    axs[1, 1].hist(x=variance, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    axs[1, 1].set_title('MCMC results for var')
    for ax in axs.flat:
        ax.set(ylabel='Frequency')


