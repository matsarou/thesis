import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import custom_queue as cq
from distributions.NormalDistribution import Normal
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
    gamma_tu = Gamma_tu(alpha=0.01, beta=0.01)
    normal_b0 = Normal_b0(known_var=0.0001, prior_mean=0.0, prior_var=0.0001)  # mean=0, precision=0.0001, stdev=100
    normal_b1 = Normal_b1(known_var=0.0001, prior_mean=0.0, prior_var=0.0001)  # mean=0, precision=0.0001, stdev=100
    return gamma_tu, normal_b0, normal_b1

def credible_intervals(y, credible_interval = 95):
    y_up = np.percentile(y, credible_interval)
    y_lo = np.percentile(y, 100 - credible_interval)
    return y_up, y_lo

def y_single_level(param1_list,param2_list,x):
    y = lambda b0, b1: b0 + b1 * x
    return [y(param1_list[i], param2_list[i]) for i in range(len(param1_list))]

def posterior_y(param1_df, param2_df, x):
    y_up=[]
    y_lo=[]
    for x_i in range(len(x)):
        y_i=[] #size 15
        for trial in range(len(param1_df)): # iterate through trials
            y_i.append(param1_df[trial] + param2_df[trial] * x_i)
        up, lo = credible_intervals(y=y_i)
        y_up.append(up)
        y_lo.append(lo)
    return y_up, y_lo

def posterior_predictive_y(param1_df, param2_df, param3_df, x):
    y_predicted_up=[]
    y_predicted_lo=[]
    for x_i in range(len(x)):
        y_i=[] #size 15
        for trial in range(len(param3_df)): # iterate through trials
            mean = param1_df[trial] + param2_df[trial] * x_i
            tu = np.math.sqrt(1 / param3_df[trial])
            posterior_distribution = Normal(mean, tu)
            y_i.append(posterior_distribution.sample())
        up, lo = credible_intervals(y=y_i)
        y_predicted_up.append(up)
        y_predicted_lo.append(lo)
    return y_predicted_up, y_predicted_lo

def process(samples=10, trials=100, burnin=50, observations=[], predictors=[]):
    # Give the MCMC starting values for b0[t],b1[t],tu[t]
    b0_trial, b1_trial, tu_trial, variance_trial = queues(size=trials + 1)
    gamma_tu, normal_b0, normal_b1 = distributions()
    lst = []
    predictions=[]
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
        tu=gamma_tu.rvs(size=1)[0]
        tu_trial.put(tu)
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

def plot_credible_intervals(param1_df,param2_df,param3_df, x=[]):
    y_up, y_lo = posterior_y(param1_df, param2_df, x)
    y_predicted_up, y_predicted_lo=posterior_predictive_y(param1_df, param2_df, param3_df, x)
    plt.plot(x, y_up, linestyle='--', dashes=(5, 5), alpha=1.0, color='b')
    plt.plot(x, y_lo, linestyle='--', dashes=(5, 5), alpha=1.0, color='b')
    plt.plot(x, y_predicted_up, linestyle='--', dashes=(5, 5),alpha=1.0, color='r')
    plt.plot(x, y_predicted_lo, linestyle='--', dashes=(5, 5), alpha=1.0, color='r')
    plt.ylabel('Success')
    plt.xlabel('Years')
    plt.show()

def plots(b0_post, b1_post, tu_post, variance):
    x = np.arange(0, 100)
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle('Traceplots')
    ax1.plot(x,b0_post[:100].values, marker='o', color='blue')
    ax1.set_title("MCMC trial for b0")
    ax2.plot(x,b1_post[:100].values, marker='o', color='red')
    ax2.set_title("MCMC trial for b1")
    ax3.plot(x, tu_post[:100].values, marker='o', color='green')
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


