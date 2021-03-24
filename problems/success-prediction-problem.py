import csv
import os
from functools import reduce

import numpy
import pandas as pd
from matplotlib import pyplot as plt

import linear_regression.regression_MCMC as mcmc_regression
import utils
import visuals
from distributions.NormalDistribution import NormalNormalKnownPrecision

MCMC_RESULTS_PATH = 'regression.csv'

def load_data(input_path, separator):
    df = pd.read_csv(input_path, sep=separator)
    return df

def start_MCMC(df):
    obs = list(df['Success'])
    predictors = list(df['Years in School'])
    results_df = mcmc_regression.process(trials=200, samples=15, burnin=50, observations=obs, predictors=predictors)
    print(results_df.head(10))
    results_df.to_csv(MCMC_RESULTS_PATH)

def credible_intervals(quantile, df):
    rows = []
    rows.append(df.quantile(quantile))
    return rows

def summarize_results(values):
    quantiles = [0.025, 0.25, 0.50, 0.75, 0.975]
    rows= {'Quantile':[], 'b0': [], 'b1': [], 'τ': []}
    for i in range(len(quantiles)):
        intervals=[]
        for j in range(len(values)):
            intervals.append(credible_intervals(quantiles[i], values[j]))
        rows.get('Quantile').append(str(quantiles[i])+'%')
        rows.get('b0').append(intervals[0][0])
        rows.get('b1').append(intervals[1][0])
        rows.get('τ').append(intervals[2][0])
    df = pd.DataFrame.from_dict(rows)
    print(df.head())

def summarize_statistics(values):
    statistics = {'Minimum': [], 'Maximum': [], 'Mean': [], 'Median': [], 'StDev': []}
    for j in range(len(values)):
        min, max, mean, median, std, var = utils.summarize_statistics(values[j], printStat=False)
        statistics.get('Minimum').append(min)
        statistics.get('Maximum').append(max)
        statistics.get('Mean').append(mean)
        statistics.get('Median').append(median)
        statistics.get('StDev').append(std)
    df = pd.DataFrame.from_dict(statistics)
    df.insert(0, '', ['b0','b1','τ'])
    print(df.head())

def plot_mcmc_results(df):
    mcmc_regression.plots(df['b0 Proposal Distribution'],df['b1 Proposal Distribution'], df['τ Proposal Distribution'], df['variance(1/τ)'])

def model_fit(param1_df,param2_df,param3_df,x):
    # print the likelihood L, the log likelihood lnL, and the −2 log likelihood −2 lnL
    model_likelihood = likelihood(param1_df, param2_df, param3_df, x, func=lambda x, y: x * y)
    model_likelihood_log2 = numpy.log2(likelihood(param1_df, param2_df, param3_df, x, func=lambda x, y: x + y))
    model_likelihood_negative_log2 = -2*model_likelihood_log2
    rows = zip(model_likelihood, model_likelihood_log2, model_likelihood_negative_log2)
    csv_header = ['Likelihood', 'ln(Likelihood)', '-2ln(Likelihood))']
    with open('model_fit.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        for row in rows:
            writer.writerow(row)
    return model_likelihood, model_likelihood_log2, model_likelihood_negative_log2


def likelihood(param1_df, param2_df, param3_df, x, func):
    model_likelihood = []
    trials = len(param1_df)
    for i in range(trials):
        print("i=", i)
        likelihood_observations = []
        for j in range(len(x)):
            distribution = NormalNormalKnownPrecision(known_var=param3_df[i],
                                                      prior_mean=param1_df[i] + param2_df[i] * x[j])
            likelihood_observations.append(distribution.pdf(x[j]))
        model_likelihood.append(reduce(func, likelihood_observations))
    return model_likelihood


if __name__ == "__main__":
    dataset = load_data(input_path='data.csv', separator=',')
    dataset = dataset.sort_values(by='Years in School')
    if not os.path.exists(MCMC_RESULTS_PATH):
        visuals.box_plot(dataset)
        visuals.pairplot(dataset, ['IQ', 'Years in School', 'Grit'], ['Success'])
        start_MCMC(dataset)
    mcmc_df = load_data(input_path=MCMC_RESULTS_PATH, separator=',')
    plot_mcmc_results(mcmc_df)
    summarize_results([mcmc_df['b0 Proposal Distribution'], mcmc_df['b1 Proposal Distribution'], mcmc_df['τ Proposal Distribution']])
    summarize_statistics([mcmc_df['b0 Proposal Distribution'], mcmc_df['b1 Proposal Distribution'], mcmc_df['τ Proposal Distribution']])
    # plt.show()
    mcmc_regression.plot_credible_intervals(list(mcmc_df['b0 Proposal Distribution']),
                                            list(mcmc_df['b1 Proposal Distribution']),
                                            list(mcmc_df['τ Proposal Distribution']),
                                            list(dataset['Years in School']))
    # plt.show()
    model_fit = model_fit(list(mcmc_df['b0 Proposal Distribution']),
                     list(mcmc_df['b1 Proposal Distribution']),
                     list(mcmc_df['τ Proposal Distribution']),
                     list(dataset['Years in School']))
    print("model_fit = ",model_fit)


