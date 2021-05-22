import csv
import os
from functools import reduce

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import linear_regression.regression_MCMC as mcmc_regression
import utils
import visuals
from distributions.NormalDistribution import NormalNormalKnownPrecision, NormalLogNormalKnownPrecision

MCMC_RESULTS_PATH = 'Success_regression_based_on_'

def load_data(input_path, separator):
    df = pd.read_csv(input_path, sep=separator)
    return df

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

def summarize_statistics(values,header,filepath):
    statistics = {'Minimum': [], 'Maximum': [], 'Mean': [], 'Median': [], 'StDev': []}
    for j in range(len(values)):
        min, max, mean, median, std, var = utils.summarize_statistics(values[j], printStat=False)
        statistics.get('Minimum').append(min)
        statistics.get('Maximum').append(max)
        statistics.get('Mean').append(mean)
        statistics.get('Median').append(median)
        statistics.get('StDev').append(std)
    df = pd.DataFrame.from_dict(statistics)
    df.insert(0, '', header)
    print(df.head())
    df.to_csv(filepath)

def plot_mcmc_results(df):
    mcmc_regression.plots(df['b0 Proposal Distribution'],df['b1 Proposal Distribution'], df['τ Proposal Distribution'], df['variance(1/τ)'])

def model_fit(param1_df,param2_df,param3_df,x):
    x=x.values.flatten()
    # print the likelihood L, the log likelihood lnL, and the −2 log likelihood −2 lnL
    model_likelihood,model_likelihood_log2 = likelihoods(param1_df, param2_df, param3_df, x, func=lambda x, y: x * y)
    model_likelihood_negative_log2 = list(-2*np.array(model_likelihood_log2))
    rows = zip(model_likelihood, model_likelihood_log2, model_likelihood_negative_log2)
    csv_header = ['Likelihood', 'ln(Likelihood)', '-2ln(Likelihood))']
    if not os.path.exists('model_fit_Years in School.csv'):
        with open('model_fit_Years in School.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)
            for row in rows:
                writer.writerow(row)
    return model_likelihood, model_likelihood_log2, model_likelihood_negative_log2


def likelihoods(param1_df, param2_df, param3_df, x, func):
    model_likelihood = []
    model_likelihood_log2 = []
    trials = len(param1_df)
    for i in range(trials):
        print("i=", i)
        likelihood_observations = []
        log2_likelihood_observations = []
        for j in range(len(x)):
            norm_distribution = NormalNormalKnownPrecision(known_var=param3_df[i],
                                                      prior_mean=param1_df[i] + param2_df[i] * x[j])
            likelihood_observations.append(norm_distribution.pdf(x[j]))
            norm_log_distribution = NormalLogNormalKnownPrecision(known_var=param3_df[i],
                                                           prior_mean=param1_df[i] + param2_df[i] * x[j])
            log2_likelihood_observations.append(norm_log_distribution.pdf(x[j]))
        model_likelihood.append(reduce(lambda x, y: x * y, likelihood_observations))
        model_likelihood_log2.append(reduce(lambda x, y: x + y, log2_likelihood_observations))
    return model_likelihood, model_likelihood_log2

def back_transform_mcmc_params(x, y, params1_df, params2_df):
    y_mean = np.mean(y)
    x_mean = np.mean(x)
    std_factor = np.std(y)/np.std(x)
    params1_df_transf=[]
    params2_df_transf=[]
    for i in range(len(params1_df)):
        params1_df_transf.append(y_mean - params2_df[i]*std_factor*x_mean)
        params2_df_transf.append(params2_df[i] * std_factor)
    return params1_df_transf,params2_df_transf

def lin_regression_MCMC(dataset,predictor_column):
    copy=pd.DataFrame.copy(dataset)
    y=copy['Success']
    x=copy[predictor_column]
    dataset['Success'] = utils.standardize_data(data=dataset['Success'])
    dataset[predictor] = utils.standardize_data(data=dataset[predictor])
    observations = list(dataset['Success'])
    predictors = list(dataset[predictor_column])
    results_df = mcmc_regression.process(trials=200, samples=15, burnin=50, observations=observations, predictors=predictors)
    params1_back_transf,params2_back_transf = back_transform_mcmc_params(x=x, y=y,
                                                                         params1_df=list(results_df['b0 Proposal Distribution']),
                                                                         params2_df=list(results_df['b1 Proposal Distribution']))
    results_df['b0 Proposal Distribution'] = params1_back_transf
    results_df['b1 Proposal Distribution'] = params2_back_transf
    print(results_df.head(5))
    return results_df

def analyze_model(estimations_filepath, predictors):
    mcmc_df = load_data(input_path=estimations_filepath, separator=',')
    # plot_mcmc_results(mcmc_df)
    summarize_results(
        [mcmc_df['b0 Proposal Distribution'], mcmc_df['b1 Proposal Distribution'], mcmc_df['τ Proposal Distribution']])
    summarize_statistics(
        values=[mcmc_df['b0 Proposal Distribution'], mcmc_df['b1 Proposal Distribution'], mcmc_df['τ Proposal Distribution']],
        header=['b0','b1','τ'],
        filepath="./statistics/proposals_summary_years_model.csv")
    plt.show()
    mcmc_regression.plot_credible_intervals(list(mcmc_df['b0 Proposal Distribution']),
                                            list(mcmc_df['b1 Proposal Distribution']),
                                            list(mcmc_df['τ Proposal Distribution']),
                                            xlabel=predictors.columns[0],ylabel='Success',
                                            x=predictors)
    plt.show()
    model_likelihood, model_likelihood_log2, model_likelihood_negative_log2 = model_fit(list(mcmc_df['b0 Proposal Distribution']),
                          list(mcmc_df['b1 Proposal Distribution']),
                          list(mcmc_df['τ Proposal Distribution']),
                          predictors)
    # print("model_fit = ", model_likelihood, model_likelihood_log2, model_likelihood_negative_log2)
    summarize_statistics(values=[model_likelihood, model_likelihood_log2, model_likelihood_negative_log2],
                         header=['L','lnL','−2 lnL'],
                         filepath="./statistics/fit_statistics_grit_model.csv")

if __name__ == "__main__":

    # visuals.box_plot(dataset)
    # visuals.pairplot(dataset, ['IQ', 'Years in School', 'Grit'], ['Success'])
    possible_predictors = ['Years in School']
    for predictor in possible_predictors:
        dataset = load_data(input_path='data.csv', separator=',')
        dataset = dataset.sort_values(by=predictor)
        copy=pd.DataFrame.copy(dataset)
        results_filepath=MCMC_RESULTS_PATH+predictor+'.csv'
        if not os.path.exists(results_filepath):
            mcmc_model = lin_regression_MCMC(copy, predictor)
            mcmc_model.to_csv(results_filepath)
        analyze_model(results_filepath, dataset[[predictor]])