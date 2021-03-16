import os

import pandas as pd
import linear_regression.regression as mcmc_regression
import visuals
from matplotlib import pyplot as plt

MCMC_RESULTS_PATH = 'regression.csv'

def load_data(input_path, separator):
    df = pd.read_csv(input_path, sep=separator)
    return df

def start_MCMC(df):
    obs = list(df['Success'])
    predictors = list(df['Years in School'])
    results_df = mcmc_regression.process(trials=20000, samples=15, burnin=5000, observations=obs, predictors=predictors)
    print(results_df.head(10))
    results_df.to_csv(MCMC_RESULTS_PATH)

def plot_mcmc_results(df):
    mcmc_regression.plots(df['b0 Proposal Distribution'],df['b1 Proposal Distribution'], df['τ Proposal Distribution'], df['variance(1/τ)'])

if __name__ == "__main__":
    if not os.path.exists(MCMC_RESULTS_PATH):
        dataset = load_data(input_path='data.csv', separator=',')
        # Sort the dataset based on Iq
        dataset = dataset.sort_values(by='IQ')
        print(dataset.head(20))
        # visuals.box_plot(dataset)
        # visuals.pairplot(dataset, ['IQ', 'Years in School', 'Grit'], ['Success'])
        start_MCMC(dataset)
    mcmc_df = load_data(input_path=MCMC_RESULTS_PATH, separator=',')
    plot_mcmc_results(mcmc_df)
    plt.show()
