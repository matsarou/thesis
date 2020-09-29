from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

class BetaDistribution:

    def __init__(self):
        self.a = 1
        self.b = 1
        self.n = 0
        self.data = []
        # self.array = np.linspace(self.ppf(0.01), self.ppf(0.99), 200)

    def update_params(self, data = []):
        self.data = self.data +data
        self.n = len(self.data)
        self.a = self.parameter_a()
        self.b = self.parameter_b()

    def mean(self):
        tr = sum(self.data)
        fl = self.n -tr
        # return tr * self.n / (tr + fl)
        return stats.moment(self.data, 1) * self.n

    def stdev(self):
        return stats.moment(self.data, 2) * self.n

    def pdf(self):
        return stats.beta.pdf(self.data, self.a, self.b)

    def ppf(self, x):
        return stats.beta.ppf(x, self.a, self.b)

    #  computes a binomial cumulative distribution function at each of the values in x using the corresponding number
    #  of trials in n and the probability of success for each trial in p.
    def cdf(self, loc=0, scale=1):
        return stats.beta.cdf(self.data, self.a, self.b, loc, scale)

    def parameter_a(self):
        return self.mean()
        # b = self.parameter_b()
        # mean = self.mean()
        # return b * mean / (1 - mean)

    def parameter_b(self):
        return self.stdev()
        # mean = self.mean()
        # stdev = self.stdev()
        # return mean - 1 + mean * pow(1 - mean, 2) / pow(stdev, 2)

    # The z-score is 1.96 for a 95% confidence interval.
    def conf_interval(self, z_score = 1.96):
        # The size of the successes
        mean = self.mean()
        # standard error
        se = np.sqrt(mean * (1 - mean) / self.n)
        # the confidence interval
        lcb = mean - z_score * se  # lower limit of the CI
        ucb = mean + z_score * se  # upper limit of the CI
        return lcb, ucb

    def show_plot(self, params):
        fig, ax = plt.subplots(1, 1)
        # x = np.linspace(0,1,200)
        ax.plot(self.data, self.pdf(), color=params['color'],alpha=params['alpha'], label=params['label'])
        # if self.mean() > 0.0 and self.n > 0:
        #     lcb, ucb = self.conf_interval()
        #     ax.fill_between(self.data, lcb, ucb, color='b', alpha=.1)
        plt.ylabel('density')
        plt.xlabel('conversion rate')
        plt.legend(numpoints=1, loc='upper right')
        plt.show()
