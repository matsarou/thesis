# Taken from https://github.com/urigoren/conjugate_prior/blob/master/conjugate_prior/normal.py

import numpy as np
from scipy import stats

try:
    from matplotlib import pyplot as plt
except ModuleNotFoundError:
    import sys

    sys.stderr.write("matplotlib was not found, plotting would raise an exception.\n")
    plt = None


class NormalNormalKnownVar:
    __slots__ = ["mean", "var", "known_var", "data"]

    def __init__(self, known_var, prior_mean=0, prior_var=1, data = np.empty(1)):
        self.mean = prior_mean
        self.var = prior_var
        self.known_var = known_var
        self.data = np.log(data)

    def update(self, data):
        self.data = self.data + np.log(data)
        # self.var = np.var(data)
        self.mean = np.mean(data)
        n = len(self.data)
        denom = self.var + n*self.known_var
        return NormalNormalKnownVar(known_var = self.known_var,
                                    prior_mean = (self.mean * self.known_var + self.var*sum(self.data)) / denom,
                                    prior_var = denom,
                                    data = self.data)

    def update_params(self, known_var, mean, var):
        data = self.data
        n = len(data)
        denom = var + n*known_var
        return NormalNormalKnownVar(known_var = known_var,
                                    prior_mean = (mean * known_var + var*sum(data)) / denom,
                                    prior_var = denom,
                                    data = data)

    def param_mu(self):
        return self.mean

    def param_tu(self):
        return self.var

    def pdf(self, x):
        return stats.norm.pdf(x, self.mean, self.var)

    def cdf(self, x):
        return stats.norm.cdf(x, self.mean, self.var)

    def posterior(self, l, u):
        if l > u:
            return 0.0
        return self.cdf(u) - self.cdf(l)

    def predict(self, x):
        return stats.norm.cdf(x, self.mean, self.var + self.known_var)

    def sample(self):
        return np.random.normal(self.mean, self.var + self.known_var)

    def plot(self, x, color, xlabel, ylabel, label, show = False):
        if not label:
            label = 'mu=' + str(self.mean) + ', stdev=' + str(self.known_var)
        plt.plot(x, self.pdf(x), alpha=1, color=color, label=label)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.legend(numpoints=1, loc='upper right')
        if show:
            plt.show()


class NormalLogNormalKnownVar(NormalNormalKnownVar):
    def update(self, data):
        data = np.log(data)
        var = np.var(data)
        mean = np.mean(data)
        n = len(data)
        denom = (1.0 / self.var + n / self.known_var)
        return NormalLogNormalKnownVar(self.known_var, (self.mean / self.var + sum(data) / self.known_var) / denom,
                                       1.0 / denom)

    def predict(self, x):
        raise NotImplemented("No posterior predictive")

    def sample(self):
        raise np.log(np.random.normal(self.mean, self.var + self.known_var))


class Normal():
    def __init__(self, prior_mean=0, prior_var=1):
        self.mean = prior_mean
        self.var = prior_var

    def sample(self):
        return np.random.normal(self.mean, self.var)

    def update(self):
        pass