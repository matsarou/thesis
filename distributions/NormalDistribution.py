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
    __slots__ = ["mean", "var", "known_var"]

    def __init__(self, known_var, prior_mean=0, prior_var=1):
        self.mean = prior_mean
        self.var = prior_var
        self.known_var = known_var

    def update(self, data):
        var = np.var(data)
        mean = np.mean(data)
        n = len(data)
        denom = (1.0 / self.var + n / self.known_var)
        return NormalNormalKnownVar(self.known_var, (self.mean / self.var + sum(data) / self.known_var) / denom,
                                    1.0 / denom)

    def pdf(self, x):
        return stats.norm.pdf(x, self.mean, np.sqrt(self.var))

    def cdf(self, x):
        return stats.norm.cdf(x, self.mean, np.sqrt(self.var))

    def posterior(self, l, u):
        if l > u:
            return 0.0
        return self.cdf(u) - self.cdf(l)

    def plot(self, l=0.0, u=1.0):
        x = np.linspace(u, l, 1001)
        y = stats.norm.pdf(x, self.mean, np.sqrt(self.var))
        y = y / y.sum()
        plt.plot(x, y)
        plt.xlim((l, u))

    def predict(self, x):
        return stats.norm.cdf(x, self.mean, np.sqrt(self.var + self.known_var))

    def sample(self):
        return np.random.normal(self.mean, np.sqrt(self.var + self.known_var))


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
        while True:
            s=np.log(np.random.normal(self.mean, np.sqrt(self.var + self.known_var)))
            if not np.math.isnan(s):
                return s

class Normal(NormalNormalKnownVar):
    def __init__(self, prior_mean=0, prior_var=1):
        self.mean = prior_mean
        self.var = prior_var

    def sample(self):
        while True:
            random_value = np.random.normal(self.mean, self.var)
            if not np.math.isnan(random_value):
                return random_value

class NormalNormalKnownPrecision(NormalNormalKnownVar):
    __slots__ = ["mean", "var", "known_var"]

    def __init__(self, known_var, prior_mean=0):
        self.mean = prior_mean
        self.var = 1/np.sqrt(known_var)
        self.known_var = known_var

    def update(self):
        pass