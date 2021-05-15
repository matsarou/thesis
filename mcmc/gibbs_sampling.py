from distributions.GammaDistribution import GammaExponential
from distributions.NormalDistribution import NormalNormalKnownPrecisionConj
import pandas as pd
from matplotlib import pyplot as plt

from mcmc import MCMC

class MCMC_Gibbs(MCMC.Engine):
    def plot_characteristics(self):
        plt.figure(1)
        plt.ylabel('Proposal')
        plt.xlabel('Trial')
        plt.title('Traces')

    def prior_t(self, a, b):
        self.prior_tu = GammaExponential(alpha=a, beta=b)
        return self.prior_tu

    def prior_mean(self, tu, mu0, tu0):
        self.prior_mean = NormalNormalKnownPrecisionConj(tu0, mu0, tu)
        return self.prior_mean

    def sampler(self, trials, data):
        normal_mu = self.prior_mean
        gamma_tu = self.prior_tu

        my_df = []
        my_df.append({
            'Trial': 0,
            'Data': data,
            'n': len(data),
            'μ': normal_mu.mean,
            'τ': normal_mu.prec,
            'μ_prop': None,
            'a': gamma_tu.alpha,
            'b': gamma_tu.beta,
            'τ_prop': 40.123
        })

        mu_prop_traces = []
        tu_prop_traces = []
        for it in range(1, trials):
            normal_mu, mu_prop, gamma_tu, tu_prop = self.sampling(normal_mu=normal_mu, gamma_tu=gamma_tu, data=data)
            d = {
                'Trial': it,
                'Data': data,
                'n': len(data),
                'μ': normal_mu.mean,
                'τ': normal_mu.prec,
                'μ_prop': mu_prop,
                'a': gamma_tu.alpha,
                'b': gamma_tu.beta,
                'τ_prop': tu_prop
            }
            my_df.append(d)
            mu_prop_traces.append(mu_prop)
            tu_prop_traces.append(tu_prop)
        # export to a csv
        my_df = pd.DataFrame(my_df)
        my_df.to_csv("maple_syrup_mcmc.csv")
        return mu_prop_traces, tu_prop_traces

    def sampling(self, normal_mu, gamma_tu, data):
        #proposed posterior for estimation of mean
        normal_mu = normal_mu.update(data)

        # proposed posterior for estimation of variance
        while True:
            tu_prop = gamma_tu.sample()
            if tu_prop > 0:
                break
        while True:
            mu_prop = normal_mu.sample()
            if mu_prop > 0:
                break
        try:
            gamma_tu = gamma_tu.update_params(len(data)/2, sum([pow((d - mu_prop), 2) for d in data])/2)
            tau_prop = gamma_tu.sample()
        except OverflowError:
            print('errror = ',mu_prop)

        return normal_mu, mu_prop, gamma_tu, tau_prop
