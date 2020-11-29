from copy import copy
import random
import matplotlib.pyplot as plt
import numpy as np

def design_plots():
    plt.figure(1)
    plt.ylabel('posterior')
    plt.xlabel('samples')
    plt.title('Trace')

def plot_trace(trace, label):
    plt.figure(1)
    trace = copy(trace)
    # x = np.linspace(0, 9, 9)
    color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
    plt.plot(trace, marker='o', color=color, label=label)
    plt.legend(numpoints=1, loc='upper right')

class Engine():
    def __init__(self):
        print("Initiate ",self)

    def sampler(self, samples=4, mu_init=.5, prior=None, plot=False):
        if plot:
            design_plots()
        tune_param = 0.5
        acceptance_rate = 0
        while tune_param <= 10 and (acceptance_rate < 0.2 or acceptance_rate > 0.5):
            acceptance_rate, posterior = self.construct_posterior(tune_param, mu_init, samples, prior)
            if plot:
                plot_trace(trace=posterior, label='rate=' + str(acceptance_rate * 100) + '%, b=' + str(tune_param))
            tune_param += 0.5
        if tune_param == 10.5:
            print("We didn't construct the best posterior trace.")
        print("Acceptance_rate is {}%".format(acceptance_rate * 100))
        print("The variance parameter is ", tune_param - 0.5)
        plt.show()
        return np.array(posterior)

    def construct_posterior(self, tune_param, mu_init, samples, prior):
        pass



