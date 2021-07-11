from copy import copy
import random
import matplotlib.pyplot as plt
import numpy as np

class Engine():
    def __init__(self):
        print("Initiate ",self)
        self.init_dataset()

    def sampler(self, trials=4, mu_init=.5, prior=None, tune_param = 0.5, max_tune_param=10.0, data=None, plot=False):
        if plot:
            self.plot_characteristics()
        acceptance_rate = 0
        while tune_param <= max_tune_param and (acceptance_rate < 0.2 or acceptance_rate > 0.5):
            acceptance_rate, traces = self.construct_trace(tune_param, mu_init, trials, prior, data)
            if plot:
                self.plot_trace(trace=traces, label='acc. rate=' + str(acceptance_rate * 100) + '%, σ=' + str(tune_param))
            tune_param += 0.5
        if tune_param == max_tune_param+0.5:
            print("We didn't construct the best posterior trace.")
        print("Acceptance_rate is {}%".format(acceptance_rate * 100))
        print("The variance parameter is ", tune_param - 0.5)
        plt.show()
        return np.array(traces)

    def plot_trace(self, trace, label):
        plt.figure(1)
        trace = copy(trace)
        # x = np.linspace(0, 9, 9)
        color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        plt.plot(trace, marker='o', color=color, label=label)
        plt.legend(numpoints=1, loc='upper right')
        plt.xlabel('trial')
        plt.ylabel('value')

    def sampler2(self, trials=4, mu_init=.5, prior=None, tune_param = 0.5, max_tune_param=10.0, data=None, plot=False):
        if plot:
            self.plot_characteristics()
        acceptance_rate, traces = self.construct_trace(tune_param, mu_init, trials, prior, data)
        if plot:
            self.plot_trace(trace=traces, label='acc. rate=' + str(acceptance_rate * 100) + '%, σ=' + str(tune_param))
        print("Acceptance_rate is {}%".format(acceptance_rate * 100))
        print("The variance parameter is ", tune_param - 0.5)
        plt.show()
        return np.array(traces)

    def construct_trace(self, tune_param, mu_init, samples, prior, data):
        pass

    def init_dataset(self):
        pass

    def plot_characteristics(self):
        pass



