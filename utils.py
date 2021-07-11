import random
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, gamma

from distributions.NormalDistribution import NormalNormalKnownPrecisionConj,NormalLogNormalKnownPrecision


def show_all(pdfs, color='b'):
    alpha = 0.0
    for i in range(len(pdfs)):
        beta = pdfs[i]
        label = 'a=' + str(beta.a) + ', b=' + str(beta.b)
        alpha = alpha + (i+1)*0.10
        plt.plot(beta.span, beta.pdf(), alpha = alpha, color=color, label=label)
    plt.ylabel('density')
    plt.xlabel('conversion rate')
    plt.legend(numpoints=1, loc='upper right')
    plt.show()



def data_serial(success_prob = 0.20, size = 100):
    return [1 if i <= success_prob else 0 for i in np.linspace(0,1,size)]

def random_list(success_prob = 0.20, size = 100):
    x = [0,1]
    list = []
    successes = int(success_prob*size)
    list = list + [random.choice(x) if sum(list) < successes
            else 0
            for i in range(size)]
    if(sum(list) < successes):
        list = list[0:successes] + [1 for i in range(successes)] + list[2*successes:size]
    return list

def random_samples(min, max, dtype, samples):
    return np.linspace(min, max, samples, dtype=dtype)

def normal_data(mean, var, size):
    return norm.rvs(mean, var, size=size)

def gamma_data(gamma, size):
    data = gamma.rvs(size=size)
    return data

def plot_scatterplot(x, y, color):
    for i in range(10):
        txt='H'+str(i+1)+'=('+str(x[i])+','+str(y[i])+')'
        plt.annotate(txt, (x[i], y[i]), xytext=(10, 10), textcoords='offset points')
        plt.scatter(x, y, marker='o', color=color)
    plt.title('Combinations of mean and stdev')
    plt.xlabel('hypothesis for  mean')
    plt.ylabel('hypothesis for st.dev')
    plt.show()

def export_csv(filepath, data):
    df = pd.DataFrame(data, columns=data.keys())
    df.to_csv(filepath)

def plot_normal_pdf(pdfs, X, color='b'):
    alpha = 0.5
    for i in range(len(pdfs)):
        pdf = pdfs[i]
        label = 'mean=' + str(pdf.mean) + ', var=' + str(pdf.var)
        alpha = alpha + (i+1)*0.10
        plt.plot(X, pdf.pdf(X), alpha=alpha, color=color, label=label)
    plt.ylabel('density')
    plt.xlabel('conversion rate')
    plt.legend(numpoints=1, loc='upper right')
    plt.show()

def plot_multiple_pdfs(pdfs, color=None):
    alpha = 0.0
    for i in range(len(pdfs)):
        beta = pdfs[i]
        label = 'a=' + str(beta.a) + ', b=' + str(beta.b)
        # alpha = alpha + (i + 1) * 0.10
        if color==None:
            color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        plt.plot(beta.span, beta.pdf(), alpha=alpha, color=color, label=label)
    plt.legend(numpoints=1, loc='upper right')

def plot_gamma_pdfs(pdfs, X, color='b'):
    alpha = 1
    for i in range(len(pdfs)):
        pdf = pdfs[i]
        label = 'a=' + str(pdf.alpha) + ', b=' + str(pdf.beta)
        color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        plt.plot(X, pdf.pdf(X), alpha = alpha, color=color, label=label)
    plt.ylabel('Probability density')
    plt.xlabel('Hypotheses for λ')
    plt.legend(numpoints=1, loc='upper right')
    plt.show()

def summarize_statistics(data,printStat=True):
    #basic statistics
    mean1=np.mean(data)
    max1 = np.max(data)
    min1 = np.min(data)
    median1 = np.median(data)
    std1 = np.std(data)
    mode1 = np.max(data)
    # print block 1
    if printStat:
        print('Mean λ: ' + str(mean1))
        print('Max λ: ' + str(max1))
        print('Min λ: ' + str(min1))
        print('Median λ: ' + str(median1))
        print('Std λ: ' + str(std1))
        print('Mode λ: ' + str(mode1))
    return min1,max1,mean1,median1,std1,mode1

def standardize_data(data):
    mean = np.mean(data)
    std = np.std(data)
    new_data = data.map(lambda y: (y-mean)/std)
    return new_data

def likelihood_statistic_all_levels_x(param1,param2,param3,x=[]):
    log2_likelihood_observations = []
    for i in range(len(x)):
        norm_log_distribution = NormalNormalKnownPrecisionConj(known_prec=param3, mean=param1 + param2 * x[i], prec=param3)
        # norm_log_distribution = NormalLogNormalKnownPrecision(known_var=param3, prior_mean=param1 + param2 * x[i])
        log2_likelihood_observations.append(norm_log_distribution.pdf(x[i]))
    result = reduce(lambda x, y: x + y, log2_likelihood_observations)
    return result

if __name__ == "__main__":
    X = [12, 14, 18, 10, 13, 22, 17, 15, 16, 9, 19, 8, 20, 11, 21]
    lk_Years = -2*likelihood_statistic_all_levels_x(param1=3.269084, param2=0.785819, param3=0.001567, x=X)
    print("Years=",lk_Years)

    X_G = [2.2, 3.2, 3.4, 1.8, 2.8, 0.2, 4.4, 1, 4.6, 0.4, 1.60, 1.2, 0.6, 4.2]
    lk_Grit = -2*likelihood_statistic_all_levels_x(param1=3.487, param2=2.561413, param3=0.001083, x=X_G)
    print("Grit=",lk_Grit)