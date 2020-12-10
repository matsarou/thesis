import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, gamma

from distributions.NormalDistribution import NormalNormalKnownVar

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

def random_samples(min, max, samples):
    return np.linspace(min, max, samples)

def normal_data(mean, stdev, size):
    data = norm.rvs(10.0, 2.5, size=500)
    return data

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

def plot_normal_pdf(pdfs, mu, color='b'):
    alpha = 0.0
    for i in range(len(pdfs)):
        pdf = pdfs[i]
        label = 'mean=' + str(pdf.mean) + ', var=' + str(pdf.var)
        alpha = alpha + (i+1)*0.10
        plt.plot(pdf.pdf(mu), alpha = alpha, color=color, label=label)
    plt.ylabel('density')
    plt.xlabel('conversion rate')
    plt.legend(numpoints=1, loc='upper right')
    plt.show()

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

def summarize_statistics(data):
    #basic statistics
    mean1=np.mean(data)
    max1 = np.max(data)
    min1 = np.min(data)
    median1 = np.median(data)
    std1 = np.std(data)
    var1 = np.var(data)
    # print block 1
    print('Mean λ: ' + str(mean1))
    print('Max λ: ' + str(max1))
    print('Min λ: ' + str(min1))
    print('Median λ: ' + str(median1))
    print('Std λ: ' + str(std1))
    print('Var λ: ' + str(var1))