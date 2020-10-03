import random
import matplotlib.pyplot as plt
import numpy as np

def show_all(betas, color='b'):
    alpha = 0.0
    for i in range(len(betas)):
        beta = betas[i]
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