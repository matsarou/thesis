import pandas as pd
from matplotlib import pyplot as plt
import visuals
import numpy as np
from linear_regression.Normal_b0 import Normal as normal_b0
from linear_regression.Normal_b1 import Normal as normal_b1
from linear_regression.Gamma_tu import Gamma as gamma_tu
import custom_queue as cq

def box_plot(df):
    df.plot.box()

def calculate_prediction_error(observation, predictor, b0,b1):
    error=observation-(b0+b1*predictor)
    return error,pow(error,2)


df = pd.read_csv('data.csv', sep=',')
# box_plot(df)
# visuals.pairplot(df, ['IQ','Years in School','Grit'], ['Success'])
# plt.show()

# 10 trials
# 15 samples
trials = 10
n = 15

#Give the MCMC starting values for b0[t],b1[t],tu[t]
b0_trial=cq.Queue(trials + 1)
b0_trial.put(6.0)
b1_trial=cq.Queue(trials + 1)
b1_trial.put(0.3)
tu_trial=cq.Queue(trials + 1)
tu_trial.put(None)

gamma_tu=gamma_tu(a=0.01, b=0.01)
normal_b0=normal_b0(prior_mean=0.0, prior_var=0.0001)
normal_b1=normal_b1(prior_mean=0.0, prior_var=0.0001)

observations = list(df['Success'])
predictors = list(df['Years in School'])
header = ['Trial','b0 Proposal Distribution','b1 Proposal Distribution','τ Proposal Distribution','SSE[t]']
lst = []
df1 = pd.DataFrame(lst, columns=header)
for i in range(0,trials):
    trial_error = calculate_prediction_error(observations[i], predictors[i], b0_trial.peek(), b1_trial.peek())
    lst.append([i, b0_trial.peek(), b1_trial.peek(), tu_trial.peek(), trial_error[0]])
    #update tu
    gamma_tu=gamma_tu.update(num=trials, y=observations, x=predictors, b0_trial=b0_trial.peek(), b1_trial=b1_trial.peek())
    tu_trial.put(gamma_tu.sample())
    #update b0
    normal_b0 = normal_b0.update(n=trials, y = observations, x = predictors, tu_trial=tu_trial.peek(), b1_trial=b1_trial.peek())
    b0_trial.put(normal_b0.sample())
    #update b1
    normal_b1 = normal_b1.update(y = observations, x = predictors, tu_trial=tu_trial.peek(), b0_trial=b0_trial.peek())
    b1_trial.put(normal_b1.sample())

    df2 = pd.DataFrame(lst, columns=header)
    df3 = df1.append(df2, ignore_index=True)

print(df3.head(10))

df3.to_csv('regression.csv')