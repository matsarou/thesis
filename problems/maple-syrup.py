from distributions.NormalDistribution import NormalNormalKnownVar
from utils import random_samples, plot_scatterplot
import numpy as np
import pandas as pd

def likelihood(x, mu, stdev):
    normal = NormalNormalKnownVar(stdev, mu, stdev)
    return round(normal.pdf(x),2)

x = random_samples(min=0, max=100, samples=50)
mu = 10.0
stdev = 2.3
normal = NormalNormalKnownVar(stdev, mu, stdev)
normal.plot(x=x, color='b', xlabel = 'Millions of Gallons Produced (x)', ylabel='Density', show = True)

means=[9.5,9.6,10.0,10.3,10.4,10.5,10.8,11.6,11.9,12.5]
stdevs=[1.97,1.85,1.71,1.85,1.9,2.06,1.78,2.2,2.09,1.79]
plot_scatterplot(means,stdevs, 'b')

# Identify the alternative hypotheses
hypos = list(zip(means, stdevs))
print("hypotheses=",hypos)

# Set priors
Pr = [1/len(hypos) for hypo in hypos]
print("priors=",Pr)

# Collect data
x = 10.2

#Likelihood
Lr = [likelihood(x, means[i], stdevs[i]) for i in range(len(means))]
print('likelihood=',Lr)

#Pr*Lr
nominator = [Pr[i] * Lr[i] for i in range(len(Pr))]

#Normalized nominator
denominator = np.sum(nominator)

#Posterior
posterior = nominator / denominator
posterior=np.around(posterior, decimals=4)
print('posterior=',posterior)

#write to dataframe
data = {'mean':  means,
        'stdev':stdevs,
        'Prior': Pr,
        'Data': x,
        'Likelihood':Lr,
        'Pr[i] * Lr[i]': nominator,
        'Denominator': denominator,
        'Posterior': posterior
        }

df = pd.DataFrame (data, columns = data.keys())
print(df.head(10))

df.to_csv('maple-syrup-data.csv')





