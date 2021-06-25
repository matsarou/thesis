import math

import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, Trace_ELBO
from torch.distributions import constraints
import matplotlib.pyplot as plt
from pyro.optim import Adam #the problem is on Pycharm

# Question: “What is the probability
# that any famous person (like Shaq) can drop by the White
# House without an appointment?”

pyro.set_rng_seed(101)

data=[]
# Collect some data.
# - 1 for success
# - 0 for failure
for i in range(0, 6):
    data.append(torch.tensor(1.0))
for i in range(0, 4):
    data.append(torch.tensor(0.0))

pyro.clear_param_store()

def model(data):
    # define the hyperparameters that control the beta prior
    alpha0 = torch.tensor(10.0)
    beta0 = torch.tensor(10.0)
    # register a distribution named ”latent_fairness” as a learnable value for Pyro.
    f = pyro.sample("latent_fairness", dist.Beta(alpha0,beta0))

    # Condition the model on the observed data
    # Determine the binomial likelihood of the observed data, assuming each hypothesis is true
    # Register every observation as a learnable value
    for i in range(len(data)):
        sensor = pyro.sample(f'obs_{i}',
                             dist.Binomial(probs=f),
                             obs=data[i])

def guide(data):
    # register the two variational parameters with Pyro
    # - both parameters will have initial value 15.0.
    # - because we invoke constraints.positive, the optimizer
    # will take gradients on the unconstrained parameters
    # (which are related to the constrained parameters by a log)
    alpha_q = pyro.param("alpha_q", torch.tensor(15.0),
                         constraint=constraints.positive)
    beta_q = pyro.param("beta_q", torch.tensor(15.0),
                        constraint=constraints.positive)
    # sample latent_fairness from the distribution Beta(alpha_q, beta_q)
    pyro.sample("latent_fairness", dist.Beta(alpha_q, beta_q))

def main():
    # setup the optimizer
    adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
    optimizer = Adam(adam_params)

    # setup the inference algorithm
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    # Approximation steps
    losses, a, b = [], [], []
    num_steps = 4000

    # do gradient steps
    n_steps = 100
    for step in range(n_steps):
        losses.append(svi.step(data))
        a.append(pyro.param("alpha_q").item())
        b.append(pyro.param("beta_q").item())
        if step % 100 == 0:
            print('.', end='')

    plt.plot(losses)
    plt.title("ELBO")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.show()
    print('a = ', pyro.param("alpha_q").item())
    print('b = ', pyro.param("beta_q").item())

    # grab the learned variational parameters
    alpha_q = pyro.param("alpha_q").item()
    beta_q = pyro.param("beta_q").item()

    # here we use some facts about the beta distribution
    # compute the inferred mean of the coin's fairness
    # This is based on beta distribution mean and std equation
    inferred_mean = alpha_q / (alpha_q + beta_q)
    # compute inferred standard deviation
    factor = beta_q / (alpha_q * (1.0 + alpha_q + beta_q))
    inferred_std = inferred_mean * math.sqrt(factor)

    print("\nbased on the data and our prior belief, the fairness " +
          "of the coin is %.3f +- %.3f" % (inferred_mean, inferred_std))


if __name__== "__main__":
    main()