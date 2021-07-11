#Reference: https://pyro.ai/examples/svi_part_i.html#ELBO

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
    alpha_q = torch.tensor(10.0)
    beta_q = torch.tensor(10.0)
    f = pyro.sample("latent_var", dist.Beta(alpha_q,beta_q))
    for i in range(len(data)):
        pyro.sample(f'obs_{i}', dist.Binomial(probs=f), obs=data[i])

def guide(data):
    alpha_q = pyro.param("alpha_q", torch.tensor(15.0),
                         constraint=constraints.positive)
    beta_q = pyro.param("beta_q", torch.tensor(15.0),
                        constraint=constraints.positive)
    pyro.sample("latent_var", dist.Beta(alpha_q, beta_q))

def main():


    optimizer = Adam({"lr": 0.0005, "betas": (0.90, 0.999)})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    losses, a, b = [], [], []
    gradient_steps = 100
    for step in range(gradient_steps):
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
    # calculate the inferred parameters
    inferred_mean = alpha_q / (alpha_q + beta_q)
    factor = beta_q / (alpha_q * (1.0 + alpha_q + beta_q))
    inferred_std = inferred_mean * math.sqrt(factor)

    print("\nBased on the data and our prior belief, the probability of succeeding " +
          "is %.3f , %.3f" % (inferred_mean, inferred_std))


if __name__== "__main__":
    main()