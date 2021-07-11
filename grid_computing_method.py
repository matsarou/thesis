import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

# Taken from book:
# Bayesian analysis with python, by Osvaldo Martin
def posterior_grid(grid_points=50, heads=6, tails=9):
    """
    A grid implementation for the coin-flipping problem
    """
    grid = np.linspace(0, 1, grid_points)
    prior = np.repeat(1/grid_points, grid_points)  # uniform prior
    likelihood = stats.binom.pmf(heads, heads+tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior


def main():


    data = np.repeat([0, 1], (10, 3))
    points = 10
    h = data.sum()
    t = len(data) - h
    grid, posterior = posterior_grid(points, h, t)
    plt.plot(grid, posterior, 'o-')

    plt.title(f'heads = {h}, tails = {t}')
    # plt.yticks([])
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'posterior')
    plt.legend(loc=0)
    plt.show()

if __name__ == "__main__":
    main()


