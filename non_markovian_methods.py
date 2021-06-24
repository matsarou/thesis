import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

def posterior_grid_approx(grid_points, heads, tosses):
    grid = np.linspace(0, 1, grid_points)
    prior = np.repeat(5, grid_points)
    likelihood = stats.binom.pmf(heads, tosses, grid)
    unstd_posterior = likelihood * prior
    posterior = unstd_posterior / unstd_posterior.sum()
    return grid, posterior


def main():
    points = 15
    h, n = 1, 4
    grid, posterior = posterior_grid_approx(points, h, n)
    plt.plot(grid, posterior, 'o-', label='heads = {}\ntosses = {}'.format(h, n))
    plt.xlabel(r'$\theta$')
    plt.legend(loc=0)
    plt.show()

if __name__ == "__main__":
    main()


