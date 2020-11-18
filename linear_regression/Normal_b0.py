import distributions.NormalDistribution as nd
import linear_regression.Normal_b0 as nb

def sum_error(y, x, b1):
    err = lambda y, x, b1: y - b1 * x
    result = sum([err(y[i], x[i], b1) for i in range(len(x))])
    return result

class Normal(nd.Normal):

    def update(self, n, y, x, tu_trial, b1_trial):
        tu0=self.var
        mu0=self.mean
        print("Child class {}, {}".format(tu0,mu0))
        nominator=tu0*mu0+tu_trial*sum_error(y, x, b1_trial)
        denominator=tu0+n*tu_trial
        mu0_trial=nominator/denominator
        tu0_trial=denominator
        return nb.Normal(prior_mean=mu0_trial,prior_var=tu0_trial)

