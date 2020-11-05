import distributions.NormalDistribution as nd
import linear_regression.Normal_b1 as nb

def sum_error(y, x, b0):
    err = lambda y, x, b1: x*(y - b0)
    result = sum([err(y[i], x[i], b0) for i in range(len(x))])
    return result

def sum_x(x):
    sum_f = lambda x:pow(x,2)
    result = sum(sum_f(x[i]) for i in range(len(x)))
    return result

class Normal(nd.Normal):

    def update(self, y, x, tu_trial, b0_trial):
        tu1=self.var
        mu1=self.mean
        print("Child class {}, {}".format(tu1,mu1))
        nominator=tu1*mu1+tu_trial*sum_error(y, x, b0_trial)
        denominator=tu1+tu_trial*sum_x(x)
        mu1_trial=nominator/denominator
        tu1_trial=denominator
        return nb.Normal(prior_mean=mu1_trial,prior_var=tu1_trial)
