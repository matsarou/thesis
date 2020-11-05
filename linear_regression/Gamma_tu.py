import distributions.GammaDistribution as gd
import linear_regression.Gamma_tu as gt

def sum_error(y, x, b0, b1):
    err = lambda y, x, b0, b1: pow((y - (b0 +b1 * x)), 2)
    result = sum([err(y[i], x[i], b0, b1) for i in range(len(x))])
    return result

class Gamma(gd.Gamma):

    def update(self, num, y, x, b0_trial, b1_trial):
        a0=self.alpha
        b0=self.beta
        print("Child class {}, {}".format(a0,b0))
        a_trial=a0+num/2
        b_trial=b0+sum_error(y, x, b0_trial, b1_trial)/2
        return gt.Gamma(a=a_trial, b=b_trial)

