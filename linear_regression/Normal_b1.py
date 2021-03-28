import distributions.NormalDistribution as nd

def sum_error(y, x, b0):
    err = lambda y, x, b1: x*(y - b0)
    result = sum([err(y[i], x[i], b0) for i in range(len(x))])
    return result

def sum_x(x):
    sum_f = lambda x:pow(x,2)
    result = sum(sum_f(x[i]) for i in range(len(x)))
    return result

class Normal(nd.NormalLogNormalKnownPrecision):

    def update(self, y, x, tu_trial, b0_trial):
        tu1=self.known_var
        mu1=self.mean
        error=sum_error(y, x, b0_trial)
        nominator=tu1*mu1+tu_trial*error
        denominator=tu1+tu_trial*sum_x(x)
        mu1_trial=nominator/denominator
        tu1_trial=denominator
        return Normal(known_var=tu1_trial, prior_mean=mu1_trial)

