from conjugates import InferenceEngine
from distributions.GammaDistribution import GammaExponential
from utils import plot_gamma_pdf
import copy
import numpy as np

class Gamma(InferenceEngine.Engine):
    pass

def experiment_pdf(prior, data):
    engine = Gamma()
    pdfs = []
    # Add the uniform
    pdfs.append(prior)
    # Inference
    more_pdfs = engine.inference(prior, data)
    pdfs = pdfs + more_pdfs
    return pdfs


a0=2.1
b0=1.0
gamma = GammaExponential(alpha=a0,beta=b0)
data=[5]
pdfs = experiment_pdf(gamma,data)
# Plot
x = np.linspace(0, 10, 100)
plot_gamma_pdf(pdfs, x, 'b')