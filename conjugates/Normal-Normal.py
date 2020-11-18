from conjugates import InferenceEngine
from distributions.NormalDistribution import NormalNormalKnownVar
from utils import normal_data, plot_normal_pdf, random_samples
import copy

class Normal(InferenceEngine.Engine):
    pass

#Define alternative hypotheses for the mean.
mu = random_samples(10,15,10)
tu=0.25
stdev = 2

#Gather data
data=normal_data(mu[0], stdev, size=500)



def experiment_pdf(data, color):
    engine = Normal()
    normal = NormalNormalKnownVar(tu, mu[0], tu)
    pdfs = []
    # Add the uniform
    pdfs.append(copy.copy(normal))
    # Inference
    more_pdfs = engine.inference(normal, data)
    pdfs = pdfs + more_pdfs
    # Plot
    plot_normal_pdf(pdfs, mu[0], color)

experiment_pdf(data, 'b')