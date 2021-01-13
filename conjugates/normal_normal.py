from conjugates import InferenceEngine
from distributions.NormalDistribution import NormalNormalKnownVar
from utils import plot_normal_pdf, random_samples
import matplotlib.pyplot as plt
import copy

class Normal(InferenceEngine.Engine):
    pass




def plot(x, mu, stdev):
    normal = NormalNormalKnownVar(stdev, mu[0], stdev)
    label = 'mean=' + str(normal.mean) + ', var=' + str(normal.var)
    plt.plot(x, normal.pdf(x), alpha=1.0, color='orange', label=label)
    plt.ylabel('Density')
    plt.xlabel('Hypotheses for μ')
    plt.legend(numpoints=1, loc='upper right')
    plt.show()

def experiment_pdf(mu, tu0, tu, data, color):
    engine = Normal()
    normal = NormalNormalKnownVar(tu0, mu, tu)
    pdfs = []
    # Add the uniform
    pdfs.append(copy.copy(normal))
    # Inference
    more_pdfs = engine.inference(normal, data)
    pdfs = pdfs + more_pdfs
    # Plot
    plot_normal_pdf(pdfs, mu, data, color)

