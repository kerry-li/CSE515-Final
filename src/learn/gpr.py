# Gaussian process regression.

from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.gaussian_process.kernels import RBF

# Class for automated model selection described by the method at
# https://pdfs.semanticscholar.org/cc27/639170d87581e2e1ecdc4dca3716915619d2.pdf
# using only GPR priors on the latent function.
class AutoKernelGpr:

    def __init__(self, baseKernels):
        self.baseKernels = baseKernels

    # Returns the trained GPR with the optimal kernel found after searching
    # for /rounds/ rounds.
    def searchForRounds(self, rounds):
        pass

def trainGaussianProcess(X, y, kernel=None):
    # If the kernel parameter is None, then 1.0 * RBF(1.0) is used.
    gp = gpr(kernel=kernel)
    return gp.fit(X,y)

def main():
    from src.data import parseData
    X, y = parseData('../../data/winequality-red.csv')
    trainGaussianProcess(X, y)

if __name__ == '__main__':
    main()
