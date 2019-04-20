import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor as gpr

# Class for automated model selection described by the method at
# https://pdfs.semanticscholar.org/cc27/639170d87581e2e1ecdc4dca3716915619d2.pdf
# using only GPR priors on the latent function.
class AutoKernelGpr:
    def __init__(self, baseKernels, X, y):
        self.baseKernels = baseKernels
        self.X = X
        self.y = y

    # Returns the trained GPR with the optimal kernel found after searching
    # for /rounds/ rounds.
    def searchForRounds(self, rounds):
        kernel = None
        for round in range(rounds):
            kernel = self.search(kernel)
        return kernel

    def search(self, currentKernel):
        augmentedKernels = []
        for kernel in self.baseKernels:
            augmentedKernels.extend(self.augment(currentKernel, kernel))

        return self.argmaxBayesianInformationCriterion(augmentedKernels)

    def augment(self, currentKernel, kernel):
        if not currentKernel:
            return [kernel]

        def composedAddKernel(x, y, params):
            return currentKernel(x, y,
                                 params[:currentKernel.numParams]) + kernel(
                x, y, params[currentKernel.numParams:kernel.numParams])

        def composedMultKernel(x, y, params):
            return currentKernel(x, y, params[
                                       :currentKernel.numParams]) * kernel(
                x, y, params[currentKernel.numParams:kernel.numParams])

        totalParams = currentKernel.numParams + kernel.numParams
        composedAddKernel.numParams = totalParams
        composedMultKernel.numParams = totalParams
        return [composedAddKernel, composedMultKernel]

    def argmaxBayesianInformationCriterion(self, kernels):
        bestKernel = None
        maxModelEvidence = -np.inf
        for kernel in kernels:
            evidence = self.bayesianInformationCriterion(kernel)
            if evidence > maxModelEvidence:
                bestKernel = kernel
                maxModelEvidence = evidence
        return bestKernel

    # I think the idea is to construct a gp, fit it to some (X,y),
    # get gp.log_marginal_evidence(), then return BIC
    def bayesianInformationCriterion(self, kernel):
        gp = gpr(kernel=kernel, n_restarts_optimizer=10).fit(self.X, self.y)
        n, _ = self.X.shape
        return gp.log_marginal_likelihood() - kernel.numParams / 2 * np.log(n)

