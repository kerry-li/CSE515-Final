import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor as gpr

# Class for automated model selection described by the method at
# https://arxiv.org/pdf/1302.4922.pdf.
class AutoKernelGpr:

    def __init__(self, baseKernels, X, y):
        self.baseKernels = baseKernels
        self.X = X
        self.y = y
        self.bestKernelsAtEachLevel = []

    # Returns the trained GPR with the optimal kernel found after searching
    # for /rounds/ rounds.
    def searchForRounds(self, rounds):
        kernel = None
        for round in range(rounds):
            print('Starting round {}'.format(round))
            kernel = self.search(kernel)
            print('Best kernel of round {} is {}'.format(round, kernel))
            self.bestKernelsAtEachLevel.append(kernel)
        return kernel

    def search(self, currentKernel):
        augmentedKernels = []
        for baseKernel in self.baseKernels:
            augmentedKernels.extend(self.augment(currentKernel, baseKernel))

        return self.argmaxBayesianInformationCriterion(augmentedKernels)

    def augment(self, currentKernel, baseKernel):
        # Start with base kernels.
        if not currentKernel:
            return [baseKernel()]

        newKernel = baseKernel()
        composedAddKernel = currentKernel + newKernel
        composedMultKernel = currentKernel * newKernel
        return [composedAddKernel, composedMultKernel]

    # Returns the kernel from /kernels/ with the highest Bayesian Information
    # Criterion.
    def argmaxBayesianInformationCriterion(self, kernels):
        bestKernel = None
        maxBic = -np.inf
        for kernel in kernels:
            print('Evaluating kernel {}'.format(kernel))
            bic, optimizedKernel = self.bayesianInformationCriterion(kernel)
            if bic > maxBic:
                bestKernel = optimizedKernel
                maxBic = bic
        return bestKernel

    # I think the idea is to construct a gp, fit it to some (X,y),
    # get gp.log_marginal_evidence(), then return BIC
    def bayesianInformationCriterion(self, kernel):
        # Selects optimal hyperparameters with a random restarting search.
        # The paper restarts only the newly introduced parameters.
        # TODO: Only restart newly introduced parameters.
        gp = gpr(kernel=kernel, n_restarts_optimizer=0).fit(self.X, self.y)
        n, _ = self.X.shape
        return gp.log_marginal_likelihood() - kernel.n_dims / 2 * np.log(n), gp.kernel_

