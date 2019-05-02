import numpy as np

from scipy.optimize import minimize
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor as gpr


# Class for automated model selection described by the method at
# https://arxiv.org/pdf/1302.4922.pdf.
class AutoKernelGpr:
    def __init__(self, baseKernels, X, y, optimizerRounds):
        self.baseKernels = baseKernels
        self.X = X
        self.y = y
        self.bestModelsAtEachLevel = []
        self.optimizerRounds = optimizerRounds

    # Returns the trained GPR with the optimal kernel found after searching
    # for /rounds/ rounds.
    def searchForRounds(self, rounds):
        kernel = None
        for round in range(rounds):
            print('Starting round {}'.format(round))
            bestModel, bic = self.search(kernel)
            kernel = bestModel.kernel_
            print('Best kernel of round {} is {}'.format(round, kernel))
            self.bestModelsAtEachLevel.append((bestModel, bic))
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

    # Returns a gpr trained with a kernel from /kernels/ with the highest
    # Bayesian Information Criterion.
    def argmaxBayesianInformationCriterion(self, kernels):
        if self.bestModelsAtEachLevel:
            oldKernelOptimalParams = self.bestModelsAtEachLevel[-1][0].kernel_.theta
        else:
            oldKernelOptimalParams = np.empty(0)

        bestModel = None
        maxBic = -np.inf
        for kernel in kernels:
            print('Evaluating kernel {}'.format(kernel))
            # Selects optimal hyperparameters with a random restarting search.
            # The paper restarts only the newly introduced parameters.
            # TODO: Only restart newly introduced parameters.
            gp = gpr(kernel=kernel, optimizer=None).fit(self.X, self.y)
            self.optimizeKernelParams(gp,
                                      oldKernelOptimalParams,
                                      gp.kernel_.theta[oldKernelOptimalParams.size:],
                                      self.optimizerRounds)
            bic = self.bayesianInformationCriterion(gp)
            if bic > maxBic:
                bestModel = gp
                maxBic = bic
        return bestModel, maxBic

    def bayesianInformationCriterion(self, gp):
        n, _ = self.X.shape
        return gp.log_marginal_likelihood() - gp.kernel_.n_dims / 2 * np.log(
            n)

    # Returns the optimal params for /gp/.kernel_ using conjugate gradients.
    def optimizeKernelParams(self, gp, oldKernelOptimalParams,
                             newParamsInitialValues,
                             rounds=1):
        # We want to maximize LML, so we minimize negative LML.
        def negativeLml(theta):
            lml, grad = gp.log_marginal_likelihood(theta, eval_gradient=True)
            return -lml, -grad

        newParamsBounds = gp.kernel_.bounds[oldKernelOptimalParams.size:]
        newParamsLowerBounds = [min for min, _ in newParamsBounds]
        newParamsUpperBounds = [max for _, max in newParamsBounds]

        minValue = np.inf
        paramsInitial = np.concatenate([oldKernelOptimalParams, newParamsInitialValues])
        optimalParams = paramsInitial
        for round in range(rounds):
            optimizeResult = minimize(negativeLml, jac=True, x0=paramsInitial, bounds=gp.kernel_.bounds)
            if not optimizeResult.success:
                print(optimizeResult.message)
            if optimizeResult.fun < minValue:
                minValue = optimizeResult.fun
                optimalParams = optimizeResult.x
            newParamsInitial = np.random.uniform(newParamsLowerBounds,
                                                 newParamsUpperBounds)
            paramsInitial = np.concatenate([oldKernelOptimalParams, newParamsInitial])
        gp.kernel_.theta = optimalParams
        gp.log_marginal_likelihood_value_ = -minValue

