import numpy as np

# Class for automated model selection described by the method at
# https://pdfs.semanticscholar.org/cc27/639170d87581e2e1ecdc4dca3716915619d2.pdf
# using only GPR priors on the latent function.
class AutoKernelGpr:
    def __init__(self, baseKernels):
        self.baseKernels = baseKernels

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

        return self.argmaxModelEvidence(augmentedKernels)

    def augment(self, currentKernel, kernel):
        if not currentKernel:
            return [kernel]
        composedAddKernel = lambda x, y, params: currentKernel(x, y, params[
                                                                     :currentKernel.numParams]) + kernel(
            x, y, params[currentKernel.numParams:kernel.numParams])
        composedMultKernel = lambda x, y, params: currentKernel(x, y, params[
                                                                      :currentKernel.numParams]) * kernel(
            x, y, params[currentKernel.numParams:kernel.numParams])
        totalParams = currentKernel.numParams + kernel.numParams
        composedAddKernel.numParams = totalParams
        composedMultKernel.numParams = totalParams
        return [composedAddKernel, composedMultKernel]

    def argmaxModelEvidence(self, kernels):
        bestKernel = None
        maxModelEvidence = -np.inf
        for kernel in kernels:
            evidence = self.modelEvidence(kernel)
            if evidence > maxModelEvidence:
                bestKernel = kernel
                maxModelEvidence = evidence
        return bestKernel

    def modelEvidence(self, kernel):
        pass

