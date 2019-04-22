from sklearn.gaussian_process.kernels import RBF, ConstantKernel

import data
import learn.blr as blr
import learn.gpr as gpr
import learn.auto_kernel_gpr as auto_kernel_gpr

import numpy as np
import random

WHITE_WINE_FILENAME = '../data/winequality-white.csv'
RED_WINE_FILENAME = '../data/winequality-red.csv'

PERCENT_TRAINING = 0.8

def trainAndValidate(fileName):
    trainingData, valData = data.parseAndSplit(fileName,
                                               ';',
                                               PERCENT_TRAINING)
    trainingX, trainingY = trainingData
    valX, valY = valData

    linearFit = blr.fit(trainingX, trainingY)
    gprFit = gpr.trainGaussianProcess(trainingX, trainingY)

    linearPredict = linearFit.predict(valX)
    gprPredict = gprFit.predict(valX)

    whiteLinearMse = ((linearPredict - valY) ** 2).mean()
    whiteGpMse = ((gprPredict - valY) ** 2).mean()

    return whiteLinearMse, whiteGpMse

def main():
    random.seed(0)
    np.random.seed(0)

    X, y = data.parseData(RED_WINE_FILENAME)
    autoKernelGpr = auto_kernel_gpr.AutoKernelGpr([RBF, ConstantKernel], X, y)
    kernel = autoKernelGpr.searchForRounds(5)
    # gp = gpr.trainGaussianProcess(X, y)
    print(kernel)
    print(autoKernelGpr.bestKernelsAtEachLevel)



if __name__ == '__main__':
    main()