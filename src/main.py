from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, RationalQuadratic, Matern

import data
import learn.blr as blr
import learn.gpr as gpr
import learn.auto_kernel_gpr as auto_kernel_gpr

import numpy as np
import random

WHITE_WINE_FILENAME = '../data/winequality-white.csv'
RED_WINE_FILENAME = '../data/winequality-red.csv'

PERCENT_TRAINING = 1.0

def trainAndValidate(fileName):
    trainingData, valData = data.parseAndSplit(fileName,
                                               ';',
                                               PERCENT_TRAINING)
    trainingX, trainingY = trainingData
    valX, valY = valData

    linearFit = blr.fit(trainingX, trainingY)
    gprFit = gpr.trainGaussianProcess(trainingX, trainingY, ConstantKernel()*RBF())

    linearPredict = linearFit.predict(valX)
    gprPredict = gprFit.predict(valX)

    linearMse = ((linearPredict - valY) ** 2).mean()
    gpMse = ((gprPredict - valY) ** 2).mean()

    return linearMse, gpMse

def reportMSE(fileName, autoKernelGpr):
    trainingData, valData = data.parseAndSplit(fileName,
                                               ';',
                                               PERCENT_TRAINING)
    trainingX, trainingY = trainingData
    valX, valY = valData

    linearFit = blr.fit(trainingX, trainingY)
    gprFits = [gpr.trainGaussianProcess(trainingX, trainingY, kernel)\
            for kernel, _ in autoKernelGpr.bestKernelsAtEachLevel]
    
    linearPredict = linearFit.predict(valX)
    gprPredicts = [gprFit.predict(valX) for gprFit in gprFits]

    linearMse = ((linearPredict - valY) ** 2).mean()
    gpMses = [((gprPredict - valY) ** 2).mean() for gprPredict in gprPredicts]
    
    print("Linear model gives MSE: ",linearMse)

    for i in range(len(autoKernelGpr.bestKernelsAtEachLevel)):
        print(autoKernelGpr.bestKernelsAtEachLevel[i][0]," gives MSE: ",\
                gpMses[i])

    return linearMse, gpMses

def main():
    random.seed(0)
    np.random.seed(0)

    X, y = data.parseData(RED_WINE_FILENAME)
    autoKernelGpr = auto_kernel_gpr.AutoKernelGpr([RBF, ConstantKernel, WhiteKernel, RationalQuadratic, Matern], X, y)
    kernel = autoKernelGpr.searchForRounds(10)
    # gp = gpr.trainGaussianProcess(X, y)

    print(kernel)
    print(autoKernelGpr.bestKernelsAtEachLevel)

    reportMSE(RED_WINE_FILENAME,autoKernelGpr)


if __name__ == '__main__':
    main()
