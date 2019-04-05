import data
import learn.blr as blr
import learn.gpr as gpr

import numpy as np
import random

WHITE_WINE_FILENAME = '../data/winequality-white.csv'
RED_WINE_FILENAME = '../data/winequality-red.csv'

PERCENT_TRAINING = 0.8

def trainAndValidate(fileName):
    trainingData, valData = data.parseAndSplit(WHITE_WINE_FILENAME,
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

    print(trainAndValidate(WHITE_WINE_FILENAME))
    print(trainAndValidate(RED_WINE_FILENAME))


if __name__ == '__main__':
    main()