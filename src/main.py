import random
import numpy as np

from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, \
    RationalQuadratic, Matern
from sklearn.gaussian_process import GaussianProcessRegressor

import data
import learn.blr as blr
import learn.gpr as gpr
import learn.auto_kernel_gpr as auto_kernel_gpr


WHITE_WINE_FILENAME = '../data/winequality-white.csv'
RED_WINE_FILENAME = '../data/winequality-red.csv'

PERCENT_TRAINING = 0.8


def mse(valX, valY, model):
    predict = model.predict(valX)
    return ((predict - valY) ** 2).mean()


def main():
    random.seed(0)
    np.random.seed(0)

    trainingX, trainingY, valX, valY = data.parseAndSplit(RED_WINE_FILENAME)

    autoKernelGpr = auto_kernel_gpr.AutoKernelGpr(
        [RBF, ConstantKernel, WhiteKernel, RationalQuadratic, Matern], trainingX, trainingY)
    kernel = autoKernelGpr.searchForRounds(5)
    print('Best kernel: {}'.format(kernel))
    for model, bic in autoKernelGpr.bestModelsAtEachLevel:
        print(model.kernel_, bic)
        print(mse(valX, valY, model))


if __name__ == '__main__':
    main()
