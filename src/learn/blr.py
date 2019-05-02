# Bayesian linear regression.

import numpy as np

from sklearn.linear_model import BayesianRidge

def fit(X, y):
    return BayesianRidge(compute_score = True).fit(X, y)

def bic(linearFit, n, d):
    linearBIC = linearFit.scores_[-1] - (d + 2) / 2 * np.log(n)


def main():
    from src.data import parseData
    X, y = parseData('../../data/winequality-red.csv')
    fit(X, y)

if __name__ == '__main__':
    main()
