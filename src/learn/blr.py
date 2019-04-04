# Bayesian linear regression.

from sklearn.linear_model import BayesianRidge

def fit(X, y):
    return BayesianRidge().fit(X, y)

def main():
    from src.data import parseData
    X, y = parseData('../../data/winequality-red.csv')
    fit(X, y)

if __name__ == '__main__':
    main()
