# Gaussian process regression.

from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.gaussian_process.kernels import RBF

def trainGaussianProcess(X, y, kernel=None):
    # If the kernel parameter is None, then 1.0 * RBF(1.0) is used.
    gp = gpr(kernel=kernel, optimizer=None)
    gp.fit(X,y)
    return gp