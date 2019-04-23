import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class MultipleRegression:

    ################################################################################
    # Constructor                                                                  #
    #                                                                              #
    # The following is the constructor for the linear regression model.            #
    ################################################################################
    def __init__(self):
        self.line = None
        self.regression_coefficients = None
        self.data = None
        self.targets = None

    ################################################################################
    # FIT                                                                          #
    #                                                                              #
    # The following is the method that calculates the optimal weights for the      #
    # regression.                                                                  #
    ################################################################################
    def fit(self, data, targets, return_line=False):
        self.data = np.array(data)
        self.targets = np.array(targets)
        weights = np.linalg.solve(np.dot(self.data.T, self.data), np.dot(self.data.T, self.targets))
        y_hat = np.dot(data, weights)
        self.line = y_hat
        self.regression_coefficients = weights
        if return_line:
            return self.line

    ################################################################################
    # COEFFICIENTS                                                                 #
    #                                                                              #
    # The following returns the optimal coefficients of the regression.            #
    ################################################################################
    def coefficients(self):
        if self.coefficients is None:
            raise Exception("Fit method must be called first.")
        else:
            return self.regression_coefficients

    ################################################################################
    # ACCURACY                                                                     #
    #                                                                              #
    # The following returns the R-Squared score of the regression.                 #
    ################################################################################
    def accuracy(self):
        if self.line is None:
            raise Exception("Accuracy cannot be calculated if the model hasn't been fit.")
        else:
            d1 = self.targets - self.line
            d2 = self.targets - self.targets.mean()
            r_squared = 1 - d1.dot(d1) / d2.dot(d2)
            return r_squared

    #################################################################################
    # PLOT                                                                          #
    #                                                                               #
    # The following returns a plot of the regression versus the original data if    #
    # the data is two dimensional.                                                  #
    #################################################################################
    def plot(self):
        if self.line is None or self.data.shape != (len(self.data), 2):
            raise Exception("Plot cannot be constructed if the model hasn't been fit.")
        else:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(self.data[:, 0], self.data[:, 1], self.targets, c="blue", label="Data")
            ax.scatter(self.data[:, 0], self.data[:, 1], self.line, c="red", label="Regression Predictions")
            plt.title("Regression VS Data")
            plt.legend()
            plt.show()

