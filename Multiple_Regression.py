import numpy as np
import matplotlib.pyplot as plt


class MultipleRegression:

    ################################################################################
    # Constructor                                                                  #
    #                                                                              #
    # The following is the constructor for the linear regression model.            #
    ################################################################################
    def __init__(self):
        self.line = None
        self.coefficients = None

    ################################################################################
    # FIT                                                                          #
    #                                                                              #
    # The following is the method that calculates the optimal weights for the      #
    # regression.                                                                  #
    ################################################################################
    def fit(self, data, targets, return_line=False):
        weights = np.linalg.solve(np.dot(data.T, data), np.dot(data.T, targets))
        y_hat = np.dot(data, weights)
        self.line = y_hat
        self.coefficients = weights
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
            return self.coefficients

    ################################################################################
    # ACCURACY                                                                     #
    #                                                                              #
    # The following returns the R-Squared score of the regression.                 #
    ################################################################################
    def accuracy(self, targets):
        if self.line is None:
            raise Exception("Accuracy cannot be calculated if the model hasn't been fit.")
        else:
            d1 = targets - self.line
            d2 = targets - targets.mean()
            r_squared = 1 - d1.dot(d1) / d2.dot(d2)
            return r_squared

    #################################################################################
    # PLOT                                                                          #
    #                                                                               #
    # The following returns a plot of the regression versus the original data if    #
    # the data is two dimensional.                                                  #
    #################################################################################
    def plot(self, data, targets):
        if self.line is None and data.shape != (len(data), 2):
            raise Exception("Plot cannot be constructed if the model hasn't been fit.")
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(data[:, 0], data[:, 1], targets, color="blue", label="Data")
            ax.plot(data[:, 0], data[:, 1], self.line, color="red", label="Regression Line")
            plt.show()























