import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:

    ################################################################################
    # Constructor                                                                  #
    #                                                                              #
    # The following is the constructor for the linear regression model.            #
    ################################################################################
    def __init__(self):
        self.line = None
        self.alpha = None
        self.beta = None
        self.denominator = None
        self.data = None
        self.targets = None

    ################################################################################
    # FIT                                                                          #
    #                                                                              #
    # The following is the method that calculates the optimal weights for the      #
    # regression. It will either solve via the derivative solution, or with the    #
    # method of gradient descent.                                                  #
    ################################################################################
    def fit(self, data, targets, return_line=False, gradient_descent=False, eta=0.001, precision=0.0001):
        dimensionality = (len(data), )
        if data.shape != dimensionality:
            raise Exception("Please ensure the data has dimensionality of 1")

        self.data = np.array(data)
        self.targets = np.array(targets)

        if gradient_descent:
            self.denominator = np.dot(self.data, self.data) - self.data.mean() * self.data.sum()
            curr_alpha = np.random.randn()
            curr_beta = np.random.randn()
            step_alpha = 999999
            step_beta = 999999
            iteration = 0
            while step_alpha > precision and step_beta > precision:
                self.alpha = curr_alpha
                self.beta = curr_beta
                curr_alpha = self.alpha - eta * (-2 / len(self.data) * (np.dot(self.data, self.targets) - self.alpha*np.dot(self.data, self.data) - self.beta * self.data.sum()))
                curr_beta = self.beta - eta * (-2 / len(self.data) * (self.targets.sum() - self.alpha * self.data.sum() - len(self.data) * self.beta))
                step_alpha = abs(curr_alpha - self.alpha)
                step_beta = abs(curr_beta - self.beta)
                if iteration > 10**6:
                    print("Max iterations reached, returning estimate.")
                    break
                iteration += 1
            self.line = self.alpha * self.data + self.beta

        else:
            self.denominator = np.dot(self.data, self.data) - self.data.mean() * self.data.sum()
            self.alpha = (np.dot(self.data, self.targets) - self.targets.mean() * self.data.sum()) / self.denominator
            self.beta = (self.targets.mean() * np.dot(self.data, self.data) - self.data.mean() * np.dot(self.data, self.targets)) / self.denominator
            self.line = self.alpha * self.data + self.beta
        if return_line:
            return self.line

    ################################################################################
    # COEFFICIENTS                                                                 #
    #                                                                              #
    # The following returns the optimal coefficients of the regression line.       #
    ################################################################################
    def coefficients(self):
        if self.line is None:
            raise Exception("Coefficients cannot be calculated if the model hasn't been fit.")
        return self.alpha, self.beta

    ################################################################################
    # ACCURACY                                                                     #
    #                                                                              #
    # The following returns the R-Squared score of the regression line.            #
    ################################################################################
    def accuracy(self):
        if self.line is None:
            raise Exception("Accuracy cannot be calculated if the model hasn't been fit.")
        numerator = ((self.targets - self.line) ** 2).sum()
        denominator = ((self.targets - self.targets.mean()) ** 2).sum()
        r_squared = 1 - numerator / denominator
        return r_squared

    #################################################################################
    # PLOT                                                                          #
    #                                                                               #
    # The following returns a plot of the regression line versus the original data. #
    #################################################################################
    def plot(self):
        if self.line is None:
            raise Exception("Plot cannot be constructed if the model hasn't been fit.")
        plt.scatter(self.data, self.targets, label="Data")
        plt.plot(self.data, self.line, color='red', label="Regression Line")
        plt.title("Regression VS Data")
        plt.xlabel("Data")
        plt.ylabel("Targets")
        plt.legend()
        plt.show()
