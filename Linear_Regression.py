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

    ################################################################################
    # FIT                                                                          #
    #                                                                              #
    # The following is the method that calculates the optimal weights for the      #
    # regression. It will either solve via the derivative solution, or with the    #
    # method of gradient descent.                                                  #
    ################################################################################
    def fit(self, data, targets, return_line=False, gradient_descent=False, eta=0.001, precision=0.0001):
        # Use your derivatives here to find the optimal solution.
        # 'data' here is the data, and 'targets' will be the y = 5x + 3 you create in your main.
        # Return the line of best fit if return_line=True.
        dimensionality = (len(data), )
        if data.shape != dimensionality:
            raise Exception("Please ensure the data has dimensionality of 1")

        data = np.array(data)
        targets = np.array(targets)

        if gradient_descent:
            self.denominator = np.dot(data, data) - data.mean() * data.sum()
            curr_alpha = np.random.randn()
            curr_beta = np.random.randn()
            step_alpha = 999999
            step_beta = 999999
            iteration = 0
            while step_alpha > precision and step_beta > precision:
                self.alpha = curr_alpha
                self.beta = curr_beta
                curr_alpha = self.alpha - eta * (-2 / len(data) * (np.dot(data, targets) - self.alpha*np.dot(data, data) - self.beta * data.sum()))
                curr_beta = self.beta - eta * (-2 / len(data) * (targets.sum() - self.alpha * data.sum() - len(data) * self.beta))
                step_alpha = abs(curr_alpha - self.alpha)
                step_beta = abs(curr_beta - self.beta)
                if iteration > 10**6:
                    print("Max iterations reached, returning estimate.")
                    break
                iteration += 1
            self.line = self.alpha * data + self.beta

        else:
            self.denominator = np.dot(data, data) - data.mean() * data.sum()
            self.alpha = (np.dot(data, targets) - targets.mean() * data.sum()) / self.denominator
            self.beta = (targets.mean() * np.dot(data, data) - data.mean() * np.dot(data, targets)) / self.denominator
            self.line = self.alpha * data + self.beta
        if return_line:
            return self.line

    ################################################################################
    # COEFFICIENTS                                                                 #
    #                                                                              #
    # The following returns the optimal coefficients of the regression line.       #
    ################################################################################
    def coefficients(self):
        # Use this method to return alpha and beta if called.
        if self.line is None:
            raise Exception("Coefficients cannot be calculated if the model hasn't been fit.")
        return self.alpha, self.beta

    ################################################################################
    # ACCURACY                                                                     #
    #                                                                              #
    # The following returns the R-Squared score of the regression line.            #
    ################################################################################
    def accuracy(self, targets):
        # Use this method to calculate the and return the R^2.
        if self.line is None:
            raise Exception("Accuracy cannot be calculated if the model hasn't been fit.")
        numerator = ((targets - self.line) ** 2).sum()
        denominator = ((targets - targets.mean()) ** 2).sum()
        r_squared = 1 - numerator / denominator
        return r_squared

    #################################################################################
    # PLOT                                                                          #
    #                                                                               #
    # The following returns a plot of the regression line versus the original data. #
    #################################################################################
    def plot(self, data, targets):
        # Use this method to produce a plot of the line of best fit vs a scatter plot of the data.
        if self.line is None:
            raise Exception("Plot cannot be constructed if the model hasn't been fit.")
        plt.scatter(data, targets, label="Data")
        plt.plot(data, self.line, color='red', label="Regression Line")
        plt.title("Regression VS Data")
        plt.xlabel("Data")
        plt.ylabel("Targets")
        plt.legend()
        plt.show()
